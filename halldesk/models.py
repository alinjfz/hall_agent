import logging
import os
from atexit import register
from functools import lru_cache
from random import uniform
from threading import BoundedSemaphore, Lock
from time import monotonic, sleep

import httpx
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import APIConnectionError

DEFAULTS = {"openai": "gpt-6-luna", "vertex": "gemini-3.1-flash-lite", "lmstudio": "qwen/qwen3.5-9b"}
VERTEX_REQUEST_SLOTS = BoundedSemaphore(2)
VERTEX_MAX_ATTEMPTS = 3
VERTEX_REQUEST_INTERVAL = 1.0
_vertex_pacing_lock = Lock()
_vertex_next_request = 0.0


class TransientModelError(RuntimeError):
    pass


def _wait_for_vertex_slot(deadline):
    global _vertex_next_request
    while True:
        with _vertex_pacing_lock:
            now = monotonic()
            if max(now, _vertex_next_request) >= deadline:
                raise TimeoutError("Vertex request budget exhausted while pacing")
            delay = _vertex_next_request - now
            if delay <= 0:
                _vertex_next_request = now + VERTEX_REQUEST_INTERVAL
                return
        sleep(delay)


class VertexChatModel(ChatGoogleGenerativeAI):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        budget = kwargs.pop("timeout", None) or self.timeout or 20
        kwargs["max_retries"] = 1
        deadline = monotonic() + budget
        for attempt in range(VERTEX_MAX_ATTEMPTS):
            if not VERTEX_REQUEST_SLOTS.acquire(timeout=max(0, deadline - monotonic())):
                raise TimeoutError("Timed out waiting for a Vertex request slot")
            try:
                _wait_for_vertex_slot(deadline)
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError("Vertex request budget exhausted")
                return super()._generate(messages, stop=stop, run_manager=run_manager, timeout=remaining, **kwargs)
            except Exception as exc:
                status = getattr(exc, "code", None) or getattr(exc.__cause__, "code", None)
                retryable = status in {408, 429, 500, 502, 503, 504} or isinstance(exc, httpx.TransportError)
                if not retryable or attempt + 1 == VERTEX_MAX_ATTEMPTS:
                    raise
                delay = 0.5 * 2**attempt + uniform(0, 0.5)
                if monotonic() + delay >= deadline:
                    raise
            finally:
                VERTEX_REQUEST_SLOTS.release()
            sleep(delay)


def model_timeout(exc):
    if isinstance(exc, TransientModelError) and exc.__cause__:
        exc = exc.__cause__
    return (isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.casefold()
            or getattr(exc, "status_code", None) in {408, 504})


def transient_error(exc):
    status = getattr(exc, "status_code", None)
    return (isinstance(exc, (TransientModelError, APIConnectionError)) or model_timeout(exc)
            or status == 429 or isinstance(status, int) and status >= 500)


def configured_provider():
    return os.getenv("HALL_PROVIDER", "lmstudio")


def model_id(provider, override=None):
    if provider == "lmstudio":
        return override or os.getenv("HALL_LM_STUDIO_MODEL") or DEFAULTS[provider]
    return override or DEFAULTS[provider]


def setup_issue(provider):
    if provider == "openai":
        return None if os.getenv("OPENAI_API_KEY") else "Set OPENAI_API_KEY to start automatic triage."
    if provider == "vertex":
        return None if os.getenv("GOOGLE_CLOUD_PROJECT") else "Set GOOGLE_CLOUD_PROJECT and configure Google ADC."
    if provider == "lmstudio":
        return None
    return "Set HALL_PROVIDER to lmstudio, openai, or vertex."


def openai_model(name, *, timeout):
    return ChatOpenAI(model=name, use_responses_api=True, timeout=timeout, max_retries=0)


@lru_cache(maxsize=1)
def local_http_client():
    client = httpx.Client(trust_env=False)
    register(client.close)
    return client


def lmstudio_model(name, *, timeout=18):
    qwen = name.casefold().startswith("qwen/qwen3")
    return ChatOpenAI(
        model=name, base_url=os.getenv("HALL_LM_STUDIO_BASE_URL") or "http://127.0.0.1:1234/v1",
        api_key=os.getenv("HALL_LM_STUDIO_API_KEY") or "lm-studio",
        use_responses_api=False, temperature=0, timeout=timeout, max_retries=0, max_tokens=650,
        http_client=local_http_client(),
        extra_body={"chat_template_kwargs": {"enable_thinking": False},
                    "reasoning_effort": "none"} if qwen else None,
    )


def embedding_model():
    """Return (embeddings, model name), or (None, None) for lexical-only search.

    Embeddings are opt-in so the quickstart and tests need no embedding service.
    """
    name = os.getenv("HALL_EMBEDDING_MODEL")
    if not name:
        return None, None
    provider = configured_provider()
    try:
        if provider == "vertex":
            from google.auth import default
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            return GoogleGenerativeAIEmbeddings(
                model=name, vertexai=True, credentials=credentials, project=os.environ["GOOGLE_CLOUD_PROJECT"],
                location=os.getenv("GOOGLE_CLOUD_LOCATION", "global")), name
        if provider == "lmstudio":
            return OpenAIEmbeddings(
                model=name, base_url=os.getenv("HALL_LM_STUDIO_BASE_URL") or "http://127.0.0.1:1234/v1",
                api_key=os.getenv("HALL_LM_STUDIO_API_KEY") or "lm-studio",
                check_embedding_ctx_length=False, http_client=local_http_client(), max_retries=0, timeout=10), name
        return OpenAIEmbeddings(model=name, max_retries=1, timeout=10), name
    except Exception as exc:
        logging.warning("Embedding model unavailable (%s); handbook search is lexical only", type(exc).__name__)
        return None, None


def structured_model(model, schema, **kwargs):
    if not isinstance(model, tuple):
        return model.with_structured_output(schema, **kwargs)

    primary, fallback = (item.with_structured_output(schema, **kwargs) for item in model)

    def invoke(messages, config):
        try:
            return primary.invoke(messages, config)
        except Exception as exc:
            if transient_error(exc):
                raise TransientModelError(type(exc).__name__) from exc
            raise

    return RunnableLambda(invoke).with_fallbacks([fallback], exceptions_to_handle=(TransientModelError,))


def triage_model(*, timeout=20):
    provider = configured_provider()
    if issue := setup_issue(provider):
        raise ValueError(issue)
    name = model_id(provider, os.getenv("HALL_MODEL"))
    if provider == "lmstudio":
        fallback = os.getenv("HALL_FALLBACK_MODEL")
        if fallback and fallback != name:
            return lmstudio_model(name, timeout=timeout / 2), lmstudio_model(fallback, timeout=timeout / 2)
        return lmstudio_model(name, timeout=timeout)
    if provider == "openai":
        return openai_model(name, timeout=timeout)

    from google.auth import default
    credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    return VertexChatModel(
        model=name, vertexai=True, credentials=credentials,
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
        timeout=timeout, max_retries=1,
    )
