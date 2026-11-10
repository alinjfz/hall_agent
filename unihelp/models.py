import logging
import os
from atexit import register
from functools import lru_cache
from math import isfinite

import httpx
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import APIConnectionError

DEFAULTS = {"openai": "gpt-6-luna", "vertex": "gemini-3.1-flash-lite", "lmstudio": "qwen/qwen3.5-9b"}
# Total tries per Vertex request. The Google SDK backs off and retries 408, 429 and 5xx responses.
VERTEX_MAX_ATTEMPTS = 3


class TransientModelError(RuntimeError):
    pass


def vertex_request_interval():
    try:
        interval = float(os.getenv("UNIHELP_VERTEX_REQUEST_INTERVAL") or "1")
    except ValueError:
        interval = 0
    if not isfinite(interval) or interval <= 0:
        raise ValueError("UNIHELP_VERTEX_REQUEST_INTERVAL must be a positive number of seconds.")
    return interval


@lru_cache(maxsize=4)
def vertex_rate_limiter(interval):
    """One limiter per process, so every Vertex chat model shares the same request pacing."""
    return InMemoryRateLimiter(requests_per_second=1 / interval, check_every_n_seconds=0.1, max_bucket_size=1)


class VertexChatModel(ChatGoogleGenerativeAI):
    """Marks Gemini on Vertex, whose JSON schema output cannot be combined with extra tools."""


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
    return os.getenv("UNIHELP_PROVIDER", "lmstudio")


def model_id(provider, override=None):
    if provider == "lmstudio":
        return override or os.getenv("UNIHELP_LM_STUDIO_MODEL") or DEFAULTS[provider]
    return override or DEFAULTS[provider]


def setup_issue(provider):
    if provider == "openai":
        return None if os.getenv("OPENAI_API_KEY") else "Set OPENAI_API_KEY to start automatic triage."
    if provider == "vertex":
        if not os.getenv("GOOGLE_CLOUD_PROJECT"):
            return "Set GOOGLE_CLOUD_PROJECT and configure Google ADC."
        try:
            vertex_request_interval()
        except ValueError as exc:
            return str(exc)
        return None
    if provider == "lmstudio":
        return None
    return "Set UNIHELP_PROVIDER to lmstudio, openai, or vertex."


def openai_model(name, *, timeout):
    return ChatOpenAI(model=name, use_responses_api=True, timeout=timeout, max_retries=0)


@lru_cache(maxsize=1)
def vertex_api_key_client(api_key):
    from google import genai
    from google.genai.types import HttpOptions

    client = genai.Client(vertexai=True, api_key=api_key, http_options=HttpOptions(api_version="v1"))
    register(client.close)
    return client


@lru_cache(maxsize=1)
def local_http_client():
    client = httpx.Client(trust_env=False)
    register(client.close)
    return client


def lmstudio_model(name, *, timeout=18):
    qwen = name.casefold().startswith("qwen/qwen3")
    return ChatOpenAI(
        model=name, base_url=os.getenv("UNIHELP_LM_STUDIO_BASE_URL") or "http://127.0.0.1:1234/v1",
        api_key=os.getenv("UNIHELP_LM_STUDIO_API_KEY") or "lm-studio",
        use_responses_api=False, temperature=0, timeout=timeout, max_retries=0, max_tokens=650,
        http_client=local_http_client(),
        extra_body={"chat_template_kwargs": {"enable_thinking": False},
                    "reasoning_effort": "none"} if qwen else None,
    )


def embedding_model():
    """Return (embeddings, model name), or (None, None) for lexical-only search.

    Embeddings are opt-in so the quickstart and tests need no embedding service.
    """
    name = os.getenv("UNIHELP_EMBEDDING_MODEL")
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
                model=name, base_url=os.getenv("UNIHELP_LM_STUDIO_BASE_URL") or "http://127.0.0.1:1234/v1",
                api_key=os.getenv("UNIHELP_LM_STUDIO_API_KEY") or "lm-studio",
                check_embedding_ctx_length=False, http_client=local_http_client(), max_retries=0, timeout=10), name
        return OpenAIEmbeddings(model=name, max_retries=1, timeout=10), name
    except Exception as exc:
        logging.warning("Embedding model unavailable (%s); handbook search is lexical only", type(exc).__name__)
        return None, None


def structured_model(model, schema, **kwargs):
    if isinstance(model, VertexChatModel) and (tools := kwargs.get("tools")):
        # Gemini's JSON schema adapter does not accept extra tools. Offer the
        # final draft schema as a function alongside the handbook lookup.
        def parse(message):
            calls = message.tool_calls
            parsed = None
            raw = message
            if len(calls) == 1 and calls[0]["name"] == schema.__name__ and not message.invalid_tool_calls:
                # The graph should only execute real lookup calls.
                raw = None
                try:
                    parsed = schema.model_validate(calls[0]["args"])
                except ValueError:
                    pass
            return {"raw": raw, "parsed": parsed} if kwargs.get("include_raw") else parsed

        return model.bind_tools([schema, *tools], tool_choice="any") | RunnableLambda(parse)

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
    name = model_id(provider, os.getenv("UNIHELP_MODEL"))
    if provider == "lmstudio":
        fallback = os.getenv("UNIHELP_FALLBACK_MODEL")
        if fallback and fallback != name:
            return lmstudio_model(name, timeout=timeout / 2), lmstudio_model(fallback, timeout=timeout / 2)
        return lmstudio_model(name, timeout=timeout)
    if provider == "openai":
        return openai_model(name, timeout=timeout)

    api_key = os.getenv("GOOGLE_API_KEY")
    credentials = None
    if not api_key:
        from google.auth import default

        credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    model = VertexChatModel(
        model=name, vertexai=True, credentials=credentials,
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
        timeout=timeout, max_retries=VERTEX_MAX_ATTEMPTS,
        rate_limiter=vertex_rate_limiter(vertex_request_interval()),
    )
    if api_key:
        model.client = vertex_api_key_client(api_key)
    return model
