import os

import httpx

from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from openai import APIConnectionError

DEFAULTS = {"openai": "gpt-6-luna", "vertex": "gemini-3.1-flash-lite", "lmstudio": "qwen/qwen3.5-9b"}


class TransientModelError(RuntimeError):
    pass


class LocalAsyncClient(httpx.AsyncClient):
    """A connection pool owned by one graph invocation."""


async def close_owned_model(model):
    if isinstance(model, ModelWithFallback):
        await close_owned_model(model.primary)
        await close_owned_model(model.fallback)
        return
    client = getattr(model, "http_async_client", None)
    if isinstance(client, LocalAsyncClient):
        await client.aclose()


def transient_error(exc):
    status = getattr(exc, "status_code", None)
    return (isinstance(exc, (TransientModelError, TimeoutError, APIConnectionError))
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


def openai_model(name):
    return ChatOpenAI(model=name, use_responses_api=True, timeout=30, max_retries=1)


def lmstudio_model(name, *, timeout=18):
    qwen = name.casefold().startswith("qwen/qwen3")
    return ChatOpenAI(
        model=name, base_url=os.getenv("HALL_LM_STUDIO_BASE_URL") or "http://127.0.0.1:1234/v1",
        api_key=os.getenv("HALL_LM_STUDIO_API_KEY") or "lm-studio",
        use_responses_api=False, temperature=0, timeout=timeout, max_retries=0, max_tokens=650,
        http_async_client=LocalAsyncClient(timeout=timeout, trust_env=False),
        extra_body={"chat_template_kwargs": {"enable_thinking": False},
                    "reasoning_effort": "none"} if qwen else None,
    )


class ModelWithFallback:
    def __init__(self, primary, fallback):
        self.model_name = primary.model_name
        self.primary = primary
        self.fallback = fallback

    def with_structured_output(self, schema, **kwargs):
        primary = self.primary.with_structured_output(schema, **kwargs)
        fallback = self.fallback.with_structured_output(schema, **kwargs)

        async def invoke(messages, config):
            try:
                return await primary.ainvoke(messages, config)
            except Exception as exc:
                if transient_error(exc):
                    raise TransientModelError(type(exc).__name__) from exc
                raise

        return RunnableLambda(invoke).with_fallbacks([fallback], exceptions_to_handle=(TransientModelError,))


def triage_model():
    provider = configured_provider()
    if issue := setup_issue(provider):
        raise ValueError(issue)
    name = model_id(provider, os.getenv("HALL_MODEL"))
    if provider == "lmstudio":
        primary = lmstudio_model(name)
        fallback = os.getenv("HALL_FALLBACK_MODEL")
        return ModelWithFallback(primary, lmstudio_model(fallback)) if fallback and fallback != name else primary
    if provider == "openai":
        return openai_model(name)

    from google.auth import default
    from langchain_google_genai import ChatGoogleGenerativeAI

    credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    return ChatGoogleGenerativeAI(
        model=name, vertexai=True, credentials=credentials,
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
        timeout=30, max_retries=1,
    )
