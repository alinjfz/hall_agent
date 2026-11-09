import os

from langchain_openai import ChatOpenAI


DEFAULTS = {"openai": "gpt-6-luna", "vertex": "gemini-3.1-flash-lite", "lmstudio": "gemma"}


def configured_provider():
    return os.getenv("HALL_PROVIDER", "openai")


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
        if not (os.getenv("HALL_LM_STUDIO_MODEL") or os.getenv("HALL_MODEL")):
            return "Set HALL_LM_STUDIO_MODEL to the model ID shown in LM Studio."
        return None
    return "Set HALL_PROVIDER to openai, vertex, or lmstudio."


def openai_model(name):
    return ChatOpenAI(model=name, use_responses_api=True, timeout=30, max_retries=1)


def triage_model():
    provider = configured_provider()
    if issue := setup_issue(provider):
        raise ValueError(issue)
    name = model_id(provider, os.getenv("HALL_MODEL"))
    if provider == "openai":
        return openai_model(name)

    if provider == "lmstudio":
        qwen = name.casefold().startswith("qwen/qwen3")
        return ChatOpenAI(
            model=name, base_url=os.getenv("HALL_LM_STUDIO_BASE_URL") or "http://127.0.0.1:1234/v1",
            api_key=os.getenv("HALL_LM_STUDIO_API_KEY") or "lm-studio",
            use_responses_api=False, temperature=0, timeout=60, max_retries=0, max_tokens=650,
            extra_body={"chat_template_kwargs": {"enable_thinking": False},
                        "reasoning_effort": "none"} if qwen else None,
        )

    from google.auth import default
    from langchain_google_genai import ChatGoogleGenerativeAI

    credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    return ChatGoogleGenerativeAI(
        model=name, vertexai=True, credentials=credentials,
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
        timeout=30, max_retries=1,
    )
