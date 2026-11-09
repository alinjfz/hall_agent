import pytest


@pytest.fixture(autouse=True)
def isolate_local_configuration(monkeypatch):
    """AppTest executes app.py in-process; never load a developer's .env."""
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("HALL_BACKGROUND_WORKER", "false")
    monkeypatch.setenv("HALL_CHASE_PASSES", "full")
    monkeypatch.delenv("HALL_POLICY_FILE", raising=False)
