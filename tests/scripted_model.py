from itertools import chain, repeat
from time import monotonic, sleep

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


class ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def wait_for_triage(ui, store, *report_ids):
    deadline = monotonic() + 10
    while not all(store.report(report_id)["proposal"] for report_id in report_ids):
        assert monotonic() < deadline, f"Triage did not finish for reports {report_ids}"
        sleep(0.02)
        # AppTest has no browser timer to trigger Streamlit's periodic fragment.
        ui.run(timeout=30)
        assert not ui.exception, ui.exception
    ui.run(timeout=30)


def scripted(location, topic, interpretation):
    interpretation = dict(interpretation)
    if interpretation.get("intent") == "new":
        interpretation.setdefault("category", topic)
    policy = AIMessage(content="", tool_calls=[
        {"name": "search_rules", "args": {"topic": topic}, "id": "policy"},
    ])
    answer = AIMessage(content="", tool_calls=[
        {"name": "Interpretation", "args": interpretation, "id": "interpret"},
    ])
    return ScriptedModel(messages=chain([policy], repeat(answer)))
