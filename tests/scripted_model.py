from itertools import chain, repeat
from time import monotonic, sleep

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from halldesk.replies import ReplyDraft
from halldesk.safety import SafetyCheck


class ScriptedModel(GenericFakeChatModel):
    safety: SafetyCheck = SafetyCheck(hazard=None, evidence=None)

    def with_structured_output(self, schema, *, include_raw=False, tools=None, **kwargs):
        if schema is SafetyCheck:
            return RunnableLambda(lambda _: self.safety)
        if schema is ReplyDraft:
            return RunnableLambda(lambda _: None)

        def answer(messages):
            raw = self.invoke(messages)
            if raw.tool_calls and raw.tool_calls[0]["name"] == schema.__name__:
                parsed = schema.model_validate(raw.tool_calls[0]["args"])
                return {"raw": None, "parsed": parsed} if include_raw else parsed
            return {"raw": raw, "parsed": None}
        return RunnableLambda(answer)


def wait_for_triage(ui, store, *report_ids):
    deadline = monotonic() + 10
    while not all(store.report(report_id)["proposal"] for report_id in report_ids):
        assert monotonic() < deadline, f"Triage did not finish for reports {report_ids}"
        sleep(0.02)
        # AppTest has no browser timer to trigger Streamlit's periodic fragment.
        ui.run(timeout=30)
        assert not ui.exception, ui.exception
    ui.run(timeout=30)


def scripted(location, topic, interpretation, *, safety=None):
    interpretation = dict(interpretation)
    if interpretation.get("intent") == "new":
        interpretation.setdefault("category", topic)
    query = {"escalation": "escalation chasing repair", "lighting": "fixed lighting",
             "heating": "heating faults", "plumbing": "tap shower leak"}.get(topic, topic)
    policy = AIMessage(content="", tool_calls=[
        {"name": "search_policy", "args": {"query": query}, "id": "policy"},
    ])
    answer = AIMessage(content="", tool_calls=[
        {"name": "Interpretation", "args": interpretation, "id": "interpret"},
    ])
    return ScriptedModel(messages=chain([policy], repeat(answer)),
                         safety=SafetyCheck.model_validate(safety) if safety else SafetyCheck(hazard=None, evidence=None))
