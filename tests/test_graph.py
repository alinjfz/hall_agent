from time import time

import httpx
import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from openai import APIConnectionError

from unihelp import graph
from unihelp.context import retrieve_policy
from unihelp.replies import grounded_reply
from unihelp.safety import SafetyCheck
from unihelp.store import ReviewError, Store
from unihelp.workflow import review, triage


@pytest.mark.parametrize("assessment", ["routine", "uncertain", "conflicting"])
def test_combined_assessment_uses_independent_safety_only_when_needed(tmp_path, monkeypatch, assessment):
    from unihelp import workflow
    from unihelp.interpretation import Interpretation

    routine = assessment == "routine"
    message = "My radiator is cold." if routine else "My radiator is cold and I can smell gas."
    store = Store(tmp_path / "workspace.sqlite3")
    store.seed(samples=False)
    report_id = store.add_report("Flat 5", "Repair", message)
    calls = []

    class Model:
        def with_structured_output(self, schema, **kwargs):
            calls.append(schema)
            if schema is graph.CombinedAssessment:
                return RunnableLambda(lambda _: graph.CombinedAssessment(
                    safety=SafetyCheck(hazard=None, evidence=None,
                                       assessment="uncertain" if assessment == "uncertain" else "not_current"),
                    interpretation=Interpretation(intent="new", category="heating", fitting="radiator",
                                                  evidence="My radiator is cold"),
                ))
            assert schema is SafetyCheck
            return RunnableLambda(lambda _: SafetyCheck(hazard="gas", evidence="I can smell gas.",
                                                        assessment="current"))

    monkeypatch.setattr(workflow, "triage_model", lambda **_: Model())
    proposal = triage(store, report_id)
    assert not store.jobs()
    if routine:
        assert calls == [graph.CombinedAssessment]
        assert proposal["kind"] == "create" and "policy repair target is 2 days" in proposal["reply"]
    else:
        assert calls == [graph.CombinedAssessment, SafetyCheck]
        assert proposal["kind"] == "manual" and proposal["urgency"] == "emergency"


def test_isolated_detector_chirp_needs_no_model_call(tmp_path):
    store = Store(tmp_path / "workspace.sqlite3")
    store.seed(samples=False)
    report_id = store.add_report("Flat 5", "Alarm", "My smoke detector keeps chirping every few minutes.")
    calls = []

    class Model:
        def with_structured_output(self, schema, **kwargs):
            calls.append(schema)
            return RunnableLambda(lambda _: SafetyCheck(hazard="fire", evidence=store.report(report_id)["message"]))

    proposal = triage(store, report_id, model=Model())
    assert proposal["urgency"] == "urgent"
    assert proposal["signals"] == ["fire_equipment"]
    assert not calls


def test_laundry_hours_requires_relevant_handbook_coverage(tmp_path, monkeypatch):
    from tests.scripted_model import scripted

    monkeypatch.setattr(graph, "search_handbook", lambda _: [
        Document(page_content="Use the posted quiet hours.", metadata={"id": "quiet-hours"})
    ])
    store = Store(tmp_path / "workspace.sqlite3")
    store.seed(samples=False)
    message = "What time does the laundry room close on Sundays?"
    report_id = store.add_report("Flat 5", "Laundry", message)
    model = scripted("Flat 5", "question", {"intent": "question", "evidence": "close on Sundays?"},
                     draft={"body": "Use the hours posted in your building.", "section_ids": ["quiet-hours"]})

    proposal = triage(store, report_id, model=model, draft_replies=True)
    assert proposal["kind"] == "manual"
    assert proposal["reason"] == "The handbook does not list the requested laundry hours."


def test_vertex_rate_limit_cooldown_applies_to_other_requests(monkeypatch):
    from types import SimpleNamespace

    from unihelp import models

    now = 0.0
    attempts = []

    def sleep(seconds):
        nonlocal now
        now += seconds

    class RateLimit(Exception):
        code = 429
        response = SimpleNamespace(headers={"retry-after": "5"})

    def generate(*args, **kwargs):
        attempts.append(now)
        if len(attempts) == 1:
            raise RateLimit
        return "ok"

    monkeypatch.setenv("UNIHELP_VERTEX_REQUEST_INTERVAL", "2")
    monkeypatch.setattr(models, "_vertex_next_request", 0.0)
    monkeypatch.setattr(models, "_vertex_rate_limits", 0)
    monkeypatch.setattr(models, "monotonic", lambda: now)
    monkeypatch.setattr(models, "sleep", sleep)
    monkeypatch.setattr(models.ChatGoogleGenerativeAI, "_generate", generate)

    with pytest.raises(RateLimit):
        models.VertexChatModel.model_construct(timeout=4)._generate([])
    assert models.vertex_rate_limits() == 1
    peer = models.VertexChatModel.model_construct(timeout=10)
    assert peer._generate([]) == "ok"
    assert peer._generate([]) == "ok"
    assert attempts == [0.0, 5.0, 7.0]


def test_private_lighting_fault_is_not_a_dark_shared_route(tmp_path):
    from tests.scripted_model import scripted

    message = "My fixed ceiling light broke. There is no other lighting in the bedroom."
    store = Store(tmp_path / "workspace.sqlite3")
    store.seed(samples=False)
    report_id = store.add_report("Flat 5 bedroom", "Lighting", message)
    model = scripted("Flat 5 bedroom", "lighting", {
        "intent": "new", "fitting": "fixed ceiling light", "evidence": "My fixed ceiling light broke.",
        "other_lighting_works": False, "lighting_evidence": "There is no other lighting in the bedroom.",
    }, safety={"hazard": "dark_route", "evidence": "There is no other lighting in the bedroom.",
               "assessment": "current"})

    proposal = triage(store, report_id, model=model)
    assert proposal["kind"] == "create" and proposal["priority"] == "P2"
    assert "urgency" not in proposal


def test_lost_key_with_unknown_access_prompts_for_clarification(tmp_path):
    from tests.scripted_model import scripted

    store = Store(tmp_path / "workspace.sqlite3")
    store.seed(samples=False)
    message = "I lost my key. Can reception help?"
    report_id = store.add_report("Flat 5", "Access", message)
    model = scripted("Flat 5", "access", {
        "intent": "access", "category": "spare_key", "evidence": "I lost my key.",
    })

    assert triage(store, report_id, model=model)["kind"] == "access_question"


@pytest.mark.parametrize(("message", "hazard", "evidence"), [
    ("I smell gas in the kitchen.", "gas", "I smell gas in the kitchen."),
    ("there is a fire. i'm not joking!", "fire", "there is a fire."),
])
def test_model_confirms_emergency_and_records_only_one_reply(tmp_path, message, hazard, evidence):
    from tests.scripted_model import scripted

    store = Store(tmp_path / "workspace.sqlite3")
    store.seed(samples=False)
    report_id = store.add_report("Flat 1", "Emergency", message)
    report = store.report(report_id)
    assert report["proposal"] is None
    assert not store.activity(report_id)
    model = scripted("Flat 1", "heating", {"intent": "manual"},
                     safety={"assessment": "current", "hazard": hazard, "evidence": evidence})
    assert triage(store, report_id, model=model)["urgency"] == "emergency"
    report = store.report(report_id)
    assert store.jobs() == []
    assert [row["event"] for row in store.activity(report_id)] == ["emergency_reply", "proposal_ready"]
    result = review(store, report_id, report["version"], "acknowledge", reviewer_id="duty-team")
    assert result["reviewer_id"] == "duty-team"
    assert sum(row["event"] == "emergency_reply" for row in store.activity(report_id)) == 1


@pytest.mark.parametrize(("message", "evidence"), [
    ("everything is burning! sorry i was joking", "sorry i was joking"),
    ("everything is burning. i was joking!", "i was joking!"),
    ("i am not being serious there is a fire", "i am not being serious"),
])
def test_explicitly_retracted_fire_claim_is_spam(tmp_path, message, evidence):
    from tests.scripted_model import scripted

    store = Store(tmp_path / "workspace.sqlite3")
    store.seed(samples=False)
    report_id = store.add_report("Flat 1", "Report", message)
    model = scripted("Flat 1", "heating", {"intent": "manual"},
                     safety={"assessment": "spam", "hazard": None, "evidence": evidence})
    proposal = triage(store, report_id, model=model)
    assert proposal["kind"] == "spam"
    assert proposal.get("urgency") is None and not proposal.get("reply")
    assert [row["event"] for row in store.activity(report_id)] == ["proposal_ready"]
    assert review(store, report_id, store.report(report_id)["version"], "acknowledge")["action"] == "spam"


def test_spam_label_cannot_hide_later_hazard_or_post_automatic_reply(tmp_path):
    from tests.scripted_model import scripted

    store = Store(tmp_path / "workspace.sqlite3")
    store.seed(samples=False)
    report_id = store.add_report("Flat 1", "Report", "I was joking. The kitchen bin is on fire now.")
    model = scripted("Flat 1", "heating", {"intent": "manual"},
                     safety={"assessment": "spam", "hazard": None, "evidence": "I was joking."})
    proposal = triage(store, report_id, model=model)
    assert proposal["urgency"] == "emergency" and "unverified_safety" in proposal["signals"]
    assert [row["event"] for row in store.activity(report_id)] == ["proposal_ready"]


def test_staff_feedback_enables_live_routing_without_guidance(tmp_path, monkeypatch):
    import json

    from unihelp import workflow
    from unihelp.interpretation import Interpretation

    class Model:
        def with_structured_output(self, schema, **kwargs):
            assert schema is graph.CombinedAssessment

            def answer(messages):
                payload = json.loads(messages[-1].content)
                assert "staff_examples" not in payload
                message = payload["resident_report"]
                hazard = "I can smell gas." if "gas" in message else None
                return graph.CombinedAssessment(
                    safety=SafetyCheck(assessment="current" if hazard else "not_current",
                                       hazard="gas" if hazard else None, evidence=hazard),
                    interpretation=None if hazard else Interpretation(
                        intent="out_of_scope" if payload["out_of_scope_enabled"] else "manual",
                        evidence=message),
                )

            return RunnableLambda(answer)

    monkeypatch.setattr(workflow, "triage_model", lambda **_: Model())
    store = Store(tmp_path / "workspace.sqlite3")
    store.seed(samples=False)
    source = store.add_report("Flat 2", "Kitchen", "Where can I find good food in central London?")
    assert triage(store, source)["kind"] == "manual"
    report = store.report(source)
    store.save_eval_draft(report, {"kind": "spam"}, "Outside residence support.",
                          as_test=True, enable_out_of_scope=True)
    assert store.feedback_case(report)["status"] == "holdout"
    store.requeue_review(source, report["version"])
    assert triage(store, source)["kind"] == "spam"
    unseen = store.add_report("Flat 2", "Kitchen", "Can you recommend a sushi restaurant?")
    assert triage(store, unseen)["kind"] == "spam"
    mixed = store.add_report("Flat 2", "Kitchen", "Any restaurant tips? I can smell gas.")
    assert triage(store, mixed)["urgency"] == "emergency"


@pytest.mark.parametrize("label", ["routine", "urgent", "emergency", "spam"])
def test_staff_label_rule_applies_to_new_reports(tmp_path, label):
    from tests.scripted_model import scripted

    store = Store(tmp_path / "workspace.sqlite3")
    store.seed(samples=False)
    source = store.add_report("Flat 2", "Report", "Original report for this condition.")
    triage(store, source, model=scripted("Flat 2", None, {"intent": "manual"}))
    store.set_report_label(source, store.report(source)["version"], label,
                           routing_condition="Resident reports the test condition.")
    rule, = store.staff_routing_rules()
    report_id = store.add_report("Flat 3", "Report", "The test condition is happening here too.")
    facts = {"intent": "manual", "staff_rule_id": rule["id"], "staff_rule_evidence": "The test condition"}
    proposal = triage(store, report_id, model=scripted("Flat 3", None, facts))
    assert proposal["kind"] == ("spam" if label == "spam" else "manual")
    assert proposal.get("urgency") == (label if label in {"urgent", "emergency"} else None)
    assert "Matches staff routing rule" in proposal["reason"]
    assert not any(event["event"] == "emergency_reply" for event in store.activity(report_id))

    hazard = store.add_report("Flat 3", "Report", "The test condition. I can smell gas.")
    model = scripted("Flat 3", None, facts, safety={"assessment": "current", "hazard": "gas",
                                                    "evidence": "I can smell gas."})
    proposal = triage(store, hazard, model=model)
    assert proposal["urgency"] == "emergency"
    assert proposal["signals"] == ["gas"]


@pytest.mark.parametrize("check", [
    None,
    APIConnectionError(request=httpx.Request("POST", "http://127.0.0.1:1234/v1/chat/completions")),
    SafetyCheck(hazard="fire", evidence="fabricated quote", assessment="current"),
    SafetyCheck(hazard="fire", evidence="my radiator is cold", assessment="current"),
    SafetyCheck(hazard=None, evidence=None, assessment="uncertain"),
    SafetyCheck(hazard="fire", evidence="My radiator is cold.", assessment="not_current"),
], ids=["timeout", "connection", "fabricated", "changed-case", "uncertain", "inconsistent"])
def test_unverified_safety_goes_to_urgent_review(tmp_path, check):
    from unihelp.interpretation import Interpretation

    class Model:
        def with_structured_output(self, schema, **kwargs):
            def answer(_):
                if schema is SafetyCheck:
                    if check is None:
                        raise TimeoutError
                    if isinstance(check, APIConnectionError):
                        raise check
                    return check
                return {"parsed": Interpretation(intent="new", category="heating", fitting="radiator",
                                                 evidence="radiator is cold")}
            return RunnableLambda(answer)

    store = Store(tmp_path / "workspace.sqlite3")
    store.seed(samples=False)
    report_id = store.add_report("Flat 1", "Repair", "My radiator is cold.")
    proposal = triage(store, report_id, model=Model())
    assert proposal["kind"] == "manual" and proposal["urgency"] == "urgent"
    assert proposal["signals"] == ["unverified_safety"]
    if isinstance(check, APIConnectionError):
        assert "retry when the model service is available" in proposal["reason"]
    assert store.report(report_id)["triage_attempts"] == 0
    assert not store.jobs()


def test_rejecting_legacy_proposal_creates_unlabelled_quality_draft(tmp_path):
    store = Store(tmp_path / "workspace.sqlite3")
    store.seed(samples=False)
    report_id = store.add_report("Flat 1", "Repair", "My radiator is cold.")
    store.save_proposal(store.report(report_id), {"kind": "manual", "reason": "Unclear."})
    assert review(store, report_id, store.report(report_id)["version"], "reject")["action"] == "rejected"
    draft, = store.eval_cases("draft")
    assert draft["expected"] == {}
    assert draft["snapshot"]["report_id"] == report_id
    assert draft["snapshot"]["review_change"]["after"] == "rejected"


def test_approved_guidance_retires_after_its_rule_changes(tmp_path):
    store = Store(tmp_path / "workspace.sqlite3")
    store.seed(samples=False)
    report_id = store.add_report("Flat 1", "Repair", "My radiator is cold.")
    store.save_eval_draft(store.report(report_id), {"kind": "create", "category": "heating", "priority": "P2", "days": 2},
                          "The radiator is explicitly identified.",
                          interpretation={"intent": "new", "category": "heating", "fitting": "radiator", "evidence": "radiator"})
    store.decide_eval(1, "guidance")
    policies = retrieve_policy()
    assert len(store.active_guidance(policies)) == 1
    policies["heating"]["revision"] += 1
    assert store.active_guidance(policies) == []
    assert store.eval_cases("retired")[0]["id"] == 1


def test_paraphrased_holdout_cannot_become_guidance(tmp_path):
    store = Store(tmp_path / "workspace.sqlite3")
    store.seed(samples=False)
    for message in ("The tap is dripping.", "My faucet keeps dripping."):
        report_id = store.add_report("Flat 1", "Repair", message)
        store.save_eval_draft(store.report(report_id), {"kind": "create", "category": "plumbing", "priority": "P3", "days": 14},
                              "A contained drip.", interpretation={"intent": "new", "category": "plumbing"})
    store.decide_eval(1, "holdout")
    with pytest.raises(ReviewError, match="similar report"):
        store.decide_eval(2, "guidance")


@pytest.mark.parametrize(("body", "sources", "valid"), [
    ("Repair #41 is due on 12 November 2026 at 12:00.", ["chasing"], True),
    ("Repair #42 is due on 12 November 2026 at 12:00.", ["chasing"], False),
    ("Repair #41 is due on 13 November 2026.", ["chasing"], False),
    ("Your repair will be done tomorrow.", ["chasing"], False),
    ("Repair #41 is still open.", ["unretrieved"], False),
])
def test_reply_rejects_invented_facts(body, sources, valid):
    job = {"id": 41, "deadline": "2026-11-12T12:00:00+00:00"}
    assert grounded_reply(body, sources, ["chasing"], job) is valid


def test_transient_model_failure_retries_with_one_deadline_without_creating_a_job(tmp_path, monkeypatch):
    from unihelp.models import TransientModelError
    from unihelp.interpretation import Interpretation

    deadlines = []
    delays = []
    interpret_report = graph.interpret_report

    def track_deadline(factory, report, context, expires):
        deadlines.append(expires)
        return interpret_report(factory, report, context, expires)

    monkeypatch.setattr(graph, "interpret_report", track_deadline)
    monkeypatch.setattr(graph, "sleep", delays.append)

    class FlakyModel:
        calls = 0

        def with_structured_output(self, schema, **kwargs):
            if schema is SafetyCheck:
                return RunnableLambda(lambda _: SafetyCheck(hazard=None, evidence=None))
            def answer(_):
                self.calls += 1
                if self.calls == 1:
                    raise TransientModelError("temporary")
                return {"raw": None, "parsed": Interpretation(
                    intent="new", category="heating", fitting="radiator", evidence="radiator is cold")}
            return RunnableLambda(answer)

    store = Store(tmp_path / "workspace.sqlite3")
    store.seed(samples=False)
    report_id = store.add_report("Flat 1", "Repair", "My radiator is cold.")
    proposal = triage(store, report_id, model=FlakyModel())
    assert proposal["kind"] == "create"
    assert store.report(report_id)["triage_attempts"] == 2
    assert len(deadlines) == 2 and deadlines[0] == deadlines[1]
    assert delays == [0.3]
    assert not store.jobs()



def test_negated_fitting_never_matches_the_lone_job(tmp_path):
    from tests.scripted_model import scripted

    store = Store(tmp_path / "workspace.sqlite3")
    store.seed()
    original = store.report(1)
    message = "Following up on the wall sconce, not the ceiling light repair #101."
    with store.transaction(write=True) as db:
        db.execute("UPDATE reports SET message=? WHERE id=1", (message,))
    model = scripted(original["location"], "escalation", {"intent": "chase", "evidence": message,
                                                         "candidate_ids": [101]})
    assert triage(store, 1, model=model)["kind"] == "manual"
    assert not store.jobs()[0]["escalated_at"]



def test_staff_can_review_after_proposal_save_is_interrupted(tmp_path, monkeypatch):
    from tests.scripted_model import scripted

    store = Store(tmp_path / "workspace.sqlite3")
    store.seed(samples=False)
    report_id = store.add_report("Flat 1", "Repair", "My radiator is cold.")
    report = store.report(report_id)
    model = scripted(report["location"], "heating", {
        "intent": "new", "fitting": "radiator", "evidence": "radiator is cold",
    })
    save = store.save_proposal

    def interrupted(*args, **kwargs):
        save(*args, **kwargs)
        raise RuntimeError("Stopped after SQLite committed")

    with monkeypatch.context() as patch:
        patch.setattr(store, "save_proposal", interrupted)
        with pytest.raises(RuntimeError, match="Stopped after SQLite committed"):
            triage(store, report_id, model=model)
    proposal_version = store.report(report_id)["version"]
    receipt = review(store, report_id, proposal_version, "approve")
    assert receipt["action"] == "created_job"
    assert review(store, report_id, proposal_version, "approve") == receipt
    assert len(store.jobs()) == 1


def test_saved_handbook_reply_is_not_redrafted_after_interrupted_save(tmp_path, monkeypatch):
    from tests.scripted_model import scripted

    store = Store(tmp_path / "workspace.sqlite3")
    store.seed(samples=False)
    report_id, report = question_report(store)
    model = scripted(report["location"], "question", {
        "intent": "question", "evidence": report["message"],
    }, draft={"body": "Use the quiet hours posted in your building.", "section_ids": ["quiet-hours"]})
    save = store.save_proposal
    draft = graph.draft_with_search
    calls = []

    def tracked_draft(*args, **kwargs):
        calls.append(True)
        return draft(*args, **kwargs)

    def interrupted(*args, **kwargs):
        save(*args, **kwargs)
        raise RuntimeError("Stopped after SQLite committed")

    monkeypatch.setattr(graph, "draft_with_search", tracked_draft)
    with monkeypatch.context() as patch:
        patch.setattr(store, "save_proposal", interrupted)
        with pytest.raises(RuntimeError, match="Stopped after SQLite committed"):
            triage(store, report_id, model=model, draft_replies=True)

    saved = store.report(report_id)
    assert saved["proposal"]["reply_sources"] == ["quiet-hours"]
    assert review(store, report_id, saved["version"], "acknowledge")["action"] == "answer"
    assert len(calls) == 1
    assert sum(row["event"] == "proposal_ready" for row in store.activity(report_id)) == 1


def test_interpretation_is_one_call_without_tools(tmp_path):
    import json

    from unihelp.interpretation import Interpretation

    store = Store(tmp_path / "workspace.sqlite3")
    store.seed()
    report = store.report(1)
    context = graph.retrieve_context(store, report)
    assert context["jobs"]
    context["job_reports"] = {101: {"message": "private report text"}}
    seen = []

    class Model:
        def with_structured_output(self, schema, *, tools=None, **kwargs):
            def answer(messages):
                seen.append(tools)
                payload = json.loads(messages[-1].content)
                assert payload["open_jobs"] == context["jobs"]
                assert "private report text" not in messages[-1].content
                return {"parsed": Interpretation(intent="chase", candidate_ids=[99999])}
            return RunnableLambda(answer)

    parsed, nominated = graph.interpret_report(lambda **_: Model(), report, context, time() + 20)
    assert seen == [None]
    assert nominated == [99999] and parsed.candidate_ids == []


def draft_model(drafts, bindings=None):
    """Return each scripted draft response in turn and record the tools bound per call."""
    replies = iter(drafts)

    class Model:
        def with_structured_output(self, schema, *, tools=None, **kwargs):
            def answer(_):
                if bindings is not None:
                    bindings.append(tools)
                return next(replies)
            return RunnableLambda(answer)
    return lambda **_: Model()


SECTION = {"id": "quiet-hours", "title": "Quiet hours and shared spaces", "text": "Use the posted quiet hours."}


@pytest.mark.parametrize(("failure", "call_count"), [("unknown", 1), ("too_many", 1), ("extra_round", 2)])
def test_draft_search_loop_fails_closed(failure, call_count):
    from langchain_core.messages import AIMessage

    bindings = []

    def tool_calls():
        return {"raw": AIMessage(content="", tool_calls=[
            {"name": "unknown" if failure == "unknown" else "search_policy",
             "args": {"query": "heating faults"}, "id": f"r{len(bindings)}-{index}"}
            for index in range(graph.MAX_TOOL_CALLS + 1 if failure == "too_many" else 1)])}

    class Calls:
        def __iter__(self):
            return self

        def __next__(self):
            return tool_calls()

    draft, shown = graph.draft_with_search(draft_model(Calls(), bindings), time() + 20,
                                           {"report": "My radiator is cold.", "job": None}, [SECTION])
    assert draft is None
    assert len(bindings) == call_count
    if failure == "extra_round":
        assert bindings[0] and bindings[1] is None  # The last round cannot call tools.
        assert {section["id"] for section in shown} > {"quiet-hours"}
    else:
        assert shown == [SECTION]


def test_draft_can_cite_a_section_found_by_its_own_search():
    from langchain_core.messages import AIMessage
    from langsmith import trace, tracing_context

    from evals.run import descendants, local_trace_client
    from unihelp.replies import ReplyDraft

    search = {"raw": AIMessage(content="", tool_calls=[
        {"name": "search_policy", "args": {"query": "spare key"}, "id": "lookup"}])}
    final = {"raw": None, "parsed": ReplyDraft(body="A spare key request is routine.", section_ids=["keys-and-access"])}
    with tracing_context(enabled="local", client=local_trace_client()), trace("test") as root:
        draft, shown = graph.draft_with_search(draft_model([search, final]), time() + 20,
                                               {"report": "Can I get a spare key?", "job": None}, [SECTION])
    assert draft.section_ids == ["keys-and-access"]
    assert [section["id"] for section in shown] == ["quiet-hours", "keys-and-access"]
    lookup = next(node for node in descendants(root) if node.name == "search_handbook")
    assert lookup.run_type == "retriever"
    assert lookup.outputs["documents"][0].page_content == shown[1]["text"]


@pytest.mark.parametrize("lookup", [False, True])
def test_vertex_drafts_directly_or_after_a_handbook_lookup(monkeypatch, lookup):
    import json

    from langchain_core.messages import AIMessage, ToolMessage

    from unihelp.models import VertexChatModel
    from unihelp.replies import ReplyDraft

    section = "keys-and-access" if lookup else SECTION["id"]
    fields = {"body": "Ask residence staff for help.", "section_ids": [section]}

    def bind_tools(self, tools, *, tool_choice):
        assert tools == [ReplyDraft, graph.search_policy] and tool_choice == "any"
        name, args = ("search_policy", {"query": "spare key"}) if lookup else ("ReplyDraft", fields)
        return RunnableLambda(lambda _: AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call"}]))

    def generate(self, messages, config=None, **kwargs):
        assert lookup and isinstance(messages[-1], ToolMessage) and messages[-1].tool_call_id == "call"
        return AIMessage(content=json.dumps(fields))

    monkeypatch.setattr(VertexChatModel, "bind_tools", bind_tools)
    monkeypatch.setattr(VertexChatModel, "invoke", generate)
    model = VertexChatModel.model_construct()
    draft, shown = graph.draft_with_search(lambda **_: model, time() + 20,
                                           {"report": "Can I get a spare key?", "job": None}, [SECTION])
    assert draft.section_ids == [section]
    assert section in {passage["id"] for passage in shown}


def question_report(store, message="What are the quiet hours in my building?"):
    report_id = store.add_report("Flat 1", "Question", message)
    return report_id, store.report(report_id)


@pytest.mark.parametrize(("draft", "kind"), [
    ({"body": "Use the quiet hours posted in your building.", "section_ids": ["quiet-hours"]}, "answer"),
    ({"body": "Quiet hours start at 22:00.", "section_ids": ["quiet-hours"]}, "manual"),  # Invented time.
    ({"body": "Use the posted quiet hours.", "section_ids": ["water-leaks"]}, "manual"),  # Unretrieved source.
    ({"body": "I could not find that.", "section_ids": ["quiet-hours"], "covered": False}, "manual"),
    (None, "manual"),
])
def test_handbook_question_needs_a_grounded_answer(tmp_path, draft, kind):
    from tests.scripted_model import scripted

    store = Store(tmp_path / "workspace.sqlite3")
    store.seed(samples=False)
    report_id, report = question_report(store)
    model = scripted(report["location"], "question", {
        "intent": "question", "evidence": "What are the quiet hours in my building?"}, draft=draft)
    proposal = triage(store, report_id, model=model, draft_replies=True)
    assert proposal["kind"] == kind
    assert "quiet-hours" in proposal["retrieval"]["section_ids"]
    if kind == "answer":
        assert proposal["rule"]["id"] == "QUESTION-1" and proposal["reply_sources"] == ["quiet-hours"]
        receipt = review(store, report_id, store.report(report_id)["version"], "acknowledge")
        assert receipt["action"] == "answer"
    else:
        assert proposal["reason"] == graph.UNANSWERED and "reply_sources" not in proposal
    assert not store.jobs()


def test_question_the_handbook_does_not_cover_goes_to_staff(tmp_path):
    from tests.scripted_model import scripted

    store = Store(tmp_path / "workspace.sqlite3")
    store.seed(samples=False)
    message = "Zxqv wvbn?"
    report_id, report = question_report(store, message)
    model = scripted(report["location"], "question", {"intent": "question", "evidence": message},
                     draft={"body": "Yes.", "section_ids": ["quiet-hours"]})
    proposal = triage(store, report_id, model=model, draft_replies=True)
    assert proposal["kind"] == "manual"
    assert proposal["reason"] == "The handbook does not cover this question."
    assert proposal["retrieval"]["section_ids"] == []


def test_evidence_ignores_case_spacing_and_typographic_quotes(tmp_path):
    from unihelp.evidence import quoted_in
    from tests.scripted_model import scripted

    message = "The radiator in my room  isn’t heating."
    assert quoted_in("the radiator in my room isn't heating", message)
    assert not quoted_in("the radiator in my room is cold", message)
    assert not quoted_in("  ", message) and not quoted_in(None, message)
    store = Store(tmp_path / "workspace.sqlite3")
    store.seed(samples=False)
    report_id = store.add_report("Flat 1", "Repair", message)
    model = scripted(store.report(report_id)["location"], "heating", {
        "intent": "new", "fitting": "radiator", "evidence": "the radiator in my room isn't heating"})
    assert triage(store, report_id, model=model)["kind"] == "create"


@pytest.mark.parametrize("interpret_times_out", [False, True])
def test_clean_safety_check_never_raises_an_alert(tmp_path, interpret_times_out):
    from unihelp.interpretation import Interpretation

    class Model:
        def with_structured_output(self, schema, **kwargs):
            def answer(_):
                if schema is SafetyCheck:
                    return SafetyCheck(hazard=None, evidence="")  # Small models return "" for null.
                if interpret_times_out:
                    raise TimeoutError
                return {"parsed": Interpretation(intent="new", category="heating", fitting="radiator",
                                                 evidence="radiator is cold")}
            return RunnableLambda(answer)

    store = Store(tmp_path / "workspace.sqlite3")
    store.seed(samples=False)
    report_id = store.add_report("Flat 1", "Repair", "My radiator is cold.")
    proposal = triage(store, report_id, model=Model())
    assert proposal.get("urgency") is None and not proposal.get("signals")
    assert proposal["kind"] == ("manual" if interpret_times_out else "create")
    if interpret_times_out:
        assert proposal["reason"] == "timed out"
