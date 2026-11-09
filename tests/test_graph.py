from time import monotonic

import pytest
from langchain_core.runnables import RunnableLambda

from halldesk import graph
from halldesk.context import retrieve_policy
from halldesk.replies import grounded_reply
from halldesk.safety import SafetyCheck
from halldesk.store import ReviewError, Store
from halldesk.workflow import review, triage


def test_emergency_is_ready_at_intake_and_records_only_one_reply(tmp_path):
    store = Store(tmp_path / "workspace.sqlite3")
    store.seed(samples=False)
    report_id = store.add_report("Flat 1", "Emergency", "I smell gas in the kitchen.")
    report = store.report(report_id)
    assert report["proposal"]["urgency"] == "emergency"
    assert not store.claim_triage()
    assert store.jobs() == []
    assert [row["event"] for row in store.activity(report_id)] == ["emergency_reply", "proposal_ready"]
    result = review(store, report_id, report["version"], "acknowledge", reviewer_id="duty-team")
    assert result["reviewer_id"] == "duty-team"
    assert sum(row["event"] == "emergency_reply" for row in store.activity(report_id)) == 1


@pytest.mark.parametrize("bad_evidence", [False, True])
def test_unverified_safety_goes_to_urgent_review(tmp_path, bad_evidence):
    from halldesk.interpretation import Interpretation

    class Model:
        def with_structured_output(self, schema, **kwargs):
            def answer(_):
                if schema is SafetyCheck:
                    if bad_evidence:
                        return SafetyCheck(hazard="fire", evidence="fabricated quote")
                    raise TimeoutError
                return {"parsed": Interpretation(intent="new", category="heating", fitting="radiator",
                                                 evidence="radiator is cold")}
            return RunnableLambda(answer)

    store = Store(tmp_path / "workspace.sqlite3")
    store.seed(samples=False)
    report_id = store.add_report("Flat 1", "Repair", "My radiator is cold.")
    proposal = triage(store, report_id, model=Model())
    assert proposal["kind"] == "manual" and proposal["urgency"] == "urgent"
    assert proposal["signals"] == ["unverified_safety"]
    assert store.report(report_id)["triage_attempts"] == 1
    assert not store.jobs()


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


def test_transient_model_failure_retries_without_creating_a_job(tmp_path):
    from halldesk.models import TransientModelError
    from halldesk.interpretation import Interpretation

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



def test_staff_can_review_after_publish_checkpoint_is_interrupted(tmp_path, monkeypatch):
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


def test_interpretation_receives_jobs_and_defaults_to_handbook_search(tmp_path):
    import json

    from langsmith import trace, tracing_context
    from evals.run import descendants, local_trace_client
    from halldesk.interpretation import Interpretation

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

    with tracing_context(enabled="local", client=local_trace_client()), trace("test") as root:
        parsed, sections, nominated = graph.interpret_with_tools(
            lambda **_: Model(), report, context, monotonic() + 20)
    assert len(seen) == 1 and [tool.name for tool in seen[0]] == ["search_policy"]
    assert nominated == [99999] and parsed.candidate_ids == []
    assert sections[0]["id"] == "chasing-repairs"
    search = next(node for node in descendants(root) if node.name == "search_handbook")
    assert search.run_type == "retriever"
    assert search.outputs["documents"][0]["page_content"] == sections[0]["text"]


@pytest.mark.parametrize(("failure", "call_count"), [("unknown", 1), ("too_many", 1), ("extra_round", 2)])
def test_tool_loop_fails_closed(tmp_path, failure, call_count):
    from langchain_core.messages import AIMessage

    store = Store(tmp_path / "workspace.sqlite3")
    store.seed()
    report = store.report(1)
    bindings = []

    class Model:
        def with_structured_output(self, schema, *, tools=None, **kwargs):
            def answer(messages):
                bindings.append(tools)
                calls = [
                    {"name": "unknown" if failure == "unknown" else "search_policy",
                     "args": {"query": "heating faults"}, "id": f"r{len(bindings)}-{index}"}
                    for index in range(5 if failure == "too_many" else 1)
                ]
                return {"raw": AIMessage(content="", tool_calls=calls)}
            return RunnableLambda(answer)

    parsed, sections, nominated = graph.interpret_with_tools(
        lambda **_: Model(), report, graph.retrieve_context(store, report), monotonic() + 20)
    assert parsed is nominated is None
    assert len(bindings) == call_count
    if failure == "extra_round":
        assert bindings[0] and bindings[1] is None
        assert sections
    else:
        assert not sections
