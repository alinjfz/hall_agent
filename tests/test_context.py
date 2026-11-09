import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx
from langchain_core.messages import AIMessage

from tests.test_scripted_cases import evaluate
from halldesk.context import relevant_examples, retrieve_policy
from halldesk.store import Store
from halldesk.workflow import review, revise, triage
from tests.scripted_model import ScriptedModel, scripted


def test_policy_and_guidance():
    with TemporaryDirectory() as directory:
        store = Store(Path(directory) / "workspace.sqlite3")
        store.seed(samples=False)
        assert store.reports() == []
        guide = store.add_report("Flat 1", "Light clarification", "The wall light stopped working in this room.")
        holdout = store.add_report("Flat 2", "Uncertain repair", "The ceiling light stopped working in this room.")
        store.save_eval_draft(store.report(guide), {"kind": "manual"}, "")
        store.save_eval_draft(store.report(guide), {"kind": "lighting_question"}, "Check the other lighting.",
                              interpretation={"intent": "new", "category": "lighting", "fitting": "wall light",
                                              "evidence": "wall light"})
        assert not relevant_examples(store.eval_cases("guidance"), store.report(holdout)["message"])
        store.decide_eval(1, "guidance")
        store.save_eval_draft(store.report(holdout), {"kind": "lighting_question"}, "")
        store.decide_eval(2, "holdout")

        requests = []

        def respond(request):
            payload = json.loads(request.content)
            requests.append(payload)
            schema = payload["response_format"]["json_schema"]["name"]
            answer = {"hazard": None, "evidence": None} if schema == "SafetyCheck" else {
                "intent": "new", "category": "lighting", "candidate_ids": [], "evidence": "ceiling light",
                "fitting": "ceiling light", "other_lighting_works": None, "lighting_evidence": None,
            }
            return httpx.Response(200, json={
                "id": "local-test", "object": "chat.completion", "created": 1, "model": "test-gemma",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {
                    "role": "assistant", "content": json.dumps(answer),
                }}],
            })

        env = {"HALL_PROVIDER": "lmstudio", "HALL_LM_STUDIO_MODEL": "test-gemma", "HALL_MODEL": "",
               "HALL_LM_STUDIO_BASE_URL": "http://127.0.0.1:1234/v1", "HALL_POLICY_FILE": ""}
        with httpx.Client(transport=httpx.MockTransport(respond)) as client, \
             patch.dict("os.environ", env), \
             patch("halldesk.models.local_http_client", return_value=client):
            case = store.eval_cases("holdout")[0]
            score = evaluate(case["expected"] | {"snapshot": case["snapshot"], "name": case["name"],
                                                 "guidance": store.eval_cases("guidance")})
            assert all(score["checks"].values()), score
            assert score["retrieval"]["example_ids"] == [1]
            assert {"LIGHTING-1", "HEATING-1", "APPLIANCE-1", "ACCESS-1", "MULTIPLE-1"} <= {
                rule["id"] for rule in score["retrieval"]["policies"]
            }
            def interpretations():
                return [request for request in requests
                        if request["response_format"]["json_schema"]["name"] == "Interpretation"]

            request = interpretations()[0]
            assert request["response_format"]["type"] == "json_schema"
            assert request["model"] == "test-gemma"
            assert not request.get("tools")  # Interpretation is one structured call with no tools.
            assert "LIGHTING-1" not in request["messages"][-1]["content"]
            assert "The wall light stopped working" in request["messages"][-1]["content"]
            store.decide_eval(1, "retired")
            second_id = store.add_report("Flat 2", "Another repair", store.report(holdout)["message"])
            result = triage(store, second_id)
            assert not result["retrieval"]["example_ids"]
            assert "The wall light stopped working" not in interpretations()[-1]["messages"][-1]["content"]
            assert store.jobs() == []

        with patch.dict("os.environ", {"HALL_POLICY_FILE": str(Path(directory) / "missing.json")}):
            invalid = triage(store, guide, model=scripted("", "review", {}))
            assert invalid["kind"] == "manual" and invalid["retrieval"]["policies"] == []

        pack = Path(directory) / "replacement.json"
        policies = retrieve_policy()
        policies["lighting"]["other_lighting_works"]["days"] = 9
        pack.write_text(json.dumps(list(policies.values())))
        report_id = store.add_report("Flat 3", "Replace lamp", "Ceiling light broke. Another light still works.")
        with patch.dict("os.environ", {"HALL_POLICY_FILE": str(pack)}):
            proposal = triage(store, report_id, model=scripted(store.report(report_id)["location"], "lighting", {
                "intent": "new", "candidate_ids": [], "evidence": "Ceiling light", "fitting": "Ceiling light",
                "other_lighting_works": True, "lighting_evidence": "Another light still works.",
            }))
        assert proposal["kind"] == "create" and proposal["days"] == 9


def test_duplicate_history_and_model_safety():
    with TemporaryDirectory() as directory:
        store = Store(Path(directory) / "duplicates.sqlite3")
        store.seed(samples=False)
        original_id = store.add_report("Flat 3", "Lighting", "The ceiling bulb above the cooker is broken.")
        original = store.report(original_id)
        proposal = triage(store, original_id, model=scripted(original["location"], "lighting", {
            "intent": "new", "fitting": "ceiling bulb", "evidence": "ceiling bulb",
            "other_lighting_works": None,
        }))
        assert proposal["kind"] == "lighting_question"
        revise(store, original_id, store.report(original_id)["version"], True)
        job = review(store, original_id, store.report(original_id)["version"], "approve")
        for location in ("Flat 3", "Flat 4"):
            report_id = store.add_report(location, "Kitchen light", "The light over the cooker has gone out.")
            report = store.report(report_id)
            result = triage(store, report_id, model=scripted(report["location"], "lighting", {
                "intent": "new", "fitting": "light over the cooker", "evidence": "light over the cooker",
                "candidate_ids": [job["job_id"]],
            }))
            if location == "Flat 3":
                assert result["kind"] == "manual"
                assert result["possible_duplicates"][0]["report_id"] == original_id
            else:
                assert result["kind"] == "lighting_question"
                assert "possible_duplicates" not in result

        def interpretation_message(**fields):
            return AIMessage(content="", tool_calls=[
                {"name": "Interpretation", "args": fields, "id": "interpret"},
            ])

        report_id = store.add_report("Flat 4", "Light", "The ceiling light failed. The wall light still works.")
        first = {"intent": "new", "category": "lighting", "fitting": "ceiling light", "evidence": "ceiling light"}
        second = first | {"other_lighting_works": True, "lighting_evidence": "The wall light still works."}
        before = store.jobs()
        model = ScriptedModel(messages=iter([
            interpretation_message(**first),
            interpretation_message(**second),
            interpretation_message(**first),
        ]))
        assert triage(store, report_id, model=model)["kind"] == "lighting_question"
        assert store.jobs() == before

        report_id = store.add_report("Flat 4", "Unknown fault", "The ceiling light failed.")
        uncategorized = {key: value for key, value in first.items() if key != "category"}
        model = ScriptedModel(messages=iter(interpretation_message(**uncategorized) for _ in range(3)))
        assert triage(store, report_id, model=model)["kind"] == "manual"

        report_id = store.add_report("Flat 4", "Two issues", "The radiator is still cold. The socket is sparking.")
        model = ScriptedModel(messages=iter([]))
        urgent = triage(store, report_id, model=model)
        assert urgent["urgency"] == "urgent"
        assert urgent["followup_suggestions"] == ["The radiator is still cold."]
