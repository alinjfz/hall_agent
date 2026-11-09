import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from halldesk.store import Store
from halldesk.workflow import review, revise, triage
from tests.scripted_model import scripted


NOW = datetime(2026, 11, 8, 12, tzinfo=timezone.utc)


def prepare_case(store, case):
    snapshot = case.get("snapshot")
    store.seed(samples=snapshot is None)
    with store.transaction(write=True) as db:
        if snapshot:
            report_id = db.execute("""INSERT INTO reports (location,message,subject,received_at)
                VALUES (?,?,?,?)""", (snapshot["location"], snapshot["message"],
                                     case["name"], snapshot["at"])).lastrowid
            job_reports = snapshot.get("job_reports", {})
            for job in snapshot["jobs"]:
                original = job_reports.get(str(job["id"]))
                source_id = None
                if original:
                    source_id = db.execute("INSERT INTO reports (location,message) VALUES (?,?)",
                                           (job["location"], original["message"])).lastrowid
                db.execute("""INSERT INTO jobs (id,location,fitting,deadline,priority,status,category,labels,
                    escalated_at,created_from)
                    VALUES (:id,:location,:fitting,:deadline,:priority,:status,:category,:labels,
                    :escalated_at,:created_from)""",
                           job | {"labels": json.dumps(job["labels"]), "created_from": source_id})
            db.executemany("""INSERT INTO eval_cases (id,name,message_key,snapshot,expected,notes,status,interpretation,policy_revisions)
                VALUES (?,?,?,?,?,?,'guidance',?,?)""", [
                (row["id"], row["name"], row["message_key"], json.dumps(row["snapshot"]),
                 json.dumps(row["expected"]), row["notes"], json.dumps(row["interpretation"]),
                 json.dumps(row["policy_revisions"])) for row in case.get("guidance", [])
            ])
            return report_id
        if "message" in case:
            db.execute("UPDATE reports SET message=? WHERE id=?", (case["message"], case["report"]))
        if case.get("boundary"):
            db.execute("UPDATE jobs SET deadline=? WHERE id=101", (NOW.isoformat(),))
        if case.get("job_state") == "closed":
            db.execute("UPDATE jobs SET status='closed' WHERE id=101")
        if case.get("job_state") == "escalated":
            db.execute("UPDATE jobs SET escalated_at=? WHERE id=101", (NOW.isoformat(),))
    return case["report"]


def evaluate(case, model=None):
    with TemporaryDirectory() as directory:
        snapshot = case.get("snapshot")
        now = datetime.fromisoformat(snapshot["at"]) if snapshot else NOW
        store = Store(Path(directory) / "eval.sqlite3", clock=lambda: now)
        report_id = prepare_case(store, case)
        report = store.report(report_id)
        before = store.jobs()
        if model is None and "script" in case:
            model = scripted(report["location"], case["topic"], case["script"], safety=case.get("safety"))
        proposal = triage(store, report_id, model=model)
        actual = proposal["kind"]
        checks = {"route": actual == case["kind"], "triage_is_read_only": store.jobs() == before,
                  "retrieved_policy": bool(proposal["retrieval"]["policies"])}
        if "hazard" in case:
            checks["hazard"] = (proposal.get("urgency") or "none") == case["hazard"]
        if "urgency" in case:
            checks["urgency"] = proposal.get("urgency") == case["urgency"]
        if "reply_contains" in case:
            checks["reply"] = case["reply_contains"] in proposal.get("reply", "")
        if "related_jobs" in case:
            checks["related_jobs"] = [job["id"] for job in proposal.get("related_jobs", [])] == case["related_jobs"]
        if "job" in case:
            checks["job"] = proposal.get("job", {}).get("id") == case["job"]
        if case["kind"] == "create":
            checks["priority"] = proposal.get("priority") == case["priority"]
            checks["days"] = proposal.get("days") == case["days"]
        if "category" in case:
            checks["category"] = proposal.get("category") == case["category"]
        if "issues" in case:
            checks["issues"] = proposal.get("issues") == case["issues"]
        if "candidates" in case:
            checks["candidates"] = [job["id"] for job in proposal.get("candidates", [])] == sorted(case["candidates"])
        if rule_id := case.get("rule_id") or (
            {"escalation": "OVERDUE-1", "lighting": "LIGHTING-1"}.get(case.get("topic"))
            if case["kind"] != "manual" else None
        ):
            checks["rule"] = proposal.get("rule", {}).get("id") == rule_id
            checks["expected_policy_retrieved"] = any(
                rule["id"] == rule_id for rule in proposal["retrieval"]["policies"]
            )
        if "duplicate_jobs" in case:
            checks["duplicate_jobs"] = proposal.get("duplicate_job_ids") == case["duplicate_jobs"]
        if "reply_source" in case:
            source = proposal.get("reply_source", {})
            checks["reply_source"] = source.get("id") == case["reply_source"]
            checks["grounded_reply"] = bool(source.get("summary") and source["summary"] in proposal.get("reply", ""))
        if "choose" in case and actual == "job_question":
            proposal = revise(store, report_id, store.report(report_id)["version"], case["choose"])
            checks["chosen_job"] = proposal.get("job", {}).get("id") == case["choose"]
        if "answer" in case and actual in {"lighting_question", "plumbing_question"}:
            proposal = revise(store, report_id, store.report(report_id)["version"], case["answer"])
            if "priority" in case:
                checks["clarified_priority"] = proposal.get("priority") == case["priority"]
        final_kind = case.get("final_kind", "create" if "answer" in case else
                              "escalate" if "choose" in case else case["kind"])
        checks["handoff"] = proposal["kind"] == final_kind
        version = store.report(report_id)["version"]
        if proposal["kind"] in {"escalate", "create"}:
            confirmed_at = now
            if delay := case.get("approval_delay_hours"):
                now += timedelta(hours=delay)
            receipt = review(store, report_id, version, "approve")
            after = store.jobs()
            replay = review(store, report_id, version, "approve")
            checks["replay"] = receipt == replay and after == store.jobs()
            if final_kind == proposal["kind"] == "create":
                created = next(job for job in after if job["id"] == receipt["job_id"])
                target_start = confirmed_at if case.get("approval_delay_hours") else now
                checks["write"] = (len(after) == len(before) + 1 and created["created_from"] == report_id
                                   and created["priority"] == case["priority"]
                                   and created["category"] == proposal.get("category", "lighting")
                                   and created["deadline"] == (target_start + timedelta(days=case["days"])).isoformat())
            elif final_kind == proposal["kind"] == "escalate":
                changed = [job for old, job in zip(before, after) if old != job and job["escalated_at"]]
                checks["write"] = (len(after) == len(before) and len(changed) == 1
                                   and changed[0]["id"] == case.get("choose", case.get("job")))
            else:
                checks["write"] = False
        elif actual == final_kind == "split":
            receipt = review(store, report_id, version, "approve")
            children = [row for row in store.reports() if row["parent_report_id"] == report_id]
            checks["child_reports"] = (
                [row["id"] for row in children] == receipt["report_ids"]
                and [row["message"] for row in children] == proposal["issues"]
                and all(row["location"] == report["location"] and not row["proposal"] for row in children)
            )
            checks["replay"] = (review(store, report_id, version, "approve") == receipt
                                and [row for row in store.reports() if row["parent_report_id"] == report_id] == children)
            checks["write"] = store.jobs() == before
        else:
            if (followup := case.get("followup")) and actual == final_kind == "manual":
                receipt = review(store, report_id, version, "acknowledge", followup=followup)
                children = [row for row in store.reports() if row["parent_report_id"] == report_id]
                checks["followup"] = (
                    len(children) == 1 and children[0]["message"] == followup
                    and children[0]["location"] == report["location"]
                    and receipt["report_ids"] == [children[0]["id"]]
                )
            checks["write"] = store.jobs() == before
        return {"checks": checks, "retrieval": proposal["retrieval"]}

CASES = json.loads(Path(__file__).with_name("scripted_cases.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_scripted_policy_and_review(case):
    result = evaluate(case)
    assert all(result["checks"].values()), result["checks"]


def test_live_dataset_contract():
    from collections import Counter

    from evals.run import load_cases
    from halldesk.retrieval import handbook_index

    cases = load_cases("triage")
    counts = Counter(case["slice"] for case in cases)
    assert counts == {"hazard": 22, "edge": 12, "normal": 13, "ambiguous": 10,
                      "adversarial": 8, "non-English": 9}
    holdout = load_cases("safety")
    assert len({case["id"] for case in [*cases, *holdout]}) == len(cases) + len(holdout)
    assert len({case["inputs"]["message"] for case in holdout} &
               {case["inputs"]["message"] for case in cases}) == 0
    sections, _ = handbook_index()
    for case in [*cases, *holdout]:
        expected = case["outputs"]
        if case in cases:
            assert expected.get("route"), case["id"]
        else:
            assert "urgency" in expected, case["id"]
        if case["slice"] == "hazard":
            assert expected["urgency"] in {"urgent", "emergency"}, case["id"]
        for key in ("issues", "evidence", "safety_evidence", "lighting_evidence"):
            quotes = expected.get(key) or []
            if isinstance(quotes, str):
                quotes = [quotes]
            assert all(quote.strip() and quote in case["inputs"]["message"] for quote in quotes), case["id"]
        if case in cases:
            assert set(case.get("policy_sections", [])) <= sections.keys(), case["id"]
