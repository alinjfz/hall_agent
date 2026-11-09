import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter
from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv
from langsmith import Client, traceable, tracing_context
from langsmith.schemas import Example
from langchain_core.tracers.langchain import wait_for_all_tracers

from evals.judge import judge_clarification, judge_model_id
from halldesk.models import configured_provider, model_id, setup_issue
from halldesk.store import Store
from halldesk.workflow import review, revise, triage
from tests.scripted_model import scripted

NOW = datetime(2026, 11, 8, 12, tzinfo=timezone.utc)


CLARIFICATION_FACTS = {
    "plumbing_question": "Whether the tap or shower is only dripping or water is spreading",
    "job_question": "Which fitting or job the resident is following up on",
    "lighting_question": "Whether another light works in the same room",
}


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
            db.executemany("""INSERT INTO eval_cases (id,name,message_key,snapshot,expected,notes,status)
                VALUES (?,?,?,?,?,?,'guidance')""", [
                (row["id"], row["name"], row["message_key"], json.dumps(row["snapshot"]),
                 json.dumps(row["expected"]), row["notes"]) for row in case.get("guidance", [])
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


@traceable(name="evaluate_case", process_inputs=lambda inputs: {"case": inputs["case"]["name"]})
def evaluate(case, live, use_judge, passes="full"):
    with TemporaryDirectory() as directory:
        snapshot = case.get("snapshot")
        now = datetime.fromisoformat(snapshot["at"]) if snapshot else NOW
        store = Store(Path(directory) / "eval.sqlite3", clock=lambda: now)
        report_id = prepare_case(store, case)
        report = store.report(report_id)
        before = store.jobs()
        model = None if live else scripted(report["location"], case["topic"], case["script"])
        audit = {}
        started = perf_counter()
        proposal = triage(store, report_id, model=model, passes=passes, audit=audit)
        latency_ms = round((perf_counter() - started) * 1000)
        actual = proposal["kind"]
        checks = {"route": actual == case["kind"], "triage_is_read_only": store.jobs() == before,
                  "retrieved_policy": bool(proposal["retrieval"]["policies"])}
        if audit.get("model_error"):
            checks["model_completed"] = False
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
        score = None
        if use_judge and case["kind"] in CLARIFICATION_FACTS:
            choices = [f"#{job['id']} {job['fitting']}" for job in proposal.get("candidates", [])]
            if actual == "lighting_question":
                choices = ["Yes, another light works", "No, this was the only working light"]
            elif actual == "plumbing_question":
                choices = ["Only dripping", "Water is spreading"]
            score = judge_clarification(report["message"], CLARIFICATION_FACTS[case["kind"]],
                                        proposal.get("question"), choices)
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
        interpretation = audit.get("interpretation", {})
        evidence = interpretation.get("evidence")
        requested_topics = audit.get("requested_policy_topics", [])
        relevant_topics = {case.get("topic")}
        if case.get("topic") == "escalation":
            relevant_topics.add("follow_up")
        diagnostics = {
            "model_candidates_correct": (set(interpretation.get("candidate_ids", [])) == set(case["model_candidates"])
                                         if "model_candidates" in case and interpretation else None),
            "model_quote_in_report": evidence in report["message"] if evidence else None,
            "policy_lookups_on_topic": set(requested_topics) <= relevant_topics if requested_topics and case.get("topic") else None,
        }
        return {"case": case["name"], "expected": case["kind"], "actual": actual,
                "hazard": case.get("hazard"), "ambiguity": case.get("ambiguity"),
                "actual_urgency": proposal.get("urgency"), "latency_ms": latency_ms, "trajectory": audit, "diagnostics": diagnostics,
                "checks": checks, "judge": score, "retrieval": proposal["retrieval"], "trace_id": store.report(report_id)["trace_id"]}


def run_case(case, live, use_judge, passes="full"):
    try:
        return evaluate(case, live, use_judge, passes)
    except Exception as exc:
        logging.error("Case %s failed with %s", case["name"], type(exc).__name__)
        return {"case": case["name"], "expected": case["kind"], "actual": "error",
                "hazard": case.get("hazard"), "ambiguity": case.get("ambiguity"),
                "error": type(exc).__name__, "judge": None, "checks": {"completed": False}}


def experiment_scores(outputs):
    checks = outputs["checks"]
    scores = [{"key": name, "score": int(passed)} for name, passed in checks.items()]
    scores.append({"key": "case_passed", "score": int(all(checks.values()))})
    for name, value in outputs.get("diagnostics", {}).items():
        if value is not None:
            scores.append({"key": name, "score": int(value)})
    if outputs["judge"] is not None:
        scores.append({"key": "clarification_quality", "score": int(outputs["judge"]["resolves_missing_fact"])})
    return {"results": scores}


def run_experiment(cases, args, agent):
    # Guidance is a run setting. Both sides of the comparison share the same dataset.
    fixtures = [{key: value for key, value in case.items() if key != "guidance"} for case in cases]
    guidance = {case["name"]: case.get("guidance", []) for case in cases}
    fingerprint = sha256(json.dumps(fixtures, sort_keys=True).encode()).hexdigest()[:12]
    dataset_name = f"halldesk-{fingerprint}"
    prefix = "halldesk-with-guidance" if any(guidance.values()) else "halldesk-without-guidance"
    hosted = args.langsmith == "hosted"
    client = Client() if hosted else Client(auto_batch_tracing=False, info={})
    if hosted:
        if client.has_dataset(dataset_name=dataset_name):
            dataset = client.read_dataset(dataset_name=dataset_name)
        else:
            dataset = client.create_dataset(dataset_name, description="HallDesk labelled evaluation fixtures")
        dataset_id = dataset.id
    else:
        dataset_id = uuid5(NAMESPACE_URL, dataset_name)
    examples = [
        Example(id=uuid5(dataset_id, case["name"]), dataset_id=dataset_id, inputs={"case": case})
        for case in fixtures
    ]
    if hosted:
        existing = {example.id for example in client.list_examples(dataset_id=dataset_id)}
        missing = [{"id": example.id, "inputs": example.inputs} for example in examples if example.id not in existing]
        if missing:
            client.create_examples(dataset_id=dataset_id, examples=missing)

    def target(inputs):
        case = inputs["case"]
        # The SDK captures the local experiment result. Disable child traces so
        # LangChain callbacks do not try to send them to the configured endpoint.
        with tracing_context(enabled=hosted):
            return run_case(case | {"guidance": guidance[case["name"]]}, args.live, args.judge, args.passes)

    experiment = client.evaluate(
        target, data=dataset_name if hosted else examples, evaluators=[experiment_scores],
        experiment_prefix=f"{prefix}-{args.passes}", metadata={"model": agent or "scripted", "judge": args.judge, "passes": args.passes},
        upload_results=hosted,
    )
    return [row["run"].outputs for row in experiment]


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Use the configured model instead of scripted outputs")
    parser.add_argument("--judge", action="store_true", help="Use an LLM judge for clarifications")
    parser.add_argument("--agent", choices=["openai", "vertex", "lmstudio"], help="Override the live triage provider")
    parser.add_argument("--model", help="Override the live agent model ID")
    parser.add_argument("--case", action="append", help="Run only the named case, repeatable")
    parser.add_argument("--calibrate-judge", action="store_true", help="Compare the GPT judge against labelled examples")
    parser.add_argument("--curated-db", type=Path, help="Read approved holdouts from a HallDesk database")
    parser.add_argument("--without-guidance", action="store_true", help="Omit approved examples from curated runs")
    parser.add_argument("--langsmith", choices=["local", "hosted"], help="Run through the LangSmith experiment SDK")
    parser.add_argument("--suite", choices=["all", "chase"], default="all")
    parser.add_argument("--passes", choices=["single", "specialist", "full"], default="full")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.agent or args.model) and not args.live:
        parser.error("--agent and --model require --live")
    if args.curated_db and args.suite != "all":
        parser.error("--suite chase uses the starter fixtures")
    if args.curated_db and not args.live:
        parser.error("--curated-db requires --live")
    if args.without_guidance and not args.curated_db:
        parser.error("--without-guidance requires --curated-db")
    if args.calibrate_judge and (args.live or args.judge or args.case or args.langsmith):
        parser.error("--calibrate-judge runs the separate calibration dataset")
    if args.langsmith == "hosted" and not os.getenv("LANGSMITH_API_KEY"):
        parser.error("Set LANGSMITH_API_KEY for hosted experiments.")
    if args.agent:
        os.environ["HALL_PROVIDER"] = args.agent
        os.environ.pop("HALL_MODEL", None)
    if args.model:
        os.environ["HALL_MODEL"] = args.model
    provider = configured_provider()
    if args.live and (issue := setup_issue(provider)):
        parser.error(issue)
    if (args.judge or args.calibrate_judge) and not os.getenv("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY to run the GPT judge.")
    if args.calibrate_judge:
        examples = json.loads(Path(__file__).with_name("judge_cases.json").read_text())
        results = []
        for example in examples:
            score = judge_clarification(example["report"], example["expected_fact"],
                                        example["question"], example["choices"])
            results.append({"case": example["name"], "expected": example["expected"], "judge": score,
                            "checks": {"matches_label": score["resolves_missing_fact"] == example["expected"]}})
    else:
        if args.curated_db:
            if not args.curated_db.is_file():
                parser.error("Curated database does not exist.")
            workspace = Store(args.curated_db)
            guidance = [] if args.without_guidance else workspace.eval_cases("guidance")
            cases = [
                row["expected"] | {"name": f"#{row['id']} {row['name']}", "snapshot": row["snapshot"],
                                   "guidance": guidance}
                for row in workspace.eval_cases("holdout")
            ]
            if not cases:
                parser.error("Approve at least one holdout in HallDesk’s Evals view.")
        else:
            cases = json.loads(Path(__file__).with_name("cases.json").read_text())
        if args.suite == "chase":
            labels = json.loads(Path(__file__).with_name("chase_cases.json").read_text())
            missing = set(labels) - {case["name"] for case in cases}
            if missing:
                parser.error(f"Chase suite references unknown cases: {', '.join(sorted(missing))}")
            cases = [case | labels[case["name"]] for case in cases if case["name"] in labels]
        if args.case:
            unknown = set(args.case) - {case["name"] for case in cases}
            if unknown:
                parser.error(f"Unknown cases: {', '.join(sorted(unknown))}")
            cases = [case for case in cases if case["name"] in args.case]
        if args.langsmith:
            agent = model_id(provider, os.getenv("HALL_MODEL")) if args.live else None
            results = run_experiment(cases, args, agent)
        else:
            results = []
            for case in cases:
                result = run_case(case, args.live, args.judge, args.passes)
                results.append(result)
                if args.live:
                    print(f"[{len(results)}/{len(cases)}] {case['name']}: {result['actual']}", file=sys.stderr)
    wait_for_all_tracers()
    passed = sum(all(row["checks"].values()) for row in results)
    judged = [row["judge"] for row in results if row["judge"] is not None]
    latencies = [row["latency_ms"] for row in results if "latency_ms" in row]
    output = {
        "mode": "judge_calibration" if args.calibrate_judge else "live" if args.live else "scripted",
        "agent": model_id(provider, os.getenv("HALL_MODEL")) if args.live else None,
        "passed": passed, "total": len(results),
        "suite": args.suite, "passes": args.passes,
        "median_triage_ms": median(latencies) if latencies else None,
        "structured_passes": sum(len(row.get("trajectory", {}).get("stages", [])) for row in results),
        "hazard_failures": {
            "missed": sum(row.get("hazard") in {"urgent", "emergency"} and not row.get("actual_urgency")
                          and "actual_urgency" in row for row in results),
            "false_alert": sum(row.get("hazard") == "none" and bool(row.get("actual_urgency")) for row in results),
        },
        "failed_checks": {
            name: sum(row["checks"].get(name) is False for row in results)
            for name in sorted({name for row in results for name, ok in row["checks"].items() if not ok})
        },
        "slices": {
            key: {value: {"passed": sum(all(row["checks"].values()) for row in results if row.get(key) == value),
                          "total": sum(row.get(key) == value for row in results)}
                  for value in sorted({row[key] for row in results if row.get(key) is not None})}
            for key in ("expected", "hazard", "ambiguity")
        },
        "judge_agreement_pct": round(100 * passed / len(results), 1) if args.calibrate_judge else None,
        "judge_model": judge_model_id() if judged else None,
        "judge_passed": sum(score["resolves_missing_fact"] for score in judged),
        "judge_total": len(judged), "results": results,
    }
    if args.output:
        args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"{passed}/{len(results)} {'reference labels matched' if args.calibrate_judge else 'deterministic cases passed'}")
    if args.calibrate_judge:
        print(f"Judge agreement: {output['judge_agreement_pct']}%")
    if args.suite == "chase":
        print(f"Structured passes: {output['structured_passes']} · median triage: {output['median_triage_ms']} ms")
        for dimension, slices in output["slices"].items():
            print(dimension + ": " + ", ".join(
                f"{value} {counts['passed']}/{counts['total']}" for value, counts in slices.items()
            ))
    if judged and not args.calibrate_judge:
        print(f"GPT judge: {output['judge_passed']}/{len(judged)} clarifications passed")
    for row in results:
        failed = [name for name, ok in row["checks"].items() if not ok]
        actual = row.get("actual", str(row["judge"]["resolves_missing_fact"]) if row["judge"] else "")
        print(f"{row['case']}: {actual}" + (f" (failed: {', '.join(failed)})" if failed else ""))
    return int(passed != len(results))


if __name__ == "__main__":
    raise SystemExit(main())
