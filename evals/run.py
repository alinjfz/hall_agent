"""Run reproducible live experiments against the production triage entry point."""

import argparse
import inspect
import json
import os
from collections import Counter, defaultdict
from contextlib import nullcontext
from contextvars import ContextVar
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from statistics import mean, median
from tempfile import TemporaryDirectory
from threading import Lock
from time import perf_counter
from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.context import register_configure_hook
from langchain_core.tracers.langchain import wait_for_all_tracers
from langsmith import Client, trace, tracing_context
from langsmith.schemas import Example

from evals import evaluators
from halldesk.models import configured_provider, model_id, setup_issue
from halldesk.safety import safety_signals
from halldesk.store import Store
from halldesk.workflow import triage

HERE = Path(__file__).resolve().parent
DATASET_NAMES = {"triage": "unihelp-triage-v2", "safety": "unihelp-safety-holdout", "curated": "unihelp-curated"}
REPETITIONS = 3


class LocalTraceClient(Client):
    """Capture the local run tree without posting callback events."""

    def create_run(self, *args, **kwargs):
        return None

    def update_run(self, *args, **kwargs):
        return None


def local_trace_client():
    return LocalTraceClient(api_url="http://127.0.0.1:1", api_key="local", auto_batch_tracing=False, info={})


def load_cases(suite):
    path = HERE / ("dataset.json" if suite == "triage" else "safety_holdout.json")
    return json.loads(path.read_text())


def prepare(store, inputs, guidance=()):
    store.seed(samples=False)
    with store.transaction(write=True) as db:
        report_id = db.execute("INSERT INTO reports (location,message,received_at) VALUES (?,?,?)",
                               (inputs["location"], inputs["message"], inputs["clock"])).lastrowid
        for job in inputs["open_jobs"]:
            original = inputs.get("job_reports", {}).get(str(job["id"]))
            source_id = (db.execute("INSERT INTO reports (location,message,deleted_at) VALUES (?,?,?)",
                                    (job["location"], original["message"], inputs["clock"])).lastrowid
                         if original else None)
            db.execute("""INSERT INTO jobs
                          (id,location,fitting,deadline,priority,status,category,labels,escalated_at,created_from,assignee)
                          VALUES (:id,:location,:fitting,:deadline,:priority,:status,:category,:labels,:escalated_at,
                                  :created_from,:assignee)""",
                       job | {"labels": json.dumps(job["labels"]), "created_from": source_id,
                              "assignee": job.get("assignee")})
        if guidance:
            db.executemany("""INSERT INTO eval_cases
                (id,name,message_key,snapshot,expected,notes,status,interpretation,policy_revisions)
                VALUES (?,?,?,?,?,?,'guidance',?,?)""", [
                (row["id"], row["name"], row["message_key"], json.dumps(row["snapshot"]), json.dumps(row["expected"]),
                 row["notes"], json.dumps(row["interpretation"]), json.dumps(row["policy_revisions"]))
                for row in guidance
            ])
    return report_id


def descendants(run):
    for child in sorted(run.child_runs, key=lambda node: node.dotted_order):
        yield child
        yield from descendants(child)


def read_trace(run, topics, expected_sections, provider_calls):
    runs = sorted(descendants(run), key=lambda node: (node.start_time, node.dotted_order))
    tools, evidence = [], []
    for node in runs:
        if node.run_type == "tool":
            topic = node.inputs.get("topic") if node.name in {"search_rules", "search_policy"} else None
            query = node.inputs.get("query")
            on_topic = topic in topics if topic else None
            if node.name == "search_policy" and expected_sections:
                payload = (node.outputs or {}).get("output")
                if hasattr(payload, "content"):
                    payload = payload.content
                if isinstance(payload, dict):
                    payload = payload.get("content")
                if isinstance(payload, str):
                    try:
                        on_topic = bool({section["id"] for section in json.loads(payload)} & set(expected_sections))
                    except (TypeError, ValueError, KeyError):
                        pass
            origin = (node.extra or {}).get("metadata", {}).get("lookup_origin")
            tools.append({"name": node.name, "origin": origin, "topic": topic, "query": query, "on_topic": on_topic})
        if node.name == "apply_policy":
            interpretation = node.inputs.get("interpretation", {})
            if hasattr(interpretation, "model_dump"):
                interpretation = interpretation.model_dump()
            if isinstance(interpretation, dict):
                evidence.extend(interpretation[key] for key in ("evidence", "lighting_evidence")
                                if interpretation.get(key))
                evidence.extend(interpretation.get("issues") or [])
        if node.name == "check_safety" and (quote := (node.outputs or {}).get("safety_evidence")):
            evidence.append(quote)
    passes = [node for node in runs if node.name == "interpretation_pass"]
    state_errors = [node.outputs.get("model_error") for node in runs if node.name == "interpret" and node.outputs]
    failure = (next((error for error in state_errors if error), "") if state_errors
               else next((node.error for node in passes if node.error), ""))
    return {
        "tools": tools,
        "steps": [node.name for node in runs],
        "model_calls": max(provider_calls, sum(node.run_type == "llm" for node in runs)),
        "draft_rejected_by_grounding": any(
            (node.extra or {}).get("metadata", {}).get("draft_rejected_by_grounding") for node in runs
        ),
        "failed_interpretation": bool(failure) or bool(passes and not any(node.name == "apply_policy" for node in runs)),
        "error_kind": ("timeout" if "timeout" in failure.casefold() or "timed out" in failure.casefold()
                       else "model_error" if failure else None),
    }, evidence


def evaluate_case(inputs, *, topics, expected_sections=(), passes, hosted, model=None, guidance=()):
    # The baseline mixes LangSmith decorators and LangChain callbacks. Keep a
    # local LangChain tracer as well so local runs count actual chat model calls.
    with TemporaryDirectory() as directory:
        store = Store(Path(directory) / "case.sqlite3", clock=lambda: datetime.fromisoformat(inputs["clock"]))
        report_id = prepare(store, inputs, guidance)
        before = store.jobs()
        started = perf_counter()
        counter = ModelCalls()
        token = callback_tracer.set(counter)
        try:
            local = None if hosted else local_trace_client()
            with tracing_context(enabled=True if hosted else "local", client=local), trace("live_triage") as root:
                options = {"passes": passes} if "passes" in inspect.signature(triage).parameters else {}
                proposal = triage(store, report_id, model=model, **options)
        finally:
            callback_tracer.reset(token)
        trajectory, evidence = read_trace(root, topics, expected_sections, counter.count)
        reports = store.reports()
        no_writes = (before == store.jobs() and len(reports) == 1 and not reports[0]["result"]
                     and not any(row["event"] in {"reply_approved", "review_complete"}
                                 for row in store.activity(report_id)))
        signals = safety_signals(inputs["message"])
        interpret = next((node.outputs or {} for node in descendants(root) if node.name == "interpret"), {})
        return {
            "route": proposal["kind"], "urgency": proposal.get("urgency"),
            "intake_urgency": ("emergency" if signals & {"fire", "gas"} else "urgent" if signals else None),
            "job": (proposal.get("job") or {}).get("id"),
            "candidates": sorted(job["id"] for job in proposal.get("candidates", [])),
            "related_jobs": sorted(job["id"] for job in proposal.get("related_jobs", [])),
            "category": proposal.get("category"), "priority": proposal.get("priority"), "days": proposal.get("days"),
            "deadline": proposal.get("policy_deadline"), "issues": proposal.get("issues"),
            "section_ids": proposal.get("retrieval", {}).get("section_ids", []),
            "guidance_ids": proposal.get("retrieval", {}).get("example_ids", []),
            "expected_sections": expected_sections,
            "sections": interpret.get("sections", []),
            **({"nominated_job_ids": interpret.get("nominated_job_ids")}
               if "passes" not in inspect.signature(triage).parameters else {}),
            "reply": proposal.get("reply"), "reply_sources": proposal.get("reply_sources", []),
            "job_record": proposal.get("job"),
            "no_write_before_approval": no_writes, "evidence": evidence, "trajectory": trajectory,
            "trace_id": store.report(report_id)["trace_id"],
            "latency_ms": round((perf_counter() - started) * 1000),
        }


class ModelCalls(BaseTracer):
    def __init__(self):
        super().__init__()
        self.count = 0

    def on_chat_model_start(self, *args, **kwargs):
        self.count += 1
        return super().on_chat_model_start(*args, **kwargs)

    def _persist_run(self, run):
        pass


callback_tracer = ContextVar("unihelp_eval_tracer", default=None)
register_configure_hook(callback_tracer, inheritable=True, handle_class=ModelCalls)


def examples_for(client, suite, cases, hosted, *, sync=True):
    name = DATASET_NAMES[suite]
    if hosted:
        if client.has_dataset(dataset_name=name):
            dataset = client.read_dataset(dataset_name=name)
        elif sync:
            dataset = client.create_dataset(dataset_name=name, description="Synthetic UniHelp fixtures")
        else:
            raise ValueError(f"Sync the complete {suite} suite before running a hosted subset.")
        dataset_id = dataset.id
    else:
        dataset_id = uuid5(NAMESPACE_URL, name)

    def payload(case):
        return {"id": uuid5(dataset_id, case["id"]), "inputs": case["inputs"], "outputs": case["outputs"],
                "metadata": {"case_id": case["id"], "slice": case["slice"],
                             "policy_topics": case["policy_topics"],
                             "policy_sections": case.get("policy_sections", [])}}

    wanted = {row["id"]: row for row in map(payload, cases)}
    if not hosted:
        return [Example(dataset_id=dataset_id, **row) for row in wanted.values()], None

    existing = {row.id: row for row in client.list_examples(dataset_id=dataset_id)}
    if sync:
        missing = [row | {"split": [row["metadata"]["slice"]]}
                   for key, row in wanted.items() if key not in existing]
        if missing:
            client.create_examples(dataset_id=dataset_id, examples=missing)
        for key, row in wanted.items():
            if key in existing and changed(row, existing[key]):
                client.update_example(key, dataset_id=dataset_id, inputs=row["inputs"], outputs=row["outputs"],
                                      metadata=row["metadata"], split=[row["metadata"]["slice"]])
    version = client.read_dataset_version(dataset_id=dataset_id, tag="latest").as_of
    pinned = {row.id: row for row in client.list_examples(dataset_id=dataset_id, as_of=version)}
    if any(key not in pinned or changed(row, pinned[key]) for key, row in wanted.items()):
        raise ValueError("The hosted dataset does not match these fixtures; sync the full suite and retry.")
    return [pinned[key] for key in wanted], version.isoformat()


def changed(payload, stored):
    return (payload["inputs"] != stored.inputs or payload["outputs"] != stored.outputs
            or any((stored.metadata or {}).get(key) != value for key, value in payload["metadata"].items()))


def summarize(rows):
    slices = defaultdict(lambda: defaultdict(list))
    for row in rows:
        slices[row["slice"]][row["case"]].append(all(value for value in row["checks"].values() if value is not None))
    return {
        label: {
            "cases": len(groups),
            "passed_trials": sum(sum(samples) for samples in groups.values()),
            "total_trials": sum(map(len, groups.values())),
            "passed_every_trial": sum(all(samples) for samples in groups.values()),
            "passed_any_trial": sum(any(samples) for samples in groups.values()),
        }
        for label, groups in sorted(slices.items())
    }


def values_for(checks, inputs, outputs, reference_outputs):
    available = locals()
    return {check.__name__: check(**{name: available[name] for name in inspect.signature(check).parameters})
            for check in checks}


def langsmith_evaluator(check):
    def score(run, example):
        if not run.outputs:
            return {"key": check.__name__, "value": "target failed"}
        value = values_for((check,), example.inputs, run.outputs, example.outputs)[check.__name__]
        if isinstance(value, dict):
            return value
        return ({"key": check.__name__, "score": value if isinstance(value, float) else int(value)} if value is not None
                else {"key": check.__name__, "value": "not applicable"})

    score.__name__ = check.__name__
    return score


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=DATASET_NAMES, default="triage")
    parser.add_argument("--agent", choices=("openai", "vertex", "lmstudio"))
    parser.add_argument("--model")
    parser.add_argument("--langsmith", choices=("local", "hosted"), default="local")
    parser.add_argument("--sync-only", action="store_true", help="Sync the full dataset without running the model")
    legacy = "passes" in inspect.signature(triage).parameters
    parser.add_argument("--passes", choices=("single", "specialist", "full"), default="full" if legacy else "single")
    parser.add_argument("--case", action="append", default=[], help="Select case IDs; repeat to select several")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--judge", action="store_true", help="Score drafted replies with the separate LM Studio model")
    parser.add_argument("--curated", type=Path, help="Approved holdouts and guidance exported by evals.curation")
    parser.add_argument("--without-guidance", action="store_true")
    parser.add_argument("--against", type=Path, help="Fail when a case regresses from a complete, comparable run")
    args = parser.parse_args()
    if not legacy and args.passes != "single":
        parser.error("The production graph uses one structured interpretation.")
    if args.agent:
        os.environ["HALL_PROVIDER"] = args.agent
    if args.model:
        os.environ["HALL_MODEL"] = args.model
    provider = configured_provider()
    if issue := setup_issue(provider):
        parser.error(issue)
    hosted = args.langsmith == "hosted"
    if args.sync_only and (not hosted or args.case):
        parser.error("--sync-only requires --langsmith hosted and the full dataset.")
    if hosted and not os.getenv("LANGSMITH_API_KEY"):
        parser.error("Set LANGSMITH_API_KEY for hosted experiments.")
    if args.without_guidance and not args.curated:
        parser.error("--without-guidance requires --curated.")
    if args.curated and (legacy or args.suite == "safety"):
        parser.error("Curated runs require the current triage graph.")
    if args.suite == "curated" and not args.curated:
        parser.error("Pass an approved export with --curated.")
    if args.curated:
        args.suite = "curated"
        export = json.loads(args.curated.read_text())
        all_cases = export["cases"]
        guidance = [] if args.without_guidance else export["guidance"]
    else:
        all_cases = load_cases(args.suite)
        guidance = []
    if not all_cases:
        parser.error("The suite is empty.")
    cases = [case for case in all_cases if not args.case or case["id"] in args.case]
    if unknown := set(args.case) - {case["id"] for case in cases}:
        parser.error(f"Unknown case IDs: {', '.join(sorted(unknown))}")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive.")

    by_input = {json.dumps(case["inputs"], sort_keys=True): case for case in cases}
    if len(by_input) != len(cases):
        parser.error("Two cases have the same input snapshot.")
    if not hosted:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
    client = Client() if hosted else local_trace_client()
    examples, version = examples_for(client, args.suite, cases, hosted, sync=not args.case)
    agent = model_id(provider, os.getenv("HALL_MODEL"))
    fingerprint = sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()
    if args.sync_only:
        print(f"{DATASET_NAMES[args.suite]}: {len(examples)} cases, version {version}, sha256 {fingerprint}")
        return 0

    inference_lock = Lock() if args.judge else None

    def target(inputs):
        case = by_input[json.dumps(inputs, sort_keys=True)]
        with inference_lock if inference_lock else nullcontext():
            return evaluate_case(inputs, topics=case["policy_topics"], expected_sections=case.get("policy_sections", []),
                                 passes=args.passes, hosted=hosted, guidance=guidance)

    checks = (*evaluators.CORE, evaluators.reply_grounded)
    graders = [langsmith_evaluator(check) for check in (
        *checks, *evaluators.TRAJECTORY, evaluators.intake_urgency, evaluators.retrieval_recall_at_3)]
    if args.judge:
        from evals.judge import RUBRICS, evaluator_for, judge_model_id
        judge_model_id()

        def serial_judge(criterion):
            judge = evaluator_for(criterion)

            def score(inputs, outputs):
                with inference_lock:
                    return judge(inputs, outputs)

            score.__name__ = criterion
            return score

        graders.extend(serial_judge(criterion) for criterion in RUBRICS)
    experiment = client.evaluate(
        target, data=examples, evaluators=graders,
        experiment_prefix=f"unihelp-{args.suite}", num_repetitions=REPETITIONS, max_concurrency=args.concurrency,
        upload_results=hosted, blocking=False, metadata={"model": agent, "provider": provider, "passes": args.passes,
                                         "dataset_version": version, "dataset_sha256": fingerprint,
                                         "guidance_examples": len(guidance)},
    )
    rows = []
    partial_path = args.output.with_suffix(".partial.jsonl") if args.output else None
    if partial_path:
        partial_path.parent.mkdir(parents=True, exist_ok=True)
    with partial_path.open("w") if partial_path else nullcontext() as progress:
        for row in experiment:
            example = row["example"]
            run = row["run"]
            expected = example.outputs
            actual = run.outputs
            scores = values_for(checks, example.inputs, actual, expected) if actual else {"completed": False}
            result = {"case": example.metadata["case_id"], "slice": example.metadata["slice"],
                      "expected": expected, "actual": actual, "checks": scores,
                      "run_id": str(run.id),
                      "trajectory_checks": values_for(evaluators.TRAJECTORY, example.inputs, actual, expected) if actual else {},
                      "error": bool(run.error or (actual and actual["trajectory"]["failed_interpretation"]))}
            if args.judge:
                result["judge"] = {grade.key: {"score": grade.score, "reason": grade.comment}
                                   for grade in row["evaluation_results"]["results"] if grade.key in RUBRICS}
            rows.append(result)
            if progress:
                progress.write(json.dumps(result, ensure_ascii=False) + "\n")
                progress.flush()
            print(f"[{len(rows)}/{len(cases) * REPETITIONS}] {result['case']}: "
                  f"{actual['route'] if actual else 'error'}", flush=True)
    wait_for_all_tracers()
    counts = Counter(row["case"] for row in rows)
    if counts != Counter({case["id"]: REPETITIONS for case in cases}):
        raise RuntimeError("Experiment did not return the required repetitions for every case.")

    ranks = {None: 0, "urgent": 1, "emergency": 2}
    def actual_urgency(row, key):
        return (row["actual"] or {}).get(key)

    def safety_counts(key):
        return {
            "misses": sum(ranks[actual_urgency(row, key)] < ranks[row["expected"]["urgency"]]
                          for row in rows if "urgency" in row["expected"]),
            "false_alarms": sum("urgency" in row["expected"] and row["expected"]["urgency"] is None
                                 and actual_urgency(row, key) is not None for row in rows),
        }

    hazards = {row["case"] for row in rows if row["expected"].get("urgency")}
    lookalikes = {row["case"] for row in rows
                  if "urgency" in row["expected"] and row["expected"]["urgency"] is None}

    def safety_cases(key):
        under_triaged = {row["case"] for row in rows if row["case"] in hazards
                         and ranks[actual_urgency(row, key)] < ranks[row["expected"]["urgency"]]}
        false_alarms = {row["case"] for row in rows if row["case"] in lookalikes and actual_urgency(row, key)}
        return {"hazards_caught_every_trial": len(hazards - under_triaged), "hazards": len(hazards),
                "lookalikes_flagged_at_least_once": len(false_alarms), "lookalikes": len(lookalikes)}

    safety = {"live": safety_counts("urgency"), "intake": safety_counts("intake_urgency")}
    safety_by_case = {"live": safety_cases("urgency"), "intake": safety_cases("intake_urgency")}
    slices = summarize(rows)
    actuals = [row["actual"] for row in rows if row["actual"]]
    nominations = [value for row in rows
                   if (value := row["trajectory_checks"].get("nominated_job")) is not None]
    interpretation = {
        "nominated_job": {"passed": sum(nominations), "total": len(nominations)},
        "median_latency_ms": median(row["latency_ms"] for row in actuals) if actuals else None,
        "model_calls": sum(row["trajectory"]["model_calls"] for row in actuals),
    }
    recalls = [score for row in rows if row["actual"]
               if (score := evaluators.retrieval_recall_at_3(row["actual"])) is not None]
    summary = {"suite": args.suite, "provider": provider, "model": agent, "passes": args.passes,
               "repetitions": REPETITIONS, "dataset": DATASET_NAMES[args.suite], "dataset_version": version,
               "guidance_examples": len(guidance),
               "interpretation": interpretation,
               "dataset_sha256": fingerprint, "experiment_url": experiment.url if hosted else None,
               "slices": slices, "safety": safety, "safety_by_case": safety_by_case,
               "retrieval_recall_at_3": mean(recalls) if recalls else None, "results": rows}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        partial_path.unlink(missing_ok=True)
    for label, stats in slices.items():
        print(f"{label}: {stats['passed_trials']}/{stats['total_trials']} trials "
              f"({stats['passed_every_trial']}/{stats['cases']} cases passed every trial; "
              f"{stats['passed_any_trial']}/{stats['cases']} passed at least one)")
    print(json.dumps(interpretation))
    print(json.dumps(safety))
    if summary["experiment_url"]:
        print(summary["experiment_url"])
    if args.suite == "safety":
        return int(bool(safety["live"]["misses"] or any(row["error"] for row in rows)))
    comparison = args.against
    if comparison is None and not args.case and args.suite == "triage" and not legacy:
        candidate = HERE / "results" / "last-approved.json"
        comparison = candidate if candidate.is_file() else None
    if comparison:
        previous = json.loads(comparison.read_text())
        if previous["dataset_sha256"] != fingerprint or previous["model"] != agent:
            parser.error("Regression comparison requires the same dataset and model.")
        if Counter(row["case"] for row in previous["results"]) != counts:
            parser.error("Regression comparison requires a complete run with matching repetitions.")
        regressions = sorted(passing_cases(previous["results"]) - passing_cases(rows))
        if regressions:
            print("Regressions: " + ", ".join(regressions))
        return int(bool(regressions or any(row["error"] for row in rows)))
    return int(any(not all(v for v in row["checks"].values() if v is not None) for row in rows))


def passing_cases(rows):
    cases = {row["case"] for row in rows}
    return cases - {row["case"] for row in rows if row["error"] or
                    not all(value for value in row["checks"].values() if value is not None)}


if __name__ == "__main__":
    raise SystemExit(main())
