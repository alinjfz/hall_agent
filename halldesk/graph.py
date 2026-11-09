"""Production graph: triage, wait for staff, then apply their decision."""

import json
import logging
from atexit import register
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from pathlib import Path
from sqlite3 import connect
from tempfile import TemporaryDirectory
from threading import Lock
from time import time
from typing import Any, TypedDict
from uuid import NAMESPACE_URL, uuid5

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, RetryPolicy, interrupt
from langsmith import get_current_run_tree, traceable

from halldesk.context import relevant_examples, retrieve_policy, search_rules
from halldesk.interpretation import Interpretation
from halldesk.models import model_timeout, structured_model, transient_error, triage_model
from halldesk.policy import build_proposal, context_issue, identifies_fitting
from halldesk.proposals import parse_proposal
from halldesk.replies import ReplyDraft, grounded_reply
from halldesk.review_policy import revise_proposal
from halldesk.safety import SafetyCheck, safety_signals
from halldesk.store import ReviewError, Store
from halldesk.tools import search_policy

PROMPTS = Path(__file__).with_name("prompts")
PROMPT = (PROMPTS / "interpret.txt").read_text()
SAFETY_PROMPT = (PROMPTS / "safety.txt").read_text()
DRAFT_PROMPT = (PROMPTS / "draft_reply.txt").read_text()
GRAPH_VERSION = "unified-3"
PROMPT_HASH = sha256("\0".join((PROMPT, SAFETY_PROMPT, DRAFT_PROMPT)).encode()).hexdigest()
DEADLINE_SECONDS = 20
MAX_ATTEMPTS = 3
MAX_TOOL_CALLS = 4
POLICY_QUERIES = {
    "chase": "chasing repair job recorded target overdue",
    "access": "keys access spare key lockout",
    "multiple": "reporting repairs separate faults",
    "plumbing": "tap shower contained drip water leak",
    "lighting": "fixed lighting repair target",
    "heating": "verified heating faults",
    "appliance": "verified appliance faults",
    "lockout": "keys access lockout",
    "spare_key": "keys access spare key",
}


@dataclass
class Context:
    """Run-scoped dependencies. Studio supplies none and gets an isolated workspace."""
    store: Any = None
    model_factory: Any = None
    draft_replies: bool = True


class State(TypedDict, total=False):
    report_id: int
    location: str
    subject: str
    message: str
    existing: bool
    expected_version: int
    expected_claim: str | None
    report: dict
    context: dict
    signals: list[str]
    safety_review: list[str]
    safety_evidence: str | None
    interpretation: dict | None
    nominated_job_ids: list[int] | None
    proposal: dict
    decision: dict
    receipt: dict
    error: str
    model_error: str
    version: int
    expires: float
    sections: list[dict]


def dump_proposal(proposal):
    return parse_proposal(proposal).model_dump(mode="json", exclude_unset=True)


def revision_thread(report_id, version):
    return f"unihelp-{report_id}-{version}"


def retrieve_context(store, report):
    policies = retrieve_policy()
    return {
        "policies": policies,
        "jobs": store.jobs(report["location"]),
        "job_reports": store.open_job_reports(report["location"]),
        "examples": relevant_examples(store.active_guidance(policies), report["message"]),
        "repeat_of_report_id": store.previous_matching_report(report),
        "at": store.clock().isoformat(),
    }


def invoke_structured(model_factory, expires, schema, messages, **kwargs):
    remaining = expires - time()
    if remaining <= 0:
        raise TimeoutError
    model = model_factory(timeout=remaining)
    if run := get_current_run_tree():
        primary = model[0] if isinstance(model, tuple) else model
        run.add_metadata({"model_id": getattr(primary, "model_name", None) or getattr(primary, "model", None)})
    result = structured_model(model, schema, method="json_schema", **kwargs).invoke(messages)
    if time() >= expires:
        raise TimeoutError
    return result


@traceable(name="interpretation_pass", process_inputs=lambda data: {"report_id": data["report"]["id"]})
def interpret_with_tools(model_factory, report, context, expires):
    guidance = [
        {"report": row["snapshot"]["message"], "note": row["notes"],
         "interpretation": {key: value for key, value in row["interpretation"].items()
                            if key in Interpretation.model_fields}}
        for row in context["examples"] if row.get("interpretation")
    ]
    payload = {"location": report["location"], "resident_report": report["message"], "open_jobs": context["jobs"]}
    if guidance:
        payload["staff_examples"] = guidance
    messages = [SystemMessage(PROMPT), HumanMessage(json.dumps(payload, ensure_ascii=False))]
    result = invoke_structured(model_factory, expires, Interpretation, messages, include_raw=True,
                               tools=[search_policy], strict=True)
    calls = getattr(result.get("raw"), "tool_calls", [])
    sections = {}
    if calls:
        if len(calls) > MAX_TOOL_CALLS or any(call["name"] != search_policy.name for call in calls):
            return None, [], None
        messages.append(result["raw"])
        for call in calls:
            reply = search_policy.invoke(call, config={"metadata": {"lookup_origin": "model"}})
            messages.append(reply)
            if reply.status != "error":
                sections.update((section["id"], section) for section in json.loads(reply.content))
        result = invoke_structured(model_factory, expires, Interpretation, messages, include_raw=True)
        if getattr(result.get("raw"), "tool_calls", []):
            return None, list(sections.values()), None
    parsed = result.get("parsed")
    if parsed is None:
        return None, list(sections.values()), None
    if not calls:
        key = parsed.intent if parsed.intent in {"chase", "access", "multiple"} else parsed.category
        query = POLICY_QUERIES.get(key, "reporting repairs maintenance")
        sections = {section["id"]: section for section in json.loads(
            search_policy.invoke({"query": query}, config={"metadata": {"lookup_origin": "default"}}))}
    nominated = list(parsed.candidate_ids) if parsed.intent == "chase" else None
    return verified_candidates(parsed, context["jobs"]), list(sections.values()), nominated


def verified_candidates(interpretation, jobs):
    if interpretation.intent != "chase":
        return interpretation
    evidence = interpretation.evidence
    known_ids = {job["id"] for job in jobs}
    if interpretation.candidate_ids and not set(interpretation.candidate_ids) <= known_ids:
        return interpretation.model_copy(update={"candidate_ids": []})
    exact = [job["id"] for job in jobs if identifies_fitting(evidence, job["fitting"])]
    if exact:
        candidates = exact
    else:
        requested = [job["id"] for job in jobs if job["id"] in interpretation.candidate_ids]
        plausible = [job["id"] for job in jobs if job["category"] == interpretation.category]
        # A category can justify asking which job. A single partial match does
        # not justify escalation; policy still demands fitting-specific evidence.
        candidates = requested if len(requested) > 1 else plausible if len(plausible) > 1 else []
    return interpretation.model_copy(update={"candidate_ids": candidates})


def store_for_run(config, runtime):
    context = runtime.context or Context()
    return context.store or studio_store(config)


def model_for_run(runtime):
    return (runtime.context or Context()).model_factory or triage_model


def build_graph():
    def load_report(state: State, config: RunnableConfig, runtime: Runtime[Context]):
        store = store_for_run(config, runtime)
        report_id = state.get("report_id", 1)
        if "message" in state:
            report_id = store.add_report(state["location"], state.get("subject", "Studio report"), state["message"])
        report = store.report(report_id)
        if "expected_version" in state and (report["version"] != state["expected_version"]
                                             or report["triage_claim"] != state.get("expected_claim")):
            raise ReviewError("Triage claim changed. Reload the report.")
        if report["result"] and not state.get("existing"):
            raise ReviewError("This report is already handled.")
        if state.get("existing") or ("message" in state and report["proposal"]):
            if not report["proposal"]:
                raise ReviewError("This report has no current proposal.")
            return {"report_id": report_id, "report": report, "proposal": report["proposal"],
                    "version": report["version"], "existing": True}
        return {"report_id": report_id, "report": report}

    def safety_gate(state: State):
        return {"signals": sorted(safety_signals(state["report"]["message"])),
                "expires": time() + DEADLINE_SECONDS}

    def retrieve(state: State, config: RunnableConfig, runtime: Runtime[Context]):
        context = retrieve_context(store_for_run(config, runtime), state["report"])
        run = get_current_run_tree()
        if run:
            run.add_metadata({"graph_version": GRAPH_VERSION, "prompt_sha256": PROMPT_HASH,
                              "policy_revisions": {rule["id"]: rule["revision"] for rule in context["policies"].values()}})
        return {"context": context}

    def urgent(state: State):
        proposal = build_proposal(state["report"], state["context"], None, signals=state["signals"])
        return {"proposal": dump_proposal(proposal)}

    def check_safety(state: State, runtime: Runtime[Context]):
        message = state["report"]["message"]
        try:
            result = invoke_structured(model_for_run(runtime), state["expires"], SafetyCheck, [
                SystemMessage(SAFETY_PROMPT),
                HumanMessage(json.dumps({"resident_report": message}, ensure_ascii=False)),
            ])
            check = SafetyCheck.model_validate(result)
        except Exception as exc:
            logging.warning("Safety screening unavailable for report %s (%s)", state["report_id"], type(exc).__name__)
            return {"safety_review": ["unverified_safety"], "safety_evidence": None}
        if check.hazard is None:  # Small models often return "" rather than null evidence.
            return {"safety_review": [], "safety_evidence": None}
        if check.hazard and check.evidence and check.evidence.strip() and check.evidence in message:
            return {"safety_review": [check.hazard], "safety_evidence": check.evidence}
        return {"safety_review": ["unverified_safety"], "safety_evidence": None}

    def interpret(state: State, config: RunnableConfig, runtime: Runtime[Context]):
        if context_issue(state["context"]):
            return {"interpretation": None, "sections": [], "model_error": ""}
        report = state["report"]
        if state["expires"] > time():
            store_for_run(config, runtime).record_triage_attempt(report["id"], report["version"])
        try:
            result, sections, nominated = interpret_with_tools(
                model_for_run(runtime), report, state["context"], state["expires"])
        except Exception as exc:
            if (transient_error(exc) and state["expires"] > time()
                    and runtime.execution_info.node_attempt < MAX_ATTEMPTS):
                raise
            logging.warning("Triage model failed for report %s (%s)", report["id"], type(exc).__name__)
            error = ("timed out" if state["expires"] <= time() or model_timeout(exc) else
                     "model unavailable" if transient_error(exc) else "unparseable model output")
            return {"interpretation": None, "sections": [], "nominated_job_ids": None, "model_error": error}
        return {"interpretation": result.model_dump() if result else None,
                "sections": sections, "nominated_job_ids": nominated,
                "model_error": "" if result else "unparseable model output"}

    def policy(state: State):
        result = Interpretation.model_validate(state["interpretation"]) if state["interpretation"] else None
        proposal = build_proposal(state["report"], state["context"], result,
                                  signals=state["safety_review"], safety_evidence=state.get("safety_evidence"),
                                  model_error=state.get("model_error", ""))
        proposal["retrieval"]["section_ids"] = [section["id"] for section in state["sections"]]
        return {"proposal": dump_proposal(proposal)}

    def draft_reply(state: State, runtime: Runtime[Context]):
        proposal = parse_proposal(state["proposal"])
        if (not (runtime.context or Context()).draft_replies or not state["sections"]
                or proposal.kind not in {"draft", "escalate", "create"} or state["expires"] <= time()):
            return {}
        try:
            draft = ReplyDraft.model_validate(invoke_structured(
                model_for_run(runtime), state["expires"], ReplyDraft, [
                    SystemMessage(DRAFT_PROMPT),
                    HumanMessage(json.dumps({"report": state["report"]["message"], "job": state["proposal"].get("job"),
                                             "sections": state["sections"]}, ensure_ascii=False)),
                ]))
        except Exception:
            return {}  # Preserve a verified proposal and any deterministic reply.
        if draft and grounded_reply(draft.body, draft.section_ids, proposal.retrieval.section_ids,
                                    state["proposal"].get("job")):
            return {"proposal": state["proposal"] | {"reply": draft.body, "reply_sources": draft.section_ids}}
        if draft and (run := get_current_run_tree()):
            run.add_metadata({"draft_rejected_by_grounding": True})
        return {}

    def publish(state: State, config: RunnableConfig, runtime: Runtime[Context]):
        store = store_for_run(config, runtime)
        report = state["report"]
        thread_id = config["configurable"]["thread_id"]
        proposal = state["proposal"] | {"review_thread_id": thread_id}
        current = store.report(report["id"])
        if (current["version"] == report["version"] + 1 and current["proposal"]
                and current["proposal"].get("review_thread_id") == thread_id):
            return {"proposal": current["proposal"], "version": current["version"]}
        run = get_current_run_tree()
        store.save_proposal(report, proposal, str(run.trace_id) if run else None,
                            interpretation=state.get("interpretation"))
        return {"proposal": proposal, "version": report["version"] + 1}

    @traceable(name="await_review")
    def await_review(state: State):
        decision = interrupt({"report_id": state["report_id"], "version": state["version"],
                              "proposal": state["proposal"], "error": state.get("error") or None})
        if isinstance(decision, str):
            decision = {"action": "finish", "decision": decision, "version": state["version"]}
        return {"decision": decision, "error": ""}

    @traceable(name="apply_decision")
    def apply_decision(state: State, config: RunnableConfig, runtime: Runtime[Context]):
        store = store_for_run(config, runtime)
        decision = state["decision"]
        report_id = state["report_id"]
        try:
            if not isinstance(decision, dict) or decision.get("version") != state["version"]:
                raise ReviewError("Review changed. Reload the report.")
            version = state["version"]
            if decision["action"] != "reply" and state["report"]["result"]:
                raise ReviewError("This report is already handled.")
            if decision["action"] == "finish":
                receipt = store.finish(report_id, version, decision["decision"], search_rules,
                                       followup=decision.get("followup", ""), reviewer_id=decision.get("reviewer_id"))
                return Command(update={"receipt": receipt}, goto=END)
            if decision["action"] == "reply":
                store.add_ticket_message(report_id, "staff", decision["body"], draft_version=version,
                                         decline=decision.get("decline", False), reviewer_id=decision.get("reviewer_id"))
                return Command(goto="await_review")
            if decision["action"] == "revise":
                report = store.report(report_id)
                if (report["version"] == version + 1 and report["proposal"] and not report["result"]
                        and report["proposal"].get("review_thread_id") == config["configurable"]["thread_id"]):
                    return Command(update={"proposal": report["proposal"], "version": version + 1}, goto="await_review")
                if report["version"] != version or report["result"]:
                    raise ReviewError("Review changed. Reload the report.")
                value = decision.get("value")
                if isinstance(value, dict) and isinstance(value.get("due_date"), str):
                    value = value | {"due_date": date.fromisoformat(value["due_date"])}
                proposal = parse_proposal(report["proposal"])
                updated = dump_proposal(revise_proposal(store, report, proposal, value))
                store.save_proposal(report, updated, reviewer_id=decision.get("reviewer_id"))
                return Command(update={"proposal": updated, "version": version + 1}, goto="await_review")
            raise ReviewError("Invalid review decision.")
        except (ReviewError, ValueError, KeyError) as exc:
            return Command(update={"error": str(exc)}, goto="await_review")

    flow = StateGraph(State, context_schema=Context)
    flow.add_node("load_report", load_report)
    flow.add_node("safety_gate", safety_gate)
    flow.add_node("retrieve", retrieve)
    flow.add_node("urgent", urgent)
    flow.add_node("check_safety", check_safety)
    flow.add_node("interpret", interpret, retry_policy=RetryPolicy(
        initial_interval=0.3, max_attempts=MAX_ATTEMPTS, jitter=False, retry_on=transient_error))
    flow.add_node("apply_policy", policy)
    flow.add_node("draft_reply", draft_reply)
    flow.add_node("publish", publish)
    flow.add_node("await_review", await_review)
    flow.add_node("apply_decision", apply_decision, destinations=("await_review", END))
    flow.add_edge(START, "load_report")
    flow.add_conditional_edges("load_report", lambda state: "await_review" if state.get("existing") else "safety_gate",
                               ["await_review", "safety_gate"])
    flow.add_edge("safety_gate", "retrieve")
    flow.add_conditional_edges("retrieve", lambda state: "urgent" if state["signals"] else ["interpret", "check_safety"],
                               ["urgent", "interpret", "check_safety"])
    flow.add_edge("urgent", "publish")
    flow.add_edge(["interpret", "check_safety"], "apply_policy")
    flow.add_edge("apply_policy", "draft_reply")
    flow.add_edge("draft_reply", "publish")
    flow.add_edge("publish", "await_review")
    flow.add_edge("await_review", "apply_decision")
    return flow


# One compiled graph per workspace keeps checkpoints beside its reports,
# including for isolated evaluation databases.
_graphs = {}
_graphs_lock = Lock()


@register
def close_checkpointers():
    for compiled in _graphs.values():
        compiled.checkpointer.conn.close()


def graph_for(store):
    path = Path(store.path).resolve()
    with _graphs_lock:
        for removed in [key for key in _graphs if not key.exists()]:
            _graphs.pop(removed).checkpointer.conn.close()
        if path not in _graphs:
            connection = connect(path, check_same_thread=False, timeout=2)
            saver = SqliteSaver(connection)
            try:
                saver.setup()
            except BaseException:
                connection.close()
                raise
            _graphs[path] = build_graph().compile(checkpointer=saver, name="unihelp")
        return _graphs[path]


studio_workspace = TemporaryDirectory(prefix="unihelp-studio-")


def studio_store(config: RunnableConfig):
    thread_id = config["configurable"]["thread_id"]
    store = Store(Path(studio_workspace.name) / f"{uuid5(NAMESPACE_URL, thread_id)}.sqlite3")
    store.seed()
    return store


graph = build_graph().compile(name="unihelp")
