"""Entry points shared by the inbox, worker and evaluation runner."""

import logging
from datetime import date
from uuid import NAMESPACE_URL, uuid5

from langgraph.types import Command
from langsmith import traceable, tracing_context
from langsmith.utils import tracing_is_enabled

from halldesk.context import search_rules
from halldesk.graph import GRAPH_VERSION, PROMPT_HASH, Context, graph_for, retrieve_context, revision_thread
from halldesk.models import triage_model
from halldesk.policy import build_proposal
from halldesk.proposals import parse_proposal
from halldesk.store import ReviewError
from halldesk.tracing import trace_client


def process_pending(store):
    pending = store.claim_triage()
    if not pending:
        return False
    report_id, claim = pending
    try:
        triage(store, report_id, claim=claim)
    except Exception:
        logging.exception("Automatic triage failed for report %s", report_id)
        store.fail_triage(report_id, claim)
    return True


def _trace_options(**metadata):
    options = {"metadata": {"graph_version": GRAPH_VERSION, "prompt_sha256": PROMPT_HASH, **metadata}}
    if tracing_is_enabled() is True:
        options["client"] = trace_client()
    return options


def triage(store, report_id, model=None, *, claim=None):
    report = store.report(report_id)
    if report["triage_claim"] != claim:
        raise ReviewError("This report is already being triaged.")
    if report["result"] or report["proposal"]:
        return report["result"] or report["proposal"]
    return _triage(store, report, model, langsmith_extra=_trace_options(
        report_id=report_id, review_version=report["version"] + 1))


@traceable(name="triage_report")
def _triage(store, report, model=None):
    factory = triage_model if model is None else lambda **_: model
    config = {"configurable": {"thread_id": revision_thread(report["id"], report["version"] + 1)}}
    state = graph_for(store).invoke(
        {"report_id": report["id"], "expected_version": report["version"], "expected_claim": report["triage_claim"]},
        config, context=Context(store, factory, draft_replies=model is None), durability="sync")
    return state["proposal"]


def _decision(store, report, version, action, **values):
    if report["version"] != version or not report["proposal"]:
        raise ReviewError("Review changed. Reload the report.")
    proposal = parse_proposal(report["proposal"])
    thread_id = proposal.review_thread_id
    if not thread_id or not thread_id.startswith("unihelp-"):
        thread_id = revision_thread(report["id"], version)
    if action == "reply" and report["result"]:
        thread_id += "-reply"
    config = {"configurable": {"thread_id": thread_id}}
    graph, context = graph_for(store), Context(store)
    snapshot = graph.get_state(config)
    if not snapshot.values:
        graph.invoke({"report_id": report["id"], "existing": True}, config, context=context, durability="sync")
    elif not snapshot.interrupts and snapshot.next:
        graph.invoke(None, config, context=context, durability="sync")
    elif not snapshot.interrupts:
        if action == "finish":
            return store.finish(report["id"], version, values["decision"], search_rules,
                                followup=values.get("followup", ""), reviewer_id=values.get("reviewer_id"))
        raise ReviewError("Review changed. Reload the report.")
    result = graph.invoke(Command(resume={"action": action, "version": version, **values}),
                          config, context=context, durability="sync")
    if result.get("error"):
        raise ReviewError(result["error"])
    return result


def feedback(report, version, value, reviewer_id):
    if not report.get("trace_id") or tracing_is_enabled() is not True:
        return
    try:
        trace_client().create_feedback(
            report["trace_id"], "staff_review", value=value,
            feedback_id=uuid5(NAMESPACE_URL, f"{report['trace_id']}:{version}:{value}"),
            source_info={"reviewer_id": reviewer_id or "staff"},
        )
    except Exception as exc:
        logging.warning("Could not send review feedback (%s)", type(exc).__name__)


def revise(store, report_id, version, value, *, reviewer_id=None):
    report = store.report(report_id)
    if report["result"]:
        raise ReviewError("Review changed. Reload the report.")
    if isinstance(value, dict) and type(value.get("due_date")) is date:
        value = value | {"due_date": value["due_date"].isoformat()}
    updated = _decision(store, report, version, "revise", value=value, reviewer_id=reviewer_id)["proposal"]
    feedback(report, version, "edited", store.staff_id(reviewer_id))
    return updated


def review_reply(store, report_id, version, body, *, decline=False, reviewer_id=None):
    report = store.report(report_id)
    result = _decision(store, report, version, "reply", body=body, decline=decline, reviewer_id=reviewer_id)
    proposal = parse_proposal(report["proposal"])
    original = proposal.reply or getattr(proposal, "question", None)
    feedback(report, version, "reply_rejected" if decline else "reply_edited" if body != original else "reply_approved",
             store.staff_id(reviewer_id))
    return result


def review(store, report_id, version, decision, *, followup="", reviewer_id=None):
    report = store.report(report_id)
    receipt = _review(store, report_id, version, decision, followup=followup, reviewer_id=reviewer_id,
                      langsmith_extra=_trace_options(report_id=report_id, review_version=version,
                                                     triage_trace_id=report["trace_id"]))
    if not report["result"]:
        feedback(report, version, "rejected" if decision == "reject" else "approved", store.staff_id(reviewer_id))
    return receipt


@traceable(name="review_report")
def _review(store, report_id, version, decision, *, followup="", reviewer_id=None):
    report = store.report(report_id)
    if report["result"]:
        return store.finish(report_id, version, decision, search_rules, followup=followup, reviewer_id=reviewer_id)
    result = _decision(store, report, version, "finish", decision=decision,
                       followup=followup, reviewer_id=reviewer_id)
    return result.get("receipt", result)
