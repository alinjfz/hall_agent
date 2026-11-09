from contextlib import contextmanager
from sqlite3 import OperationalError, SQLITE_BUSY, SQLITE_LOCKED
from time import monotonic, sleep
from typing import TypedDict
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from halldesk.context import search_rules
from halldesk.store import ReviewError


class ChaseReview(TypedDict, total=False):
    report_id: int
    version: int
    job_id: int
    decision: str
    receipt: dict
    error: str
    location: str
    subject: str
    message: str
    report: dict
    context: dict
    interpretation: dict | None
    proposal: dict


def await_review(state: ChaseReview):
    decision = interrupt({
        "action": "escalate",
        "report_id": state["report_id"],
        "version": state["version"],
        "job_id": state["job_id"],
        **({"error": state["error"]} if state.get("error") else {}),
    })
    return {"decision": decision, "error": ""}


def build_review_graph(store_for_run, *, triage_steps=(), input_schema=None):
    def apply_decision(state: ChaseReview, config: RunnableConfig):
        store = store_for_run(config)
        try:
            receipt = store.finish(state["report_id"], state["version"], state["decision"], search_rules)
        except ReviewError as exc:
            return Command(update={"error": str(exc)}, goto="await_review")
        return Command(update={"receipt": receipt}, goto=END)

    graph = StateGraph(ChaseReview, input_schema=input_schema)
    graph.add_node("await_review", await_review)
    graph.add_node("apply_decision", apply_decision, destinations=("await_review", END))
    if triage_steps:
        graph.add_sequence(triage_steps)
        graph.add_edge(START, triage_steps[0][0])
        graph.add_conditional_edges(
            triage_steps[-1][0],
            lambda state: "await_review" if state["proposal"]["kind"] == "escalate" else END,
            ["await_review", END],
        )
    else:
        graph.add_edge(START, "await_review")
    graph.add_edge("await_review", "apply_decision")
    return graph


@contextmanager
def review_graph(store):
    graph = build_review_graph(lambda config: store)
    with SqliteSaver.from_conn_string(str(store.path)) as checkpointer:
        # Initial WAL setup can race with an inbox read. Retry setup before any
        # graph node runs so an approval or escalation is never replayed here.
        deadline = monotonic() + 2
        while True:
            try:
                checkpointer.setup()
                break
            except OperationalError as exc:
                code = getattr(exc, "sqlite_errorcode", 0) & 0xFF
                if code not in {SQLITE_BUSY, SQLITE_LOCKED} or monotonic() >= deadline:
                    raise
                sleep(0.05)
        yield graph.compile(checkpointer=checkpointer, name="chase_review")


def pause_chase(store, report, proposal):
    thread_id = f"chase-{report['id']}-{report['version'] + 1}-{uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id}}
    # Create the checkpoint before publishing a proposal that points to it.
    with review_graph(store) as graph:
        graph.invoke({
            "report_id": report["id"],
            "version": report["version"] + 1,
            "job_id": proposal["job"]["id"],
        }, config, durability="sync")
    return thread_id


def resume_chase(store, report, decision):
    config = {"configurable": {"thread_id": report["proposal"]["review_thread_id"]}}
    with review_graph(store) as graph:
        state = graph.get_state(config)
        if state.interrupts:
            result = graph.invoke(Command(resume=decision), config, durability="sync")
        elif state.values.get("decision") == decision:
            # A process can stop after the decision is saved and before the write.
            result = graph.invoke(None, config, durability="sync") if state.next else state.values
        else:
            raise ReviewError("This review has already advanced. Reload the report.")
    if result.get("error"):
        raise ReviewError(result["error"])
    return result["receipt"]
