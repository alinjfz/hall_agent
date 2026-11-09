from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict
from uuid import NAMESPACE_URL, uuid5

from langchain_core.runnables import RunnableConfig
from langsmith import get_current_run_tree

from halldesk.chase_review import ChaseReview, build_review_graph
from halldesk.store import ReviewError, Store
from halldesk.workflow import Interpretation, build_proposal, interpret_report, retrieve_context


class StudioInput(TypedDict, total=False):
    report_id: int
    location: str
    subject: str
    message: str


# Agent Server checkpoints the run. SQLite holds the report, jobs, and receipt.
studio_workspace = TemporaryDirectory(prefix="halldesk-studio-")


def studio_store(config: RunnableConfig):
    thread_id = config["configurable"]["thread_id"]
    database = Path(studio_workspace.name) / f"{uuid5(NAMESPACE_URL, thread_id)}.sqlite3"
    return Store(database)


def load_report(state: ChaseReview, config: RunnableConfig):
    store = studio_store(config)
    store.seed()
    if "message" in state:
        report_id = store.add_report(state.get("location", ""), state.get("subject", "Studio report"), state["message"])
    else:
        report_id = state.get("report_id", 1)
    try:
        report = store.report(report_id)
    except StopIteration:
        raise ReviewError(f"Report #{report_id} was not found.") from None
    if report["result"] or report["proposal"]:
        raise ReviewError("Start a new Studio thread to triage this report again.")
    return {"report_id": report_id, "report": report}


def retrieve_policy_and_jobs(state: ChaseReview, config: RunnableConfig):
    return {"context": retrieve_context(studio_store(config), state["report"])}


def interpret(state: ChaseReview):
    result = interpret_report(state["report"], state["context"])
    return {"interpretation": result.model_dump() if result is not None else None}


def propose(state: ChaseReview, config: RunnableConfig):
    interpretation = state["interpretation"]
    parsed = Interpretation.model_validate(interpretation) if interpretation is not None else None
    report = state["report"]
    proposal = build_proposal(report, state["context"], parsed)
    run = get_current_run_tree()
    studio_store(config).save_proposal(report, proposal, str(run.trace_id) if run else None)
    return {"proposal": proposal, "version": report["version"] + 1,
            "job_id": (proposal.get("job") or {}).get("id")}


graph = build_review_graph(
    studio_store,
    input_schema=StudioInput,
    triage_steps=[
        ("load_report", load_report),
        ("retrieve_policy_and_jobs", retrieve_policy_and_jobs),
        ("interpret_report", interpret),
        ("apply_policy_and_save_proposal", propose),
    ],
).compile(name="chase_review")
