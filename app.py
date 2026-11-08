import os
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st
from dotenv import load_dotenv

from halldesk.context import retrieve_policy
from halldesk.models import configured_provider, model_id, setup_issue
from halldesk.store import HANDOFFS, LABELS, RESIDENCE, ReviewError, Store, is_overdue
from halldesk.workflow import process_pending, review, revise

load_dotenv()
st.set_page_config(page_title="HallDesk", page_icon=":material/mail:", layout="wide")
st.html(Path(__file__).with_name("halldesk") / "mail.css")
store = Store(os.getenv("HALL_DB", "halldesk.sqlite3"))
store.seed(samples=os.getenv("HALL_SEED_SAMPLE", "").lower() == "true")
background_worker = os.getenv("HALL_BACKGROUND_WORKER", "").lower() == "true"

FOLDERS = ("Inbox", "Needs review", "Handled")
HANDOFF_LABELS = {
    "escalate": "Escalate a job", "draft": "Prepare a draft", "create": "Create a job",
    "job_question": "Ask which job", "lighting_question": "Ask about other lighting", "manual": "Manual review",
}
provider = configured_provider()
configuration_issue = setup_issue(provider)


def london_time(timestamp, date_format="%d %b %Y, %H:%M %Z"):
    return datetime.fromisoformat(timestamp).astimezone(ZoneInfo("Europe/London")).strftime(date_format)


def job_line(job):
    overdue = " · Overdue" if is_overdue(job, store.clock()) else ""
    return f"#{job['id']} · {job['fitting']} · {job['priority']} · due {london_time(job['deadline'])}{overdue}"


def status(report):
    if report["result"]:
        return "Handled"
    if report["triage_error"]:
        return "Triage failed"
    if not report["proposal"]:
        return "Triaging" if report["triage_claim"] else "Queued"
    return {
        "escalate": "Approval needed", "create": "Approval needed", "draft": "Draft ready",
        "job_question": "Clarification", "lighting_question": "Clarification", "manual": "Manual review",
    }[report["proposal"]["kind"]]


def markdown_text(value):
    return re.sub(r"([\\`*_{}\[\]<>()#+.!|~-])", r"\\\1", " ".join(value.split()))


def close_compose():
    st.session_state.pop("composing", None)


@st.dialog("New report", on_dismiss=close_compose)
def compose():
    with st.form("compose", border=False):
        st.caption(f"{RESIDENCE} · Enter a verified room or shared area.")
        location = st.text_input("Location", key="compose-location", placeholder="Flat 12 kitchen", max_chars=100)
        subject = st.text_input("Subject", key="compose-subject", placeholder="Kitchen ceiling light has stopped working", max_chars=120)
        message = st.text_area("Resident’s report", key="compose-message", max_chars=4000, height=160)
        if st.form_submit_button("Add to inbox", type="primary", icon=":material/inbox:"):
            try:
                report_id = store.add_report(location, subject, message)
            except ReviewError as exc:
                st.error(str(exc))
                return
            close_compose()
            st.session_state.open_report = report_id
            st.rerun()


def show_review(report):
    report_id, version = report["id"], report["version"]
    proposal = report["proposal"]

    def change(value):
        try:
            revise(store, report_id, version, value)
            st.rerun()
        except ReviewError as exc:
            st.error(str(exc))

    with st.container(border=True):
        st.caption("HALLDESK HANDOFF")
        kind = proposal["kind"]
        if kind == "escalate":
            st.markdown(f"#### Escalate to {proposal['rule']['destination']}")
        elif kind == "draft":
            st.markdown("#### Draft reply")
            st.info(f"The repair is open and due by {london_time(proposal['job']['deadline'])}.")
        elif kind == "create":
            st.markdown("#### Create a repair job")
            st.text(f"{proposal['fitting']} · {proposal['priority']} · due in {proposal['days']} days from approval")
            labels = st.multiselect("Job labels", LABELS, default=proposal["labels"],
                                    key=f"labels-{report_id}-{version}")
            if labels != proposal["labels"] and st.button("Save label changes"):
                change(labels)
        elif kind == "job_question":
            st.markdown("#### Confirm the matching job")
            st.write(proposal["question"])
            for job in proposal["candidates"]:
                st.text(job_line(job))
                st.caption(", ".join(job["labels"]))
                if st.button(f"Select job #{job['id']}"):
                    change(job["id"])
        elif kind == "lighting_question":
            st.markdown("#### Confirm the other lighting")
            st.write(proposal["question"])
            if st.button("Yes, another light works"):
                change(True)
            if st.button("No, this was the only working light"):
                change(False)
        else:
            st.markdown("#### Review manually")
            st.warning(proposal["reason"])

        if "job" in proposal:
            st.text(job_line(proposal["job"]))
            st.text(f"Report evidence · {proposal['evidence']}")
        if "rule" in proposal:
            rule = proposal["rule"]
            with st.expander(f"Policy · {rule['id']} · v{rule['revision']}"):
                st.write(rule["summary"])
                st.text(rule["source"])

        if kind in {"job_question", "lighting_question"}:
            approve = False
            if st.button("Unable to confirm"):
                change(None)
        else:
            label = {"escalate": "Approve escalation", "create": "Approve new job"}.get(kind, "Mark handled")
            approve = st.button(label, type="primary",
                                disabled=kind == "create" and labels != proposal["labels"])
        reject = st.button("Reject")
        if approve or reject:
            decision = "reject" if reject else ("approve" if kind in {"create", "escalate"} else "acknowledge")
            try:
                review(store, report_id, version, decision)
                st.rerun()
            except ReviewError as exc:
                st.error(str(exc))
                st.session_state[f"refresh-{report_id}"] = True
        if st.session_state.get(f"refresh-{report_id}") and st.button("Refresh proposal"):
            try:
                store.requeue_review(report_id, version)
                st.session_state.pop(f"refresh-{report_id}", None)
                st.rerun()
            except ReviewError as exc:
                st.error(str(exc))


def show_jobs():
    st.title("Jobs")
    st.caption("Open lighting jobs and their approval history")
    search = st.text_input("Search by job ID, location, or fitting").strip().casefold()
    category = st.selectbox("Category", ["All", "lighting"])
    label = st.selectbox("Label", ["All", *LABELS])
    rows = [
        {"ID": j["id"], "Location": j["location"], "Fitting": j["fitting"], "Priority": j["priority"],
         "Deadline": london_time(j["deadline"]), "Status": "Overdue" if is_overdue(j, store.clock()) else j["status"],
         "Labels": ", ".join(j["labels"]), "Escalated at": london_time(j["escalated_at"]) if j["escalated_at"] else None}
        for j in store.jobs()
        if search in f"{j['id']} {j['location']} {j['fitting']}".casefold()
        and category in {"All", j["category"]}
        and (label == "All" or label in j["labels"])
    ]
    st.dataframe(rows, hide_index=True, width="stretch")


def show_policy():
    st.subheader("Policy pack")
    policies = retrieve_policy()
    if not policies:
        st.error("The configured policy pack is missing or invalid.")
    for rule in policies.values():
        with st.expander(f"{rule['id']} · v{rule['revision']}", expanded=True):
            st.text(rule["summary"])
            st.text(rule["source"])
            st.json({key: rule[key] for key in ("other_lighting_works", "no_other_lighting",
                                               "boundary", "destination") if key in rule})


def show_evals():
    st.subheader("Evaluation review")
    st.caption("Holdouts measure changes. Approved guidance can appear in future triage context.")
    cases = store.eval_cases()
    if not cases:
        st.info("Open a report and use Save evaluation draft to label your first case.")
    for case in cases:
        with st.expander(f"#{case['id']} · {markdown_text(case['name'])} · {case['status'].title()}"):
            st.text(case["snapshot"]["message"])
            st.text(case["notes"])
            st.json(case["expected"])
            st.caption(f"Captured {london_time(case['snapshot']['at'])} · {len(case['snapshot']['jobs'])} open jobs")
            choices = {"Approve as holdout": "holdout", "Approve as guidance": "guidance",
                       "Retire": "retired"} if case["status"] == "draft" else (
                       {"Retire": "retired"} if case["status"] != "retired" else {})
            with st.container(horizontal=True):
                for label, value in choices.items():
                    if st.button(label, key=f"eval-{case['id']}-{value}"):
                        try:
                            store.decide_eval(case["id"], value)
                            st.rerun()
                        except ReviewError as exc:
                            st.error(str(exc))


def label_report(report):
    with st.expander("Save evaluation draft"):
        st.caption("Choose the result you expect. The model’s proposal does not set the label.")
        kind = st.selectbox("Expected handoff", HANDOFFS, index=None, format_func=HANDOFF_LABELS.get,
                            key=f"expected-{report['id']}")
        snapshot = (report["proposal"] or {}).get("context_snapshot")
        jobs = {job["id"]: job for job in (snapshot["jobs"] if snapshot else store.jobs(report["location"]))}
        expected = {"kind": kind}
        if kind in {"escalate", "draft"}:
            expected["job"] = st.selectbox("Expected job", jobs, index=None,
                                           format_func=lambda job_id: job_line(jobs[job_id]))
        elif kind == "job_question":
            expected["candidates"] = st.multiselect("Expected candidates", jobs,
                                                    format_func=lambda job_id: job_line(jobs[job_id]))
        elif kind == "create":
            expected["priority"] = st.selectbox("Expected priority", ["P2", "P3"], index=None)
            expected["days"] = st.number_input("Expected target in days", min_value=1, max_value=90)
        notes = st.text_area("Reviewer notes", max_chars=1200, key=f"eval-notes-{report['id']}")
        if st.button("Save draft", disabled=not kind or (kind == "create" and not expected["priority"])):
            try:
                store.save_eval_draft(report, expected, notes)
                st.success("Draft saved. Open Evals to approve it.")
            except ReviewError as exc:
                st.error(str(exc))


if opened := st.session_state.pop("open_report", None):
    st.session_state.workspace = "Reports"
    st.session_state.mailboxes = "Inbox"
    st.session_state.pop("report-search", None)
    st.session_state["message-list"] = opened

all_reports = store.reports()
counts = {
    "Inbox": len(all_reports),
    "Needs review": sum(bool(r["proposal"]) and not r["result"] for r in all_reports),
    "Handled": sum(bool(r["result"]) for r in all_reports),
}
with st.container(key="mail-header"):
    brand, compose_button = st.columns([4, 1], vertical_alignment="center")
    with brand:
        st.title("HallDesk")
        st.caption(f"{RESIDENCE} · Maintenance workspace")
    with compose_button:
        if st.button("New report", icon=":material/edit_square:", width="stretch", type="primary"):
            st.session_state.composing = True
    if st.session_state.get("composing"):
        compose()
workspace = st.segmented_control("Workspace", ["Reports", "Jobs", "Policy", "Evals"],
                                key="workspace", selection_mode="single", label_visibility="collapsed") or "Reports"
if configuration_issue:
    st.caption(configuration_issue)
elif background_worker:
    st.caption("Queue processing is assigned to the background worker.")

if workspace == "Jobs":
    show_jobs()
elif workspace == "Policy":
    show_policy()
elif workspace == "Evals":
    show_evals()
else:
    queue, detail = st.columns([0.95, 1.75], gap="medium")
    with queue:
        query = st.text_input("Search reports", placeholder="Search subject or location", icon=":material/search:",
                              label_visibility="collapsed", key="report-search").strip().casefold()
        folder = st.segmented_control("Report filters", FOLDERS, key="mailboxes",
                                      format_func=lambda name: f"{name} · {counts[name]}",
                                      label_visibility="collapsed") or "Inbox"
        reports = {
            r["id"]: r for r in all_reports
            if (folder == "Inbox" or folder == "Handled" and r["result"]
                or folder == "Needs review" and r["proposal"] and not r["result"])
            and query in f"{r['location']} {r['subject']} {r['message']}".casefold()
        }

        def mail_row(report_id):
            report = reports[report_id]
            subject = markdown_text(report["subject"] or "Resident report")
            preview = markdown_text(report["message"][:68])
            return f"**{subject}**  \n{preview}  \n*{status(report)} · #{report_id}*"

        if st.session_state.get("message-list") not in reports:
            st.session_state["message-list"] = next(iter(reports), None)
        with st.container(height=610, border=True):
            selected = st.radio("Reports", reports, index=None, key="message-list",
                                label_visibility="collapsed", format_func=mail_row) if reports else None
            if not reports:
                st.caption("No matching reports.")
    with detail, st.container(key="reading-pane", border=True):
        if selected is None:
            st.info("Select a report to see the message and handoff.")
        else:
            report = reports[selected]
            st.caption(f"REPORT #{selected} · {status(report).upper()}")
            st.header(markdown_text(report["subject"] or "Resident report"))
            received = london_time(report["received_at"]) if report["received_at"] else ""
            st.caption(f"{markdown_text(report['location'])} · {received}")
            st.text(report["message"])
            st.divider()
            if report["result"]:
                receipt = report["result"]
                st.success(receipt["action"].replace("_", " ").capitalize())
                if receipt.get("job_id"):
                    st.write(f"Job #{receipt['job_id']} · {receipt.get('destination', 'Lighting')}")
            elif report["proposal"]:
                show_review(report)
            elif report["triage_error"]:
                st.error(report["triage_error"])
                if st.button("Retry automatic triage", disabled=bool(configuration_issue)):
                    store.retry_triage(selected)
                    st.rerun()
            else:
                st.info(configuration_issue or "HallDesk will prepare a handoff automatically.")
            if retrieval := (report["proposal"] or {}).get("retrieval"):
                with st.expander("Context used"):
                    policies = ", ".join(f"{rule['id']} v{rule['revision']}" for rule in retrieval["policies"])
                    examples = ", ".join(f"#{case_id}" for case_id in retrieval["example_ids"])
                    st.text(policies or "No valid policy retrieved.")
                    st.text(f"Approved guidance: {examples or 'none matched'}")
            label_report(report)
            if report["trace_id"]:
                with st.expander("LangSmith trace ID"):
                    st.code(report["trace_id"], language=None)


@st.fragment(run_every=3)
def automatic_triage():
    if st.session_state.get("composing"):
        return
    if (not configuration_issue and not background_worker and process_pending(store)) or store.reports() != all_reports:
        st.rerun()


automatic_triage()
