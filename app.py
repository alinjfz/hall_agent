import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from halldesk.context import REPAIR_CATEGORIES, retrieve_policy
from halldesk.proposals import parse_proposal
from halldesk.models import configured_provider, model_id, setup_issue
from halldesk.store import HANDOFFS, LABELS, LONDON, RESIDENCE, ReviewError, Store, is_overdue
from halldesk.workflow import process_pending, review, review_reply, revise

load_dotenv()
st.set_page_config(page_title="UniHelp", page_icon=":material/mail:", layout="wide")
st.html(Path(__file__).with_name("halldesk") / "mail.css")
store = Store(os.getenv("HALL_DB", "halldesk.sqlite3"))
store.seed(samples=os.getenv("HALL_SEED_SAMPLE", "").lower() == "true")
background_worker = os.getenv("HALL_BACKGROUND_WORKER", "").lower() == "true"
quality_checks_enabled = os.getenv("HALL_ENABLE_QUALITY_CHECKS", "").lower() == "true"

FOLDERS = ("Inbox", "Queued", "Needs review", "Handled", "Failed")
QUEUE_STATUSES = {"Queued", "Triaging"}
HANDOFF_LABELS = {
    "escalate": "Escalate a job", "draft": "Prepare a draft", "create": "Create a job",
    "job_question": "Ask which job", "lighting_question": "Ask about other lighting",
    "plumbing_question": "Ask about leak severity",
    "access_question": "Ask about current access", "power_question": "Ask about outage scope",
    "split": "Split into separate reports", "manual": "Manual review",
}
BADGE_COLORS = {"Handled": "green", "Handed over": "green", "Approval needed": "orange", "Draft ready": "blue",
                "Clarification": "violet", "Split suggested": "violet", "Manual review": "blue",
                "Triage failed": "red", "Emergency": "red", "Urgent": "orange"}
POLICY_TITLES = {
    "escalation": "An open repair is overdue",
    "lighting": "A light stops working",
    "heating": "The heating stops working",
    "appliance": "An appliance breaks",
    "access": "Someone is locked out or needs a spare key",
    "multiple": "One report describes several problems",
    "duplicate": "There may already be a job for the problem",
    "urgent": "A building hazard needs urgent attention",
    "fire": "Fire, smoke, alarms and fire safety equipment",
    "gas": "Suspected gas leak or carbon monoxide",
    "plumbing": "A tap or shower is dripping",
    "follow_up": "A repair remains unresolved",
    "review": "A report needs a person to assess it",
}
OUTCOMES = {
    "created_job": "Repair job created", "escalated": "Repair escalated",
    "split_report": "Report split", "draft": "Reply draft marked handled",
    "manual": "Manual review marked handled", "rejected": "Proposal rejected",
    "technician_handoff": "Handed over to technicians",
}
provider = configured_provider()
configuration_issue = setup_issue(provider)


def london_time(timestamp, date_format="%d %b %Y, %H:%M %Z"):
    return datetime.fromisoformat(timestamp).astimezone(LONDON).strftime(date_format)


def date_filter(label, key):
    with st.popover(label, icon=":material/calendar_month:"):
        st.caption("Dates are inclusive, in UK local time.")
        start = st.date_input("From", value=None, key=f"{key}-from", format="DD/MM/YYYY")
        end = st.date_input("To", value=None, key=f"{key}-to", format="DD/MM/YYYY")
        if start and end and start > end:
            st.error("Choose an end date on or after the start date.")
    return start, end


def matches_date(timestamp, bounds):
    start, end = bounds
    if start is None and end is None:
        return True
    if timestamp is None:
        return False
    day = datetime.fromisoformat(timestamp).astimezone(LONDON).date()
    return (start is None or start <= day) and (end is None or day <= end)


def job_line(job):
    overdue = " · Overdue" if is_overdue(job, store.clock()) else ""
    return f"#{job['id']} · {job['fitting']} · {job['priority']} · due {london_time(job['deadline'])}{overdue}"


def status(report):
    if report["result"]:
        if report["result"]["action"] == "technician_handoff":
            return "Handed over"
        return "Handled"
    if report["triage_error"]:
        return "Triage failed"
    if not report["proposal"]:
        return "Triaging" if report["triage_claim"] else "Queued"
    if urgency := report["proposal"].get("urgency"):
        return "Emergency" if urgency == "emergency" else "Urgent"
    return {
        "escalate": "Approval needed", "create": "Approval needed", "draft": "Draft ready",
        "job_question": "Clarification", "lighting_question": "Clarification",
        "plumbing_question": "Clarification", "access_question": "Clarification", "power_question": "Clarification",
        "split": "Split suggested", "manual": "Manual review",
    }[report["proposal"]["kind"]]


def markdown_text(value):
    return re.sub(r"([\\`*_{}\[\]<>()#+.!|~-])", r"\\\1", " ".join(value.split()))


def close_compose():
    st.session_state.pop("composing", None)


@st.dialog("New report", on_dismiss=close_compose)
def compose():
    st.caption(f"{RESIDENCE} · Choose a known location or enter a new verified room or shared area.")
    with st.form("compose", border=False):
        location = st.selectbox("Location", store.locations(), index=None, accept_new_options=True,
                                key="compose-location", placeholder="Select or type a location")
        subject = st.text_input("Subject", key="compose-subject", placeholder="Kitchen ceiling light has stopped working", max_chars=120)
        message = st.text_area("Resident’s report", key="compose-message", max_chars=4000, height=160)
        if st.form_submit_button("Add to inbox", type="primary", icon=":material/inbox:", width="stretch"):
            try:
                report_id = store.add_report(location, subject, message)
            except ReviewError as exc:
                st.error(str(exc))
                return
            close_compose()
            st.session_state.open_report = report_id
            st.rerun()


def show_previous_edits(proposal, reports):
    similar = [
        report for report in reports
        if (result := report["result"]) and result.get("reviewer_edits")
        and (previous := report["proposal"]) and previous["kind"] == "create"
        and previous.get("category", "lighting") == proposal.get("category", "lighting")
        and previous["fitting"].casefold() == proposal["fitting"].casefold()
    ]
    if not similar:
        return
    previous = max(similar, key=lambda report: report["result"]["at"])
    result = previous["result"]
    edits = result["reviewer_edits"]
    notes = []
    if "labels" in edits:
        notes.append("Tags: " + (", ".join(tag.replace("_", " ") for tag in edits["labels"]) or "none"))
    if due_date := edits.get("due_date"):
        approved_on = datetime.fromisoformat(result["at"]).astimezone(LONDON).date()
        days = (date.fromisoformat(due_date) - approved_on).days
        notes.append("Due the same day" if not days else f"Due {days} day{'s' if days != 1 else ''} after approval")
    st.caption(f"Previous review for the same fitting · Job #{result['job_id']} · " + " · ".join(notes))


def show_review(report, reports):
    report_id, version = report["id"], report["version"]
    proposal = report["proposal"]
    followup = ""

    def change(value):
        try:
            revise(store, report_id, version, value)
            st.rerun()
        except ReviewError as exc:
            st.error(str(exc))

    with st.container(border=True):
        st.caption("PROPOSED HANDOFF · INTERNAL REVIEW")
        kind = parse_proposal(proposal).kind
        if kind == "escalate":
            st.markdown(f"#### Escalate to {proposal['rule']['destination']}")
        elif kind == "draft":
            st.markdown("#### Draft reply")
            st.info(f"The repair is open and due by {london_time(proposal['job']['deadline'])}.")
        elif kind == "create":
            st.markdown("#### Create a repair job")
            st.caption(proposal.get("category", "lighting").capitalize())
            if target := proposal.get("policy_deadline"):
                target_label = f"policy repair target {london_time(target)}"
            else:
                target_label = f"policy target {proposal['days']} days from approval"
            st.text(f"{proposal['fitting']} · {proposal['priority']} · {target_label}")
            show_previous_edits(proposal, reports)
            labels = st.multiselect("Job tags", LABELS, default=proposal["labels"],
                                    key=f"labels-{report_id}-{version}", format_func=lambda tag: tag.replace("_", " "))
            saved_date = proposal.get("reviewer_edits", {}).get("due_date")
            saved_date = date.fromisoformat(saved_date) if saved_date else None
            today = store.clock().astimezone(LONDON).date()
            due_date = st.date_input("Deadline override", value=saved_date, min_value=min(today, saved_date or today),
                                     format="DD/MM/YYYY", key=f"deadline-{report_id}-{version}",
                                     help="Uses the end of the selected day in UK time. Clear the date to use the policy target.")
            has_edits = labels != proposal["labels"] or due_date != saved_date
            if has_edits and st.button("Save review edits"):
                change({"labels": labels, "due_date": due_date})
        elif kind == "split":
            st.markdown("#### Separate the reported issues")
            st.caption(f"Confirm each issue belongs at {report['location']}. Each will go through its own review.")
            for index, issue in enumerate(proposal["issues"], start=1):
                st.text(f"{index}. {issue}")
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
            if st.button("Yes, another fixed light works"):
                change(True)
            if st.button("No fixed light works in that room"):
                change(False)
        elif kind == "plumbing_question":
            st.markdown("#### Confirm the leak severity")
            st.write(proposal["question"])
            if st.button("Only dripping"):
                change(True)
            if st.button("Water is spreading"):
                change(False)
        elif kind in {"access_question", "power_question"}:
            st.write(proposal["question"])
            if kind == "access_question":
                if st.button("Resident cannot get in"):
                    change(True)
                if st.button("Resident can get in"):
                    change(False)
            else:
                if st.button("Whole flat or affected area"):
                    change(True)
                if st.button("Only one fitting"):
                    change(False)
        else:
            urgency = proposal.get("urgency")
            if urgency:
                st.error(proposal["reason"])
                if suggestions := proposal.get("followup_suggestions"):
                    st.caption("Possible separate issues. Review before queuing.")
                    for suggestion in suggestions:
                        st.text(suggestion)
                followup = st.text_area("Separate issues from this report (optional)",
                                        help="Copy the exact words for each issue onto its own line. Marking the report as handled or recording a technician handoff queues each excerpt for review.")
            else:
                st.warning(proposal["reason"])
            if destination := proposal.get("destination"):
                st.caption(f"Refer to {destination}")
            for job in proposal.get("related_jobs", []):
                st.text(job_line(job))
            if not proposal.get("related_jobs") and (job_ids := proposal.get("duplicate_job_ids")):
                st.text("Matching open jobs · " + ", ".join(f"#{job_id}" for job_id in job_ids))
            for match in proposal.get("possible_duplicates", []):
                st.caption(f"Job #{match['job_id']} · {match['fitting']} · Report #{match['report_id']}")
                st.text(match["message"])

        if "job" in proposal:
            st.text(job_line(proposal["job"]))
            st.text(f"Report evidence · {proposal['evidence']}")
        if "rule" in proposal:
            rule = proposal["rule"]
            with st.expander(f"Policy · {rule['id']} · v{rule['revision']}"):
                st.write(rule["summary"])

        approve = handoff = reject = False
        if kind in {"draft", "manual"}:
            st.caption("Marking the report as handled completes the internal review. "
                       "Review and post any resident reply separately in the conversation.")
        with st.container(horizontal=True, key=f"action-bar-{report_id}"):
            if kind in {"job_question", "lighting_question", "plumbing_question", "access_question", "power_question"}:
                if st.button("Unable to confirm"):
                    change(None)
            else:
                label = {"escalate": "Approve escalation", "create": "Approve new job",
                         "split": "Confirm split"}.get(kind, "Mark report as handled")
                approve = st.button(label, type="primary", icon=":material/check:",
                                    disabled=kind == "create" and has_edits)
                if kind == "manual" and proposal.get("urgency"):
                    handoff = st.button("Record technician handoff", icon=":material/engineering:",
                                        help="Use after contacting the technicians. This records the handoff in the report history.")
            with st.popover("Reject handoff", icon=":material/close:"):
                st.caption("Rejecting closes the report and discards any unposted reply draft.")
                reject = st.button("Confirm rejection", key=f"reject-{report_id}-{version}")
        if approve or handoff or reject:
            if reject:
                decision = "reject"
            elif handoff:
                decision = "technician_handoff"
            else:
                decision = "approve" if kind in {"create", "escalate", "split"} else "acknowledge"
            try:
                review(store, report_id, version, decision, followup="" if reject else followup)
                st.rerun()
            except ReviewError as exc:
                st.error(str(exc))
                st.session_state[f"refresh-{report_id}"] = True
        can_refresh = st.session_state.get(f"refresh-{report_id}") or (kind == "manual" and not proposal.get("urgency"))
        if can_refresh and st.button("Refresh proposal", disabled=bool(configuration_issue)):
            try:
                store.requeue_review(report_id, version)
                st.session_state.pop(f"refresh-{report_id}", None)
                st.rerun()
            except ReviewError as exc:
                st.error(str(exc))


JOB_STATUSES = {"open": "Open", "assigned": "Assigned", "handled": "Handled", "declined": "Declined"}


@st.fragment
def show_jobs():
    if saved := st.session_state.pop("job-status-saved", None):
        st.toast(saved, icon=":material/check:")
    if error := st.session_state.pop("job-status-error", None):
        st.error(error)

    with st.container(horizontal=True, vertical_alignment="bottom", key="job-filters"):
        search = st.text_input("Search", placeholder="Job ID, location, or fitting",
                               icon=":material/search:").strip().casefold()
        category = st.selectbox("Category", ["All", *REPAIR_CATEGORIES])
        label = st.selectbox("Label", ["All", *LABELS])
        dates = date_filter("Deadline dates", "job-date")
    jobs = {j["id"]: j for j in store.jobs()
        if search in f"{j['id']} {j['location']} {j['fitting']}".casefold()
        and category in {"All", j["category"]}
        and (label == "All" or label in j["labels"])
        and matches_date(j["deadline"], dates)
    }
    if not jobs:
        st.info("No jobs match these filters.")
        return

    visible_jobs = list(jobs.values())
    rows = [
        {"ID": job["id"], "Status": JOB_STATUSES.get(job["status"], job["status"]),
         "Location": job["location"], "Category": job["category"], "Fitting": job["fitting"],
         "Priority": job["priority"], "Deadline": london_time(job["deadline"]),
         "Overdue": "Overdue" if is_overdue(job, store.clock()) else "",
         "Labels": ", ".join(job["labels"]),
         "Escalated at": london_time(job["escalated_at"]) if job["escalated_at"] else None}
        for job in visible_jobs
    ]
    tint = {"Overdue": "rgba(200,60,50,.22)", "P1": "rgba(200,60,50,.22)", "P2": "rgba(220,150,40,.25)"}
    frame = pd.DataFrame(rows).style.map(
        lambda value: f"background-color: {tint[value]}" if value in tint else "",
        subset=["Priority", "Overdue"],
    )
    revision = st.session_state.get("jobs-editor-revision", 0)
    # The editor stores changes by row position. Reset them when filters change the visible jobs.
    editor_key = f"jobs-editor-{revision}-" + "-".join(str(job["id"]) for job in visible_jobs)
    statuses = {label: status for status, label in JOB_STATUSES.items()}
    previous_editor, previous_edits = st.session_state.get("jobs-editor-applied", (None, {}))

    def save_status_edits():
        edited_rows = st.session_state[editor_key]["edited_rows"]
        applied = previous_edits if previous_editor == editor_key else {}
        saved = False
        for row_index, changes in edited_rows.items():
            if changes == applied.get(row_index):
                continue
            if not 0 <= row_index < len(visible_jobs) or changes.get("Status") not in statuses:
                continue
            job = visible_jobs[row_index]
            status = statuses[changes["Status"]]
            if status == job["status"]:
                continue
            try:
                store.update_job_status(job, status)
                saved = True
            except ReviewError as exc:
                st.session_state["job-status-error"] = str(exc)
                st.session_state["jobs-editor-revision"] = revision + 1
        if saved:
            st.session_state["job-status-saved"] = "Job status saved."
        st.session_state["jobs-editor-applied"] = (editor_key, edited_rows)

    st.caption("Double-click a Status cell to change it.")
    st.data_editor(
        frame, hide_index=True, width="stretch", key=editor_key, on_change=save_status_edits,
        disabled=[column for column in frame.data.columns if column != "Status"],
        column_config={"Status": st.column_config.SelectboxColumn(options=list(statuses), required=True)},
    )
    st.caption(f"{len(jobs)} job{'s' if len(jobs) != 1 else ''}")


def show_policy():
    st.subheader("What happens when you report a problem")
    policies = retrieve_policy()
    if not policies:
        st.error("The configured policy pack is missing or invalid.")
        return
    featured = ("fire", "gas", "urgent", "access", "lighting", "review")
    st.caption(f"{len(policies)} active policies")
    ordered_topics = dict.fromkeys((*featured, *policies))
    for topic in ordered_topics:
        if topic not in policies:
            continue
        rule = policies[topic]
        with st.expander(POLICY_TITLES[topic], expanded=topic in featured):
            st.write(rule["summary"])
            if "target" in rule:
                st.caption(f"Repair target · {rule['target']['days']} days")
            if reply := rule.get("reply") or rule.get("lockout_reply"):
                st.caption("Resident guidance")
                st.write(reply)
            st.caption(f"{rule['id']} · Revision {rule['revision']}")


def show_evals():
    st.subheader("Quality checks")
    store.active_guidance(retrieve_policy())
    cases = store.eval_cases()
    if not cases:
        st.info("No test cases yet. To create one, open a report and choose Report actions → Save as quality check.")
    for case in cases:
        with st.expander(f"#{case['id']} · {markdown_text(case['name'])} · {case['status'].title()}"):
            st.text(case["snapshot"]["message"])
            st.text(case["notes"])
            st.json(case["expected"])
            st.caption(f"Captured {london_time(case['snapshot']['at'])} · {len(case['snapshot']['jobs'])} open jobs")
            choices = {"Use as test case": "holdout", "Use as example": "guidance",
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


def close_report_dialog():
    st.session_state.pop("report-dialog", None)


@st.dialog("Save as quality check", on_dismiss=close_report_dialog)
def label_report(report):
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
        expected["category"] = st.selectbox("Expected category", REPAIR_CATEGORIES)
        expected["priority"] = st.selectbox("Expected priority", ["P2", "P3"], index=None)
        expected["days"] = st.number_input("Expected target in days", min_value=1, max_value=90)
    notes = st.text_area("Reviewer notes", max_chars=1200, key=f"eval-notes-{report['id']}")
    with st.expander("Corrected interpretation (for guidance examples)"):
        corrected = st.text_area("Corrected interpretation JSON", key=f"eval-interpretation-{report['id']}",
                                 help="Enter the evidence, intent and facts that the agent should have extracted.")
    if st.button("Save draft", disabled=not kind or (kind == "create" and not expected["priority"])):
        try:
            interpretation = json.loads(corrected) if corrected.strip() else None
            store.save_eval_draft(report, expected, notes, interpretation=interpretation)
            close_report_dialog()
            st.toast("Draft saved. Open Quality checks to choose how to use it.")
            st.rerun()
        except (ReviewError, ValueError) as exc:
            st.error(str(exc))


def show_conversation(report, events):
    report_id = report["id"]
    st.markdown("#### Conversation")
    messages = [{"author": "resident", "body": report["message"], "at": report["received_at"]}]
    messages.extend(
        event["details"] | {"at": event["at"]} for event in reversed(events)
        if event["event"] in {"conversation_message", "reply_approved", "emergency_reply"}
    )
    for index, message in enumerate(messages):
        resident = message["author"] == "resident"
        author = "Resident" if resident else "Staff"
        side = "left" if resident else "right"
        avatar = ":material/person:" if resident else ":material/support_agent:"
        with st.container(horizontal=True, horizontal_alignment=side, key=f"ticket-bubble-{side}-{index}"):
            with st.chat_message(author, avatar=avatar, width="content"):
                st.text(message["body"])
                when = london_time(message["at"]) if message["at"] else "Time unavailable"
                st.caption(f"{author} · {when}")

    with st.expander("Add a message"):
        with st.form(f"ticket-message-{report_id}", clear_on_submit=True, border=False):
            author = st.radio("From", ["staff", "resident"], horizontal=True,
                              format_func=lambda role: "Staff reply" if role == "staff" else "Resident follow-up")
            body = st.text_area("Message", max_chars=4000)
            if st.form_submit_button("Add to conversation"):
                try:
                    store.add_ticket_message(report_id, author, body)
                    st.rerun()
                except ReviewError as exc:
                    st.error(str(exc))

    proposal = report["proposal"] or {}
    draft = proposal.get("reply") or proposal.get("question")
    if not draft or (report["result"] or {}).get("decision") == "reject":
        return
    version = report["version"]
    if any(event["event"] == "emergency_reply" for event in events):
        return
    if any(event["event"] in {"reply_approved", "reply_declined"}
           and event["details"]["version"] == version for event in events):
        return
    with st.expander("Proposed reply to the resident", expanded=True):
        st.caption("Review or edit the draft below. Posting adds it to the ticket conversation "
                   "as a staff reply. The handoff is reviewed separately.")
        with st.form(f"ticket-draft-{report_id}-{version}", border=False):
            reply = st.text_area("Resident reply draft", value=draft, max_chars=4000)
            with st.container(horizontal=True):
                approve = st.form_submit_button("Approve and post reply", type="primary")
                decline = st.form_submit_button("Decline draft", help="Records the decision without posting a reply.")
            if approve or decline:
                try:
                    review_reply(store, report_id, version, reply, decline=decline)
                    st.rerun()
                except ReviewError as exc:
                    st.error(str(exc))


def show_activity(report, events):
    events = events.copy()
    if report["result"] and not any(e["event"] == "review_complete" for e in events):
        events.insert(0, {"event": "review_complete", "details": report["result"], "at": report["result"].get("at")})
    if report["proposal"] and not any(e["event"].startswith("proposal_") for e in events):
        proposal = report["proposal"]
        events.append({"event": "proposal_ready", "details": {"kind": proposal["kind"]}, "at": None})
    if report["triage_error"] and not any(e["event"] == "triage_failed" for e in events):
        events.append({"event": "triage_failed", "details": {}, "at": None})
    events.append({"event": "received", "details": {}, "at": report["received_at"]})

    for event in events:
        details = event["details"]
        name = event["event"]
        if name == "review_complete":
            title = OUTCOMES[details["action"]]
        elif name in {"proposal_ready", "proposal_updated"}:
            label = HANDOFF_LABELS[details["kind"]]
            stage = "Agent proposed" if name == "proposal_ready" else "Proposal updated"
            title = f"{stage} · {label}"
        else:
            title = {
                "received": "Report received", "triage_started": "Agent started reviewing",
                "triage_failed": "Agent could not complete the review", "requeued": "Returned to the queue",
                "reply_approved": "Proposed reply approved and posted",
                "emergency_reply": "Emergency guidance posted automatically",
                "reply_declined": "Proposed reply declined",
                "conversation_message": "Resident follow-up recorded" if details.get("author") == "resident" else "Staff reply recorded",
            }[name]
        notes = []
        if (actor := event.get("reviewer_id")) and actor not in {"system", "resident"}:
            notes.append(f"Reviewer: {actor}")
        if job_id := details.get("job_id"):
            notes.append(f"Job #{job_id}")
        if destination := details.get("destination"):
            notes.append(destination)
        if deadline := details.get("deadline"):
            notes.append(f"Due {london_time(deadline)}")
        edits = details.get("reviewer_edits") or {}
        if due_date := edits.get("due_date"):
            if not details.get("deadline"):
                notes.append(f"Reviewer deadline {date.fromisoformat(due_date):%d %b %Y}")
        elif days := details.get("days"):
            notes.append(f"{days}-day repair target")
        if "labels" in edits:
            notes.append("Reviewer tags: " + (", ".join(tag.replace("_", " ") for tag in edits["labels"]) or "none"))
        if child_ids := details.get("report_ids"):
            notes.append("Reports " + ", ".join(f"#{child}" for child in child_ids))
        if note := details.get("note"):
            notes.append(note)
        timestamp = details.get("at", event["at"]) if name == "review_complete" else event["at"]
        when = london_time(timestamp) if timestamp else "Time unavailable"
        with st.container(gap=None):
            st.markdown(f"**{markdown_text(title)}**")
            st.caption(markdown_text(" · ".join([when, *notes])))


@st.dialog("Agent diagnostics", on_dismiss=close_report_dialog)
def show_diagnostics(report):
    if retrieval := (report["proposal"] or {}).get("retrieval"):
        policies = ", ".join(f"{rule['id']} v{rule['revision']}" for rule in retrieval["policies"])
        examples = ", ".join(f"#{case_id}" for case_id in retrieval["example_ids"])
        st.write("These are the policy rules and approved examples the agent used during triage.")
        st.text(policies or "No valid policy retrieved.")
        st.text(f"Approved examples: {examples or 'none matched'}")
    if report["trace_id"]:
        st.caption("LangSmith trace ID")
        st.code(report["trace_id"], language=None)


@st.dialog("Delete report", on_dismiss=close_report_dialog)
def confirm_delete(report):
    st.write(f"Remove report #{report['id']} from UniHelp? Any linked repair jobs and split reports will remain available.")
    if st.button("Delete report", type="primary", icon=":material/delete:", key="confirm-delete"):
        try:
            store.delete_report(report["id"], report["version"])
            close_report_dialog()
            st.rerun()
        except ReviewError as exc:
            st.error(str(exc))


if st.session_state.pop("return_to_inbox", False):
    st.session_state.mailboxes = "Inbox"

if (reset_count := st.session_state.pop("reset-queue-count", None)) is not None:
    st.session_state.workspace = "Reports"
    st.session_state.mailboxes = "Queued"
    for key in ("report-search", "report-date-from", "report-date-to"):
        st.session_state.pop(key, None)
    st.toast(f"Queued {reset_count} report{'s' if reset_count != 1 else ''} for triage.")

if opened := st.session_state.pop("open_report", None):
    st.session_state.workspace = "Reports"
    st.session_state.mailboxes = "Inbox"
    for key in ("report-search", "report-date-from", "report-date-to"):
        st.session_state.pop(key, None)
    st.session_state["message-list"] = opened

all_reports = store.reports()
workspaces = ["Reports", "Jobs", "Policy"]
if quality_checks_enabled:
    workspaces.append("Quality checks")
if st.session_state.get("workspace") not in workspaces:
    st.session_state.workspace = "Reports"
counts = {
    "Inbox": len(all_reports),
    "Queued": sum(status(r) in QUEUE_STATUSES for r in all_reports),
    "Needs review": sum(bool(r["proposal"]) and not r["result"] for r in all_reports),
    "Handled": sum(bool(r["result"]) for r in all_reports),
    "Failed": sum(status(r) == "Triage failed" for r in all_reports),
}
with st.container(key="mail-header"):
    brand, navigation, actions = st.columns([1.25, 3, 1.25], vertical_alignment="center")
    with brand:
        st.title("UniHelp", anchor=False)
    with navigation, st.container(horizontal=True, horizontal_alignment="center", key="workspace-nav"):
        workspace = st.segmented_control("Workspace", workspaces,
                                         key="workspace", required=True,
                                         label_visibility="collapsed") or "Reports"
    with actions, st.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center",
                               gap="xsmall", key="header-actions"):
        if st.button("New report", icon=":material/edit_square:", type="primary"):
            st.session_state.composing = True
        with st.popover("", icon=":material/more_horiz:", help="Report queue actions"):
            st.write("Queue all reports again, including handled reports. This clears current proposals "
                     "and decisions. Conversations, activity history, and existing jobs are kept.")
            if st.button("Reset and requeue all reports", disabled=not all_reports):
                st.session_state["reset-queue-count"] = store.reset_report_queue()
                st.rerun()
    if st.session_state.get("composing"):
        compose()

if workspace == "Jobs":
    show_jobs()
elif workspace == "Policy":
    show_policy()
elif workspace == "Quality checks":
    show_evals()
else:
    queue, detail = st.columns([1, 1.75], gap="medium")
    with queue, st.container(key="inbox-list", border=True, height=560):
        with st.container(horizontal=True, vertical_alignment="center", key="report-search-bar"):
            query = st.text_input("Search reports", placeholder="Search reports", icon=":material/search:",
                                  label_visibility="collapsed", key="report-search").strip().casefold()
            dates = date_filter("Received dates", "report-date")
        st.session_state.setdefault("mailboxes", "Inbox")
        folder = st.pills("Report filters", FOLDERS, key="mailboxes", required=True,
                          format_func=lambda name: f"{name} · {counts[name]}",
                          label_visibility="collapsed") or "Inbox"
        if (retried := st.session_state.pop("retried-failed-count", None)) is not None:
            st.toast(f"Queued {retried} report{'s' if retried != 1 else ''} for retry."
                     if retried else "No failed reports to retry.")
        if folder == "Failed" and counts["Failed"]:
            if st.button("Retry all failed", icon=":material/refresh:", disabled=bool(configuration_issue),
                         help=configuration_issue or "Queues all failed reports, including those hidden by filters."):
                st.session_state["retried-failed-count"] = store.retry_triage()
                st.rerun()
        reports = {
            r["id"]: r for r in all_reports
            if (folder == "Inbox" or folder == "Handled" and r["result"]
                or folder == "Queued" and status(r) in QUEUE_STATUSES
                or folder == "Failed" and status(r) == "Triage failed"
                or folder == "Needs review" and r["proposal"] and not r["result"])
            and query in f"{r['id']} {r['location']} {r['subject']} {r['message']}".casefold()
            and matches_date(r["received_at"], dates)
        }

        review_now = store.clock()
        urgent_review_after = timedelta(minutes=15)

        def awaiting_urgent_review(report):
            return (status(report) in {"Emergency", "Urgent"} and report["received_at"]
                    and review_now - datetime.fromisoformat(report["received_at"]) >= urgent_review_after)

        def mail_row(report_id):
            report = reports[report_id]
            subject = markdown_text(report["subject"] or "Resident report")
            message = " ".join(report["message"].split())
            preview = markdown_text(message[:76].rstrip() + "..." if len(message) > 76 else message)
            state = status(report)
            color = BADGE_COLORS.get(state, "gray")
            waiting = " · Awaiting review over 15 min" if awaiting_urgent_review(report) else ""
            return f"**{subject}** `#{report_id}`  \n*{preview}*  \n:{color}-badge[{state}]{waiting}"

        rank = {"Emergency": 0, "Urgent": 1}
        reports = dict(sorted(reports.items(), key=lambda row: (
            rank.get(status(row[1]), 2),
            (datetime.fromisoformat(row[1]["received_at"]).timestamp() if row[1]["received_at"] else float("inf"))
            if folder == "Needs review" or status(row[1]) in rank else 0,
            row[0],
        )))
        if st.session_state.get("message-list") not in reports:
            st.session_state["message-list"] = next(iter(reports), None)
        with st.container(key="report-rows", height=420, border=False):
            selected = st.radio("Reports", reports, index=None, key="message-list",
                                label_visibility="collapsed", width="stretch",
                                format_func=mail_row) if reports else None
            if not reports:
                with st.container(key="report-empty"):
                    st.caption("No matching reports.")
    with detail, st.container(key="reading-pane", border=True, height=560):
        if selected is None:
            st.info("Select a report to see the message and handoff.")
        else:
            report = reports[selected]
            with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
                st.badge(status(report), color=BADGE_COLORS.get(status(report), "gray"))
                with st.popover("Report actions", icon=":material/more_horiz:"):
                    if quality_checks_enabled and st.button("Save as quality check"):
                        st.session_state["report-dialog"] = ("quality", selected)
                    if (report["trace_id"] or (report["proposal"] or {}).get("retrieval")) and st.button("Agent diagnostics"):
                        st.session_state["report-dialog"] = ("diagnostics", selected)
                    if st.button("Delete report", icon=":material/delete:"):
                        st.session_state["report-dialog"] = ("delete", selected)
            if (dialog := st.session_state.get("report-dialog")) and dialog[1] == selected:
                if not st.session_state.get("composing"):
                    {"quality": label_report, "diagnostics": show_diagnostics, "delete": confirm_delete}[dialog[0]](report)
            st.header(markdown_text(report["subject"] or "Resident report"))
            if report["triage_error"]:
                st.error(configuration_issue or report["triage_error"])
                if st.button("Retry triage", icon=":material/refresh:", disabled=bool(configuration_issue)):
                    store.retry_triage(selected)
                    st.session_state["return_to_inbox"] = True
                    st.rerun()
            received = london_time(report["received_at"]) if report["received_at"] else ""
            st.caption(f"#{selected} · {markdown_text(report['location'])}" + (f" · {received}" if received else ""))
            if report["parent_report_id"]:
                st.caption(f"Split from report #{report['parent_report_id']}")
            if repeat_id := (report["proposal"] or {}).get("repeat_of_report_id"):
                st.info(f"The same message was submitted at this location in report #{repeat_id}.")
            events = store.activity(selected)
            show_conversation(report, events)
            st.divider()
            if report["result"]:
                receipt = report["result"]
                st.success(OUTCOMES[receipt["action"]])
                if receipt.get("job_id"):
                    st.write(f"Job #{receipt['job_id']}" + (f" · {receipt['destination']}" if receipt.get("destination") else ""))
                if child_ids := receipt.get("report_ids"):
                    st.text("Created reports " + ", ".join(f"#{child}" for child in child_ids))
            elif report["proposal"]:
                show_review(report, all_reports)
            elif not report["triage_error"]:
                if status(report) == "Triaging":
                    st.info("UniHelp is reviewing this report.")
                else:
                    st.info(configuration_issue or (
                        "This report is queued for the background worker." if background_worker
                        else "UniHelp will prepare a handoff automatically."
                    ))
            with st.expander("Activity history"):
                with st.container(height=280, border=False):
                    show_activity(report, events)


@st.cache_resource
def triage_executor():
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="unihelp-triage")


@st.fragment(run_every=3)
def automatic_triage():
    if st.session_state.get("composing") or st.session_state.get("report-dialog"):
        return
    pending = st.session_state.get("triage-future")
    if pending is not None and pending.done():
        st.session_state.pop("triage-future", None)
        try:
            pending.result()
        except Exception:
            logging.exception("Could not process the triage queue")
        pending = None
    if pending is None and not configuration_issue and not background_worker:
        now = store.clock().isoformat()
        if any(status(report) == "Queued" or (
            status(report) == "Triaging" and report["claim_until"] and report["claim_until"] <= now
        ) for report in all_reports):
            st.session_state["triage-future"] = triage_executor().submit(process_pending, store)
    if store.reports() != all_reports:
        st.rerun()


automatic_triage()
