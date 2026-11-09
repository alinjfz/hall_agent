"""Deterministic triage decisions; no model or job writes."""

import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from langsmith import traceable

from halldesk.safety import safety_signals
from halldesk.store import LABELS, is_overdue


def manual(reason):
    return {"kind": "manual", "reason": reason}


def urgent_proposal(report, policies, jobs, signals):
    if not signals:
        return None
    replies = []
    emergency = bool(signals & {"fire", "gas"})
    topic = "access" if signals <= {"lockout", "identifiable_key", "unsafe_door"} else "urgent"
    for hazard, fallback in (
        ("fire", "For fire or smoke, call 999 immediately and follow the emergency operator’s directions."),
        ("gas", "Call the National Gas Emergency Service immediately on 0800 111 999 from a safe place."),
    ):
        if hazard in signals:
            if topic not in {"fire", "gas"}:
                topic = hazard
            replies.append(policies.get(hazard, {}).get("reply", fallback))
    if "lockout" in signals:
        replies.append(policies.get("access", {}).get("lockout_reply",
                       "Contact on-duty residence staff now for identity and access checks."))
    if not emergency and signals != {"lockout"}:
        replies.append("Thank you for reporting this issue. It requires urgent assessment by the residence team. "
                       "Please contact the on-duty residence staff immediately so they can assess the situation.")
        if "fire_equipment" in signals and topic == "urgent":
            topic = "fire"
    proposal = manual("Emergency response needed now." if emergency else "Urgent staff handling needed now.")
    destination = policies.get(topic, {}).get("destination") if emergency or topic != "fire" else None
    proposal.update(urgency="emergency" if emergency else "urgent", reply="\n\n".join(replies),
                    signals=sorted(signals), destination=destination or "On-duty residence staff")
    if rule := policies.get(topic):
        proposal["rule"] = rule
    message = report["message"]
    mentioned_ids = {
        int(job_id) for job_id in re.findall(r"\b(?:job|repair)\s*#?\s*(\d{1,12})\b", message, re.I)
    }
    matching = [job for job in jobs or [] if job["id"] in mentioned_ids or re.search(
        rf"(?<!\w){re.escape(job['fitting'])}(?!\w)", message, re.I)]
    if matching:
        proposal["related_jobs"] = matching
        proposal["evidence"] = message
    # Keep the urgent path independent of the model. Staff can choose which
    # excerpts are separate issues when they acknowledge the urgent handoff.
    excerpts = re.split(r"(?<=[.!?])\s+|\n+|(?<=,)\s+(?:and|but)\s+", message, flags=re.I)
    proposal["followup_suggestions"] = [
        excerpt.strip() for excerpt in excerpts
        if excerpt.strip() and excerpt.strip() != message.strip() and not safety_signals(excerpt)
    ][:4]
    return proposal


def chase_proposal(job, rule, evidence, now, follow_up=None):
    if not rule or rule.get("boundary") != "strictly_after" or not rule.get("destination"):
        return manual("The escalation policy could not be verified.")
    if job["escalated_at"]:
        return manual("This job already has a recorded escalation.")
    overdue = is_overdue(job, now)
    deadline = datetime.fromisoformat(job["deadline"]).astimezone(ZoneInfo("Europe/London"))
    reply = f"Repair #{job['id']} for the {job['fitting']} is still open. Its recorded target is {deadline:%d %b %Y, %H:%M %Z}."
    if overdue:
        reply += " That target has passed."
    proposal = {"kind": "escalate" if overdue else "draft", "rule": rule, "job": job,
                "evidence": evidence, "reply": reply}
    if follow_up:
        proposal["reply"] += "\n\n" + follow_up["summary"]
        proposal["reply_source"] = follow_up
    return proposal


def repair_proposal(report, fitting, other_works, rule, category="lighting", *, confirmed_at=None):
    band = "target"
    if category == "lighting":
        band = "other_lighting_works" if other_works else "no_other_lighting"
    if not rule:
        return manual("The repair policy could not be verified.")
    terms = rule[band]
    labels = [label for label in LABELS if label.replace("_", " ") in fitting.casefold()]
    if report["location"].endswith(("kitchen", "landing")):
        labels.insert(0, "communal_area")
    proposal = {"kind": "create", "rule": rule, "fitting": fitting, "labels": labels, "category": category,
                "priority": terms["priority"], "days": terms["days"]}
    if terms.get("clock") == "confirmed" and confirmed_at is not None:
        target = confirmed_at.astimezone(timezone.utc) + timedelta(days=terms["days"])
        proposal["policy_deadline"] = target.isoformat()
    return proposal


@traceable(name="apply_policy")
def apply_policy(report, interpretation, jobs, policies, now, job_reports=None):
    message = report["message"]
    evidence = interpretation.evidence
    if proposal := urgent_proposal(report, policies, jobs, safety_signals(message)):
        return proposal
    if interpretation.safety_issue:
        if interpretation.safety_evidence and interpretation.safety_evidence in message:
            return urgent_proposal(report, policies, jobs, {interpretation.safety_issue})
        return manual("The residence duty reviewer needs to verify a possible safety issue.")
    if evidence and (not evidence.strip() or evidence not in message):
        return manual("The quoted evidence was not found in the report.")
    if interpretation.intent == "multiple":
        return handle_multiple(report, interpretation, policies)
    if (interpretation.intent == "access" or interpretation.category in {"lockout", "spare_key"}
            or re.search(r"\b(?:spare|replacement|extra) key\b", message, re.I)):
        return handle_access(report, interpretation, jobs, policies)
    if interpretation.intent == "manual":
        return manual("The residence duty reviewer needs to assess the reported issue.")
    if jobs is None or not evidence.strip():
        return manual("The interpretation lacks a verified lookup or evidence from the report.")
    if interpretation.intent == "new":
        return handle_repair(report, interpretation, jobs, policies, now, job_reports)
    return handle_chase(interpretation, jobs, policies, now)


def handle_multiple(report, interpretation, policies):
    message = report["message"]
    issues = [issue.strip() for issue in interpretation.issues or []]
    rule = policies.get("multiple")
    if (report.get("parent_report_id") or not rule or not 2 <= len(issues) <= 4
            or len({issue.casefold() for issue in issues}) != len(issues)):
        return manual("Review the individual issues and their locations manually.")
    end = 0
    for issue in issues:
        start = message.find(issue, end)
        if not issue or start < 0:
            return manual("The individual issues could not be verified against the report.")
        end = start + len(issue)
    return {"kind": "split", "issues": issues, "rule": rule}


def access_restored(message):
    return bool(re.search(r"\b(?:back inside|back in now|inside now|can (?:still )?get in)\b", message, re.I))


def handle_access(report, interpretation, jobs, policies):
    message, evidence = report["message"], interpretation.evidence
    has_access = access_restored(message)
    if interpretation.category == "lockout" and evidence.strip() and evidence in message:
        if not has_access:
            return urgent_proposal(report, policies, jobs, {"lockout"})
    rule = policies.get("access")
    if rule and interpretation.category == "lockout" and not has_access:
        return {"kind": "access_question", "question": "Can the resident get into their room or flat now?", "rule": rule}
    proposal = manual("Refer this access request to residence staff for identity and access checks.")
    if rule:
        proposal["rule"] = rule
    return proposal


def handle_repair(report, interpretation, jobs, policies, now, job_reports):
    message, evidence = report["message"], interpretation.evidence
    category = interpretation.category
    if category is None:
        return manual("The repair category could not be verified.")
    rule = policies.get(category)
    fitting = (interpretation.fitting or "").strip()
    if not fitting or fitting.casefold() not in evidence.casefold():
        return manual("The affected fitting could not be verified.")
    duplicate_jobs = [job["id"] for job in jobs if job["fitting"].casefold() == fitting.casefold()]
    if duplicate_jobs:
        proposal = manual("An open job already covers this fitting.") | {
            "duplicate_job_ids": duplicate_jobs, "related_jobs": [job for job in jobs if job["id"] in duplicate_jobs],
            "evidence": message,
            "reply": "There is already an open repair for this fitting. Residence staff can review your report alongside it.",
        }
        if duplicate_rule := policies.get("duplicate"):
            proposal["rule"] = duplicate_rule
        return proposal
    possible_duplicates = [
        {"job_id": job["id"], "fitting": job["fitting"], **job_reports[job["id"]]}
        for job in jobs
        if job_reports and job["id"] in job_reports and job["category"] == category
        and job["id"] in interpretation.candidate_ids
    ]
    if possible_duplicates:
        proposal = manual("An open job may cover this fault. Compare the original reports before creating another job.")
        proposal["possible_duplicates"] = possible_duplicates
        if duplicate_rule := policies.get("duplicate"):
            proposal["rule"] = duplicate_rule
        return proposal
    if category == "plumbing":
        if not re.search(r"\b(?:tap|shower)\b", fitting, re.I):
            return manual("Confirm the source and severity of the leak before proposing a repair.")
        if not re.search(r"\bdrip(?:s|ping)?\b", evidence, re.I):
            if rule and (question := rule.get("clarification_question")):
                return {"kind": "plumbing_question", "fitting": fitting, "rule": rule, "question": question}
            return manual("Confirm the severity of the leak before proposing a repair.")
    if category != "lighting":
        return repair_proposal(report, fitting, None, rule, category)
    if not rule:
        return manual("The lighting priority policy could not be verified.")
    lighting_evidence = interpretation.lighting_evidence or ""
    portable = re.search(r"\b(?:torch|flashlight|(?:desk|floor|table|portable) lamp|phone light)\b", lighting_evidence, re.I)
    if interpretation.other_lighting_works is None or (interpretation.other_lighting_works and portable):
        return {"kind": "lighting_question", "fitting": fitting, "rule": rule,
                "question": rule.get("clarification_question", "Does another light still work in the same room?")}
    if not interpretation.lighting_evidence or interpretation.lighting_evidence not in message:
        return manual("The other lighting in the room could not be verified.")
    confirmed_at = datetime.fromisoformat(report["received_at"]) if report.get("received_at") else now
    return repair_proposal(report, fitting, interpretation.other_lighting_works, rule, confirmed_at=confirmed_at)


def identifies_fitting(evidence, fitting):
    for match in re.finditer(rf"(?<!\w){re.escape(fitting)}(?!\w)", evidence, re.I):
        prefix = re.split(r"[.!?;,]", evidence[:match.start()])[-1]
        if not re.search(r"\b(?:not|rather than|instead of)\b[^.!?;,]{0,32}$", prefix, re.I):
            return True
    return False


def handle_chase(interpretation, jobs, policies, now):
    evidence = interpretation.evidence
    rule = policies.get("escalation")
    candidates = [job for job in jobs if job["id"] in interpretation.candidate_ids]
    if not rule:
        return manual("The escalation policy could not be verified.")
    if len(candidates) > 1:
        return {"kind": "job_question", "candidates": candidates, "rule": rule, "evidence": evidence,
                "question": "Which of these fittings is the resident following up on?"}
    if len(candidates) != 1:
        return manual("A matching open job could not be established.") | {
            "reply": "We couldn't verify a matching open repair. If you have a job reference, please share it so residence staff can check.",
        }
    job = candidates[0]
    if not identifies_fitting(evidence, job["fitting"]):
        return manual("The report evidence does not identify the selected fitting.")
    return chase_proposal(job, rule, evidence, now, policies.get("follow_up"))


def context_issue(context):
    if not context["policies"]:
        return "The policy pack is missing or invalid. Ask the policy owner to restore it."
    if len(context["jobs"]) > 25:
        return "There are too many open jobs here to verify a match automatically."
    return None


def build_proposal(report, context, interpretation):
    policies = context["policies"]
    proposal = urgent_proposal(report, policies, context["jobs"], safety_signals(report["message"]))
    if proposal is None:
        if issue := context_issue(context):
            proposal = manual(issue)
        elif interpretation is None:
            proposal = manual("The residence duty reviewer needs to interpret this report.")
        else:
            proposal = apply_policy(report, interpretation, context["jobs"], policies,
                                    datetime.fromisoformat(context["at"]), context.get("job_reports"))
    if proposal["kind"] == "manual" and not proposal.get("urgency") and policies.get("urgent"):
        if (not context_issue(context)
                and (interpretation is None or not interpretation.evidence
                     or interpretation.evidence.strip() and interpretation.evidence in report["message"])
                and proposal.get("rule", {}).get("topic") not in {"access", "duplicate"}
                and re.search(r"\b(?:power (?:is )?out|lost power|power cut|no power)\b", report["message"], re.I)):
            proposal = {"kind": "power_question", "rule": policies["urgent"],
                        "question": "Has the whole flat or affected area lost power, or just one fitting?"}
    if proposal["kind"] == "manual" and policies:
        proposal.setdefault("rule", policies["review"])
        proposal.setdefault("destination", proposal["rule"].get("destination", "Residence duty reviewer"))
        if not proposal.get("reply"):
            proposal["reply"] = "Your report needs residence staff review."
    if interpretation is not None and interpretation.missing_facts:
        proposal["missing_facts"] = interpretation.missing_facts
    proposal["retrieval"] = {
        "policies": [{"id": rule["id"], "revision": rule["revision"]} for rule in policies.values()],
        "example_ids": [example["id"] for example in context["examples"]],
    }
    proposal["context_snapshot"] = {"at": context["at"], "jobs": context["jobs"],
                                    "job_reports": context.get("job_reports", {}),
                                    "message": report["message"], "location": report["location"]}
    if repeat_id := context.get("repeat_of_report_id"):
        proposal["repeat_of_report_id"] = repeat_id
    return proposal


