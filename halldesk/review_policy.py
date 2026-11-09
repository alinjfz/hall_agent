"""Apply a staff clarification or edit without writing any jobs."""
from datetime import date

from halldesk.context import retrieve_policy, same_policy, search_rules
from halldesk.policy import chase_proposal, manual, repair_proposal, urgent_proposal
from halldesk.proposals import Proposal
from halldesk.store import LABELS, LONDON, ReviewError


def revise_proposal(store, report, proposal: Proposal, value):
    kind = proposal.kind
    if value is None and kind in {"job_question", "lighting_question", "plumbing_question", "access_question", "power_question"}:
        updated = manual(f"The residence duty reviewer could not confirm: {proposal.question}")
    elif kind == "job_question" and type(value) is int:
        offered = next((job.model_dump(exclude_unset=True) for job in proposal.candidates if job.id == value), None)
        if offered is None or offered not in store.jobs(report["location"]):
            raise ReviewError("The selected job changed. Triage the report again.")
        policies = retrieve_policy()
        updated = chase_proposal(offered, policies.get("escalation"), proposal.evidence,
                                 store.clock(), policies.get("follow_up"))
    elif kind == "lighting_question" and type(value) is bool:
        rule = search_rules("lighting")
        if not same_policy(rule, proposal.rule.model_dump(exclude_unset=True)):
            raise ReviewError("Policy changed. Triage the report again.")
        updated = repair_proposal(report, proposal.fitting, value, rule, confirmed_at=store.clock())
    elif kind == "plumbing_question" and type(value) is bool:
        rule = search_rules("plumbing")
        if not same_policy(rule, proposal.rule.model_dump(exclude_unset=True)):
            raise ReviewError("Policy changed. Triage the report again.")
        updated = (repair_proposal(report, proposal.fitting, None, rule, "plumbing") if value
                   else urgent_proposal(report, retrieve_policy(), store.jobs(report["location"]), {"building_hazard"}))
    elif kind in {"access_question", "power_question"} and type(value) is bool:
        policies = retrieve_policy()
        if not same_policy(policies.get(proposal.rule.topic), proposal.rule.model_dump(exclude_unset=True)):
            raise ReviewError("Policy changed. Triage the report again.")
        if value:
            updated = urgent_proposal(report, policies, store.jobs(report["location"]),
                                      {"lockout" if kind == "access_question" else "building_hazard"})
        else:
            updated = manual("Refer to residence staff for routine access assistance." if kind == "access_question"
                             else "Refer the local electrical fault for ordinary maintenance review.")
            updated["rule"] = proposal.rule.model_dump(exclude_unset=True)
    elif kind == "create":
        if isinstance(value, list):
            value = {"labels": value}
        if not isinstance(value, dict) or not isinstance(value.get("labels"), list):
            raise ReviewError("Choose valid job tags and a deadline.")
        labels = value["labels"]
        if any(label not in LABELS for label in labels):
            raise ReviewError("Choose valid job tags.")
        edits = proposal.reviewer_edits.model_dump(exclude_unset=True) if proposal.reviewer_edits else {}
        if "due_date" in value:
            due_date = value["due_date"]
            if due_date is None:
                edits.pop("due_date", None)
            elif type(due_date) is date and due_date >= store.clock().astimezone(LONDON).date():
                edits["due_date"] = due_date.isoformat()
            else:
                raise ReviewError("Choose today or a future date.")
        labels = list(dict.fromkeys(labels))
        if labels != proposal.labels:
            edits["labels"] = labels
        updated = proposal.model_dump(exclude_unset=True) | {"labels": labels, "reviewer_edits": edits}
    else:
        raise ReviewError("Invalid review change.")
    previous = proposal.model_dump(exclude_unset=True)
    for key in ("retrieval", "context_snapshot", "repeat_of_report_id", "review_thread_id"):
        if key in previous:
            updated[key] = previous[key]
    if updated["kind"] == "manual":
        if rule := search_rules("review"):
            updated.setdefault("rule", rule)
            updated.setdefault("destination", rule.get("destination", "Residence duty reviewer"))
        updated.setdefault("reply", "Your report needs residence staff review.")
    return updated
