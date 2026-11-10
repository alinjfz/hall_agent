"""Independent checks shared by the local report and LangSmith columns."""


def route(outputs, reference_outputs):
    return outputs.get("route") == reference_outputs["route"] if "route" in reference_outputs else None


def urgency(outputs, reference_outputs):
    return outputs.get("urgency") == reference_outputs["urgency"] if "urgency" in reference_outputs else None


def job(outputs, reference_outputs):
    keys = {"job", "candidates", "related_jobs"} & reference_outputs.keys()
    return all(outputs.get(key) == reference_outputs[key] for key in keys) if keys else None


def category(outputs, reference_outputs):
    return outputs.get("category") == reference_outputs["category"] if "category" in reference_outputs else None


def priority_deadline(outputs, reference_outputs):
    if "days" in reference_outputs:
        return ((outputs.get("priority"), outputs.get("days")) ==
                (reference_outputs["priority"], reference_outputs["days"])
                and ("deadline" not in reference_outputs or outputs.get("deadline") == reference_outputs["deadline"]))
    return None


def issues(outputs, reference_outputs):
    return outputs["issues"] == reference_outputs["issues"] if "issues" in reference_outputs else None


def no_write_before_approval(outputs):
    return outputs["no_write_before_approval"]


def evidence_is_verbatim(inputs, outputs):
    """Wording must match the report; case, spacing and typographic quotes may differ."""
    from unihelp.evidence import quoted_in
    return all(quoted_in(excerpt, inputs["message"]) for excerpt in outputs["evidence"])


def model_completed(outputs):
    return not outputs["trajectory"]["failed_interpretation"]


def tools_called(outputs):
    return {"key": "tools_called", "value": {"ordered": [
        {"name": event["name"], "origin": event["origin"]} for event in outputs["trajectory"]["tools"]
    ]}}


def tool_order(outputs):
    events = outputs["trajectory"]["steps"]
    interpretation = next((i for i, name in enumerate(events)
                           if name in {"interpretation_pass", "interpret", "assess_report"}), None)
    if interpretation is None or "apply_policy" not in events:
        return None
    policy = events.index("apply_policy")
    if "retrieve_handbook" in events:
        # Pipeline graph: retrieval follows interpretation and precedes policy. Only the
        # reply drafter may search again, and only after policy has decided the route.
        return interpretation < events.index("retrieve_handbook") < policy and all(
            index > policy for index, name in enumerate(events) if name == "search_policy")
    return interpretation < policy and all(
        index < policy for index, name in enumerate(events) if name in {"get_open_jobs", "search_rules", "search_policy"})


def no_off_topic_lookup(outputs):
    tools = [event for event in outputs["trajectory"]["tools"]
             if event["origin"] in {"model", "declared"} and event["on_topic"] is not None]
    return all(event["on_topic"] for event in tools) if tools else None


def model_calls(outputs):
    return {"key": "model_calls", "score": outputs["trajectory"]["model_calls"]}


def nominated_job(outputs, reference_outputs):
    if "nominated_job_ids" not in outputs:
        return None  # The legacy graph did not expose its pre-guard nominations.
    ids = outputs.get("nominated_job_ids")
    if "candidates" in reference_outputs:
        expected = reference_outputs["candidates"]
    elif reference_outputs.get("job") is not None:
        expected = [reference_outputs["job"]]
    elif "job" in reference_outputs and ids is not None:
        expected = []
    else:
        return None
    return ids is not None and set(ids) == set(expected)


def intake_urgency(outputs, reference_outputs):
    """Measure what the deterministic gate caught without the live model."""
    return outputs["intake_urgency"] == reference_outputs["urgency"] if "urgency" in reference_outputs else None


def retrieval_skipped(outputs):
    """Urgent and uninterpreted reports go to staff without a handbook search.

    Scoring those trials as zero recall would measure routing, not retrieval.
    The original agent looked up policy through tools, so its trials stay scored.
    """
    trajectory = outputs.get("trajectory", {})
    return "search_handbook" not in trajectory.get("steps", ["search_handbook"]) and not trajectory.get("tools")


def retrieval_recall_at_3(outputs):
    expected = set(outputs.get("expected_sections", []))
    if expected and not retrieval_skipped(outputs):
        return len(set(outputs.get("section_ids", [])[:3]) & expected) / len(expected)
    return None


def retrieval_hit_at_1(outputs):
    expected = set(outputs.get("expected_sections", []))
    if expected and not retrieval_skipped(outputs):
        return bool(set(outputs.get("section_ids", [])[:1]) & expected)
    return None


def reply_grounded(outputs, reference_outputs):
    if not outputs.get("reply"):
        return False if reference_outputs.get("reply_grounded") else None
    if not outputs.get("reply_sources"):
        if not reference_outputs.get("reply_grounded"):
            return None
        from unihelp.replies import grounded_reply_facts
        return grounded_reply_facts(outputs["reply"], outputs.get("job_record"))
    from unihelp.replies import grounded_reply
    return grounded_reply(outputs["reply"], outputs["reply_sources"], outputs["section_ids"], outputs.get("job_record"))


CORE = (route, urgency, job, category, priority_deadline, issues, no_write_before_approval, evidence_is_verbatim, model_completed)
TRAJECTORY = (tools_called, tool_order, no_off_topic_lookup, model_calls, nominated_job)
