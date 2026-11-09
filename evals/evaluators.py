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
    return all(excerpt.strip() and excerpt in inputs["message"] for excerpt in outputs["evidence"])


def model_completed(outputs):
    return not outputs["trajectory"]["failed_interpretation"]


def tools_called(outputs):
    return {"key": "tools_called", "value": {"ordered": [event["name"] for event in outputs["trajectory"]["tools"]]}}


def tool_order(outputs):
    events = outputs["trajectory"]["steps"]
    interpretation = next((i for i, name in enumerate(events) if name in {"interpretation_pass", "interpret"}), None)
    if interpretation is None or "apply_policy" not in events:
        return None
    policy = events.index("apply_policy")
    return interpretation < policy and all(
        index < policy for index, name in enumerate(events) if name in {"get_open_jobs", "search_rules", "search_policy"})


def no_off_topic_lookup(outputs):
    tools = [event for event in outputs["trajectory"]["tools"] if event["on_topic"] is not None]
    return all(event["on_topic"] for event in tools) if tools else None


def model_calls(outputs):
    return {"key": "model_calls", "score": outputs["trajectory"]["model_calls"]}


def intake_urgency(outputs, reference_outputs):
    """Measure what the deterministic gate caught without the live model."""
    return outputs["intake_urgency"] == reference_outputs["urgency"] if "urgency" in reference_outputs else None



def retrieval_recall_at_3(outputs):
    expected = set(outputs.get("expected_sections", []))
    if expected:
        return len(set(outputs.get("section_ids", [])[:3]) & expected) / len(expected)
    return None


def reply_grounded(outputs):
    if not outputs.get("reply_sources"):
        return None
    from halldesk.replies import grounded_reply
    return grounded_reply(outputs["reply"], outputs["reply_sources"], outputs["section_ids"], outputs.get("job_record"))


CORE = (route, urgency, job, category, priority_deadline, issues, no_write_before_approval, evidence_is_verbatim, model_completed)
TRAJECTORY = (tools_called, tool_order, no_off_topic_lookup, model_calls)
