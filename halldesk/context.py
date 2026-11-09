import json
import os
import re
from pathlib import Path

from langsmith import traceable


REPAIR_CATEGORIES = ("lighting", "heating", "appliance", "plumbing")
REQUIRED_TOPICS = {"lighting", "escalation", "review"}
TOPICS = set(REPAIR_CATEGORIES) | REQUIRED_TOPICS | {"access", "multiple", "duplicate", "urgent", "follow_up", "fire", "gas"}


def search_rules(topic):
    return retrieve_policy().get(topic)


def same_policy(current, proposed):
    if current is None:
        return False
    return ({key: value for key, value in current.items() if key != "source"}
            == {key: value for key, value in proposed.items() if key != "source"})


@traceable(name="retrieve_policy", run_type="retriever")
def retrieve_policy():
    """Retrieve the approved policy sections for the supported workflow."""
    path = Path(os.getenv("HALL_POLICY_FILE") or Path(__file__).with_name("rules.json"))
    try:
        rules = json.loads(path.read_text(encoding="utf-8"))
        required = ("id", "source", "summary")
        if not isinstance(rules, list) or any(
            not isinstance(rule, dict)
            or not all(isinstance(rule.get(key), str) and rule[key].strip() for key in required)
            or type(rule.get("revision")) is not int
            or rule["revision"] < 1
            for rule in rules
        ):
            return {}
        ids = [rule["id"] for rule in rules]
        topics = [rule.get("topic") for rule in rules]
        if (len(ids) != len(set(ids)) or len(topics) != len(set(topics))
                or not REQUIRED_TOPICS <= set(topics) <= TOPICS):
            return {}
        by_topic = {rule["topic"]: rule for rule in rules}
        if any("clarification_question" in rule and (
            not isinstance(rule["clarification_question"], str) or not rule["clarification_question"].strip()
        ) for rule in rules):
            return {}
        escalation = by_topic["escalation"]
        if escalation.get("boundary") != "strictly_after" or not isinstance(escalation.get("destination"), str):
            return {}
        if not escalation["destination"].strip():
            return {}
        targets = [by_topic["lighting"].get(band) for band in ("other_lighting_works", "no_other_lighting")]
        targets.extend(by_topic[category].get("target") for category in REPAIR_CATEGORIES
                       if category != "lighting" and category in by_topic)
        for terms in targets:
            if (not isinstance(terms, dict) or terms.get("priority") not in {"P2", "P3"}
                    or type(terms.get("days")) is not int or not 1 <= terms["days"] <= 90):
                return {}
        return by_topic
    except (OSError, ValueError, TypeError):
        return {}


def message_key(message):
    return " ".join(message.casefold().split())


def relevant_examples(examples, message):
    words = set(re.findall(r"[a-z]{4,}", message.casefold()))
    ranked = []
    for example in examples:
        previous = example["snapshot"]["message"]
        if message_key(previous) == message_key(message):
            continue
        overlap = words.intersection(re.findall(r"[a-z]{4,}", previous.casefold()))
        if len(overlap) >= 2:
            ranked.append((len(overlap), example))
    ranked.sort(key=lambda item: (-item[0], -item[1]["id"]))
    return [{"id": example["id"], "report": example["snapshot"]["message"][:500],
             "expected_handoff": example["expected"]["kind"]}
            for _, example in ranked[:3]]
