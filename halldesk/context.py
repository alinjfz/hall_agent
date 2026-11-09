import json
import os
import re
from pathlib import Path

from langsmith import traceable

from halldesk.retrieval import BM25, tokenize


REPAIR_CATEGORIES = ("lighting", "heating", "appliance", "plumbing")
REQUIRED_TOPICS = {"lighting", "escalation", "review"}
TOPICS = set(REPAIR_CATEGORIES) | REQUIRED_TOPICS | {"access", "multiple", "duplicate", "urgent", "follow_up", "fire", "gas", "question"}


# Handbook vocabulary for each interpreted intent or category. The resident's own
# words stay in the query; these terms bridge reports that name no handbook term.
QUERY_HINTS = {
    "chase": "chasing existing repair job recorded target",
    "access": "keys access",
    "multiple": "reporting repairs separate faults",
    "plumbing": "tap shower drip water leak",
    "lighting": "fixed lighting repair target",
    "heating": "verified heating faults",
    "appliance": "verified appliance faults",
    "lockout": "keys access cannot get into room",
    "spare_key": "keys access spare key",
}


def handbook_query(message, *, intent=None, category=None, fitting=None):
    """Build the one handbook query used in production and in the retrieval sweep."""
    hints = [QUERY_HINTS[key] for key in (intent, category) if key in QUERY_HINTS]
    return " ".join(part for part in (message, fitting, *hints) if part and part.strip())


def search_rules(topic):
    return retrieve_policy().get(topic)


def same_policy(current, proposed):
    if current is None:
        return False
    return ({key: value for key, value in current.items() if key != "source"}
            == {key: value for key, value in proposed.items() if key != "source"})


@traceable(name="retrieve_policy")
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


def normalized_terms(message):
    return set(tokenize(message))


def similar_messages(message, others):
    terms = normalized_terms(message)
    documents = {i: normalized_terms(text) for i, text in enumerate(others)}
    if any(message_key(message) == message_key(text) for text in others):
        return True
    index = BM25({key: " ".join(sorted(words)) for key, words in documents.items()})
    return any(
        len(terms & documents[key]) / len(terms | documents[key]) >= 0.82
        for key, _ in index.search(" ".join(sorted(terms)), len(documents))
    )


def relevant_examples(examples, message):
    eligible = {row["id"]: row for row in examples
                if not similar_messages(message, [row["snapshot"]["message"]])}
    documents = {key: " ".join(sorted(normalized_terms(row["snapshot"]["message"])))
                 for key, row in eligible.items()}
    terms = normalized_terms(message)
    ranked = BM25(documents).search(" ".join(sorted(terms)), len(documents))
    return [eligible[key] for key, _ in ranked if len(terms.intersection(documents[key].split())) >= 2][:3]
