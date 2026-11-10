"""Mask common contact details in traces sent to LangSmith."""

import re
from functools import lru_cache

from langsmith import Client

EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", re.I)
PHONE = re.compile(r"(?<!\w)(?:\+\d{1,3}[ .-]?\(?\d|\(?0\d)[\d ().-]{7,16}\d(?!\w)")
SELF_NAME = re.compile(r"(?i:\b(?:my name is|this is|i am|i'm)\s+)([A-Z][a-z'-]+(?: [A-Z][a-z'-]+){0,2})")
NAME_FIELDS = {"resident_name", "sender_name", "contact_name", "first_name", "last_name", "full_name"}


def anonymize(value):
    if isinstance(value, dict):
        return {key: "[name]" if key.casefold() in NAME_FIELDS and item else anonymize(item)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [anonymize(item) for item in value]
    if isinstance(value, str):
        value = EMAIL.sub("[email]", value)
        value = PHONE.sub("[phone]", value)
        return SELF_NAME.sub(lambda match: match.group().replace(match[1], "[name]"), value)
    return value


@lru_cache(maxsize=1)
def trace_client():
    return Client(anonymizer=anonymize)


@lru_cache(maxsize=1)
def trace_project_id(project_name):
    return trace_client().read_project(project_name=project_name).id
