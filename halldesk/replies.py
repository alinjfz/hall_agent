"""Validate facts in drafts before exposing them to staff."""

import re
from datetime import datetime

from pydantic import BaseModel, Field

from halldesk.store import LONDON


class ReplyDraft(BaseModel):
    body: str = Field(min_length=1, max_length=1200)
    section_ids: list[str] = Field(min_length=1, max_length=3)


JOB_REFERENCE = re.compile(r"(?i)(?:\b(?:job|repair|work order)(?:\s+(?:number|no\.?))?\s*[:#]?\s*|#)(\d+)")
DATE_REFERENCE = re.compile(
    r"(?i)\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}(?:/\d{2,4})?|"
    r"\d{1,2}\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:\s+\d{4})?)\b"
)
RELATIVE_PROMISE = re.compile(
    r"(?i)\b(?:today|tomorrow|next (?:week|month)|(?:within|in|after) \d+ (?:hours?|days?)|"
    r"by (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b"
)
TIME_REFERENCE = re.compile(r"\b\d{1,2}:\d{2}\b")


def grounded_reply(body, section_ids, retrieved_ids, job=None):
    if not body.strip() or not section_ids or not set(section_ids) <= set(retrieved_ids):
        return False
    if any(job is None or int(ref) != job["id"] for ref in JOB_REFERENCE.findall(body)):
        return False
    allowed_dates, allowed_times = set(), set()
    if job and job.get("deadline"):
        raw = datetime.fromisoformat(job["deadline"])
        due = raw.astimezone(LONDON)
        allowed_dates = {raw.date().isoformat(), due.date().isoformat()}
        for pattern in ("%d/%m/%Y", "%-d/%-m/%Y", "%d %b %Y", "%-d %b %Y", "%-d %B %Y"):
            allowed_dates.add(due.strftime(pattern).casefold())
        allowed_times = {due.strftime("%H:%M"), due.strftime("%-H:%M")}
    return (not RELATIVE_PROMISE.search(body)
            and set(DATE_REFERENCE.findall(body.casefold())) <= allowed_dates
            and set(TIME_REFERENCE.findall(body)) <= allowed_times)
