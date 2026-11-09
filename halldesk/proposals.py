"""Validated proposal types for the triage and review boundary."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class Job(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int
    location: str
    fitting: str
    deadline: str
    priority: str
    status: str
    category: str
    labels: list[str]
    escalated_at: str | None = None
    created_from: int | None = None
    assignee: str | None = None


class Rule(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    revision: int
    topic: str
    summary: str
    destination: str | None = None


class PolicyRevision(BaseModel):
    id: str
    revision: int


class Retrieval(BaseModel):
    policies: list[PolicyRevision] = Field(default_factory=list)
    example_ids: list[int] = Field(default_factory=list)
    section_ids: list[str] = Field(default_factory=list)


class ContextSnapshot(BaseModel):
    at: str
    jobs: list[Job]
    job_reports: dict[int, dict] = Field(default_factory=dict)
    message: str
    location: str


class ReviewEdits(BaseModel):
    labels: list[str] | None = None
    due_date: str | None = None


class BaseProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule: Rule | None = None
    reply: str | None = None
    reply_source: Rule | None = None
    reply_sources: list[str] = Field(default_factory=list)
    retrieval: Retrieval | None = None
    context_snapshot: ContextSnapshot | None = None
    repeat_of_report_id: int | None = None
    missing_facts: list[str] = Field(default_factory=list)
    review_thread_id: str | None = None


class ManualProposal(BaseProposal):
    kind: Literal["manual"] = "manual"
    reason: str
    urgency: Literal["urgent", "emergency"] | None = None
    destination: str | None = None
    signals: list[str] = Field(default_factory=list)
    related_jobs: list[Job] = Field(default_factory=list)
    evidence: str | None = None
    followup_suggestions: list[str] = Field(default_factory=list)
    duplicate_job_ids: list[int] = Field(default_factory=list)
    possible_duplicates: list[dict] = Field(default_factory=list)


class ChaseProposal(BaseProposal):
    kind: Literal["draft", "escalate"]
    job: Job
    evidence: str
    rule: Rule


class RepairProposal(BaseProposal):
    kind: Literal["create"] = "create"
    rule: Rule
    fitting: str
    labels: list[str]
    category: Literal["lighting", "heating", "appliance", "plumbing"]
    priority: str
    days: int = Field(ge=1)
    policy_deadline: str | None = None
    reviewer_edits: ReviewEdits | None = None


class AnswerProposal(BaseProposal):
    """A handbook answer for staff to approve. It never changes a job."""
    kind: Literal["answer"] = "answer"
    rule: Rule
    evidence: str


class JobQuestion(BaseProposal):
    kind: Literal["job_question"] = "job_question"
    question: str
    rule: Rule
    candidates: list[Job] = Field(min_length=2)
    evidence: str


class RepairQuestion(BaseProposal):
    kind: Literal["lighting_question", "plumbing_question"]
    question: str
    rule: Rule
    fitting: str


class SafetyQuestion(BaseProposal):
    kind: Literal["access_question", "power_question"]
    question: str
    rule: Rule


class SplitProposal(BaseProposal):
    kind: Literal["split"] = "split"
    issues: list[str] = Field(min_length=2, max_length=4)
    rule: Rule


Proposal = Annotated[
    ManualProposal | ChaseProposal | RepairProposal | AnswerProposal | JobQuestion | RepairQuestion | SafetyQuestion | SplitProposal,
    Field(discriminator="kind"),
]
proposal_adapter = TypeAdapter(Proposal)


def parse_proposal(value: dict) -> Proposal:
    return proposal_adapter.validate_python(value)
