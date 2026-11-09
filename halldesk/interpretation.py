from typing import Annotated, Literal

from pydantic import BaseModel, Field


PolicyTopic = Literal[
    "lighting", "heating", "appliance", "plumbing", "escalation", "review", "access",
    "multiple", "duplicate", "urgent", "follow_up", "fire", "gas",
]


class Interpretation(BaseModel):
    intent: Literal["chase", "new", "access", "multiple", "manual"]
    issues: Annotated[list[str], Field(min_length=2, max_length=4)] | None = None
    candidate_ids: list[int] = []
    evidence: str = ""
    category: Literal["lighting", "heating", "appliance", "plumbing", "lockout", "spare_key"] | None = None
    fitting: str | None = None
    other_lighting_works: bool | None = None
    lighting_evidence: str | None = None
    safety_issue: Literal["fire", "gas", "fire_equipment", "building_hazard", "dark_route",
                          "unsafe_door", "lockout", "identifiable_key"] | None = None
    safety_evidence: str | None = None
    missing_facts: list[str] = Field(default_factory=list, max_length=4)
    policy_topics: list[PolicyTopic] = Field(default_factory=list, max_length=3)
    policy_query: str | None = Field(default=None, max_length=160)
    look_up_jobs: bool = False


