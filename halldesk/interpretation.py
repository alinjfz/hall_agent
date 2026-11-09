from typing import Annotated, Literal

from pydantic import BaseModel, Field


class Interpretation(BaseModel):
    intent: Literal["chase", "new", "access", "multiple", "manual"]
    issues: Annotated[list[str], Field(min_length=2, max_length=4)] | None = None
    candidate_ids: list[int] = []
    evidence: str = ""
    category: Literal["lighting", "heating", "appliance", "plumbing", "lockout", "spare_key"] | None = None
    fitting: str | None = None
    other_lighting_works: bool | None = None
    lighting_evidence: str | None = None
    leak_contained: bool | None = None
    power_scope: Literal["fitting", "area", "unknown"] | None = None
    can_access_now: bool | None = None
    missing_facts: list[str] = Field(default_factory=list, max_length=4)
