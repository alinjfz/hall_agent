from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class Interpretation(BaseModel):
    # Keep defaults for saved interpretations and scripted callers. Live providers
    # must explicitly supply the facts that determine routing, including nulls.
    model_config = ConfigDict(json_schema_extra={"required": [
        "intent", "evidence", "category", "fitting", "other_lighting_works", "lighting_evidence",
        "leak_contained", "power_scope", "can_access_now",
    ]})

    intent: Literal["chase", "new", "access", "multiple", "question", "manual"]
    issues: Annotated[list[str], Field(min_length=2, max_length=4)] | None = None
    candidate_ids: list[int] = []
    evidence: str = Field(default="", description="One exact, contiguous quote supporting the intent and fitting.")
    category: Literal["lighting", "heating", "appliance", "plumbing", "lockout", "spare_key"] | None = None
    fitting: str | None = None
    other_lighting_works: bool | None = None
    lighting_evidence: str | None = None
    leak_contained: bool | None = None
    power_scope: Literal["fitting", "area", "unknown"] | None = None
    can_access_now: bool | None = None
    missing_facts: list[str] = Field(default_factory=list, max_length=4)
