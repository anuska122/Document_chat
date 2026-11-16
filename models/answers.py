"""Evidence-bound chat output, separate from structured invoice extraction."""
from pydantic import BaseModel, ConfigDict, Field


class CitedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=350)
    source: int = Field(ge=1, strict=True)
    quote: str = Field(min_length=4, max_length=180)


class GroundedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claims: list[CitedClaim] = Field(max_length=3)
