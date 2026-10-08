"""Response envelopes returned by the service layer (and serialized by the API layer)."""

from typing import Literal

from pydantic import BaseModel, Field

from blacksmoke.llm.types import AttemptRecord, ModelRef, sum_cost, sum_usage


class UsageSummary(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float | None = Field(
        None, description="Total cost over all attempts; null if a model has no known price."
    )

    @classmethod
    def from_attempts(cls, attempts: list[AttemptRecord]) -> "UsageSummary":
        return cls(**sum_usage(attempts).model_dump(), cost_usd=sum_cost(attempts))


class TaskResult[T: BaseModel](BaseModel):
    task: str
    output: T
    served_by: ModelRef = Field(..., description="The provider/model that produced the output.")
    num_of_reviews: int
    usage: UsageSummary
    latency_s: float
    attempts: list[AttemptRecord] = Field(
        ..., description="Every model call made, including retries, repairs, and fallbacks."
    )


class TaggingReport(BaseModel):
    task: str
    output_file: str = Field(
        ...,
        description="Result CSV; relative to the data directory when inside it, so it can be "
        "passed as file_path to topic-scoped tasks.",
    )
    num_of_reviews: int
    succeeded: int
    failed: int
    served_by: dict[str, int] = Field(..., description="Reviews processed per provider/model.")
    usage: UsageSummary
    latency_s: float


class TaskInfo(BaseModel):
    name: str
    summary: str
    kind: Literal["aggregate", "per_review"]
    data_format: str
    endpoint: str
    route: list[ModelRef]
