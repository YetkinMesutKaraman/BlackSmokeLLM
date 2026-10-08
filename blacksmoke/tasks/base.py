"""Task definitions. A task declares what to ask and how to shape the answer; the service layer
decides how to run it (one aggregate call or one call per review)."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pandas as pd
from pydantic import BaseModel

from blacksmoke.tasks.schemas import ReviewTaskRequest


@dataclass(frozen=True)
class TaskContext:
    """Per-task configuration from tasks.yaml that a task may read."""

    prompt: str
    settings: Mapping[str, Any] = field(default_factory=dict)

    def setting(self, key: str, default: Any = None) -> Any:
        return self.settings.get(key, default)


class Task[ReqT: ReviewTaskRequest, OutT: BaseModel](ABC):
    name: ClassVar[str]
    summary: ClassVar[str] = ""
    request_model: ClassVar[type[ReviewTaskRequest]]
    output_model: ClassVar[type[BaseModel]]
    data_format: ClassVar[str] = "raw"
    example_request: ClassVar[dict[str, Any]]

    def output_schema(self, request: ReqT) -> type[BaseModel]:
        """Schema enforced on the model output; may depend on the request."""
        return self.output_model

    def select_reviews(self, request: ReqT, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def prompt_vars(self, request: ReqT, context: TaskContext) -> dict[str, Any]:
        """Task-specific template variables, added to the ones the runner provides."""
        return {}


class AggregateTask[ReqT: ReviewTaskRequest, OutT: BaseModel](Task[ReqT, OutT]):
    """All selected reviews go into one prompt; the result is one structured output.

    Template variables provided by the runner: reviews, num_reviews, delimiter, output_format.
    """

    def postprocess(self, output: BaseModel, request: ReqT) -> OutT:
        return output  # type: ignore[return-value]


class PerReviewTask[ReqT: ReviewTaskRequest, OutT: BaseModel](Task[ReqT, OutT]):
    """One model call per review; results are written to a file, one row per review.

    The system prompt is rendered once per run (a stable, cacheable prefix); the user prompt is
    rendered per review with the extra variable `review`.
    """

    output_suffix: ClassVar[str] = "results"
    output_format: ClassVar[str] = "tagged"

    @abstractmethod
    def review_row(self, output: BaseModel, request: ReqT) -> dict[str, Any]:
        """Result columns for a successfully processed review."""

    @abstractmethod
    def failed_row(self, request: ReqT) -> dict[str, Any]:
        """Result columns for a review whose processing failed."""
