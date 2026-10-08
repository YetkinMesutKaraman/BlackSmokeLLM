"""Request models (API input) and output models (LLM structured output) for the tasks.

Output model fields are all required and have no defaults, so the same schema works with
OpenAI strict structured output and Gemini JSON-schema output.
"""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from blacksmoke.data.columns import Polarity
from blacksmoke.llm.types import ModelRef

# ----- Requests -------------------------------------------------------------------------------


class ReviewTaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_path: str = Field(
        ..., description="Reviews CSV path, relative to the configured data directory."
    )
    model_override: ModelRef | None = Field(
        None,
        description="Use this provider/model as the primary target. Configured fallbacks "
        "still apply unless allow_fallback is false.",
    )
    allow_fallback: bool = Field(
        True, description="Try the task's fallback targets when the primary target fails."
    )


class TopicRequest(ReviewTaskRequest):
    """For tasks scoped to one topic of a fixed polarity. Requires a tagged dataset."""

    topic_name: str = Field(..., min_length=1)


class PolarTopicRequest(TopicRequest):
    topic_polarity: Polarity


class TagReviewsRequest(ReviewTaskRequest):
    positive_topics: list[str] = Field(default_factory=list)
    negative_topics: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _clean_topics(self) -> "TagReviewsRequest":
        self.positive_topics = _unique_non_empty(self.positive_topics)
        self.negative_topics = _unique_non_empty(self.negative_topics)
        if not self.positive_topics and not self.negative_topics:
            raise ValueError("provide at least one positive or negative topic")
        return self


def _unique_non_empty(values: list[str]) -> list[str]:
    return list(dict.fromkeys(v.strip() for v in values if v and v.strip()))


# ----- Outputs --------------------------------------------------------------------------------


class ReviewsSummary(BaseModel):
    paragraphs: list[str] = Field(
        ...,
        min_length=2,
        max_length=3,
        description="Summary paragraphs; key aspects are highlighted with **.",
    )


class ExtractedTopics(BaseModel):
    positive_topics: list[str] = Field(..., description="Positive topic phrases.")
    negative_topics: list[str] = Field(..., description="Negative topic phrases.")


class BugsAndFeatures(BaseModel):
    bugs: list[str] = Field(..., description="Bugs reported in the reviews.")
    feature_requests: list[str] = Field(..., description="Features requested in the reviews.")


class RecommendedActions(BaseModel):
    recommended_actions: list[str] = Field(..., description="Actions to address the issues.")


class MarketingStrategies(BaseModel):
    marketing_strategies: list[str] = Field(..., description="Marketing strategy ideas.")


class Subtopics(BaseModel):
    subtopics: list[str] = Field(..., description="Sub-topic phrases of the main topic.")


class ReviewTags(BaseModel):
    positive_topics: list[str] = Field(..., description="Positive topics present in the review.")
    negative_topics: list[str] = Field(..., description="Negative topics present in the review.")
