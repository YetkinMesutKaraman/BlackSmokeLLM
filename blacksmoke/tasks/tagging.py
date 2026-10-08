from typing import Any, Literal

from pydantic import BaseModel, Field, create_model

from blacksmoke.data.columns import TOPIC_COLUMNS
from blacksmoke.tasks.base import PerReviewTask, TaskContext
from blacksmoke.tasks.registry import register_task
from blacksmoke.tasks.schemas import ReviewTags, TagReviewsRequest


@register_task
class TagReviewsWithTopics(PerReviewTask[TagReviewsRequest, ReviewTags]):
    name = "tag_reviews_with_topics"
    summary = (
        "Tag each review with the given positive/negative topics and write a tagged CSV "
        "that the topic-scoped tasks can read."
    )
    request_model = TagReviewsRequest
    output_model = ReviewTags
    output_suffix = "tagged"
    example_request = {
        "file_path": "reviews_2025_02_22_210638_com.gardrops.csv",
        "positive_topics": ["user-friendly_interface", "affordable_prices"],
        "negative_topics": ["frequent_app_crashes", "refund_issues"],
    }

    def output_schema(self, request: TagReviewsRequest) -> type[BaseModel]:
        return build_tag_schema(request.positive_topics, request.negative_topics)

    def prompt_vars(self, request: TagReviewsRequest, context: TaskContext) -> dict[str, Any]:
        return {
            "positive_topics": request.positive_topics,
            "negative_topics": request.negative_topics,
        }

    def review_row(self, output: BaseModel, request: TagReviewsRequest) -> dict[str, Any]:
        positive, dropped_positive = _keep_allowed(output.positive_topics, request.positive_topics)
        negative, dropped_negative = _keep_allowed(output.negative_topics, request.negative_topics)
        return {
            TOPIC_COLUMNS["positive"]: positive,
            TOPIC_COLUMNS["negative"]: negative,
            "dropped_topics": dropped_positive + dropped_negative,
        }

    def failed_row(self, request: TagReviewsRequest) -> dict[str, Any]:
        return {TOPIC_COLUMNS["positive"]: [], TOPIC_COLUMNS["negative"]: [], "dropped_topics": []}


def build_tag_schema(positive_topics: list[str], negative_topics: list[str]) -> type[BaseModel]:
    """Output schema whose lists only accept the given topics, so the model cannot invent or
    swap topics between polarities."""

    def topic_list(topics: list[str], description: str) -> tuple[Any, Any]:
        item_type = Literal[tuple(topics)] if topics else str  # type: ignore[valid-type]
        return list[item_type], Field(..., description=description)

    return create_model(
        "ReviewTags",
        positive_topics=topic_list(positive_topics, "Positive topics present in the review."),
        negative_topics=topic_list(negative_topics, "Negative topics present in the review."),
    )


def _keep_allowed(tagged: list[str], allowed: list[str]) -> tuple[list[str], list[str]]:
    allowed_set = set(allowed)
    unique = list(dict.fromkeys(tagged))
    return [t for t in unique if t in allowed_set], [t for t in unique if t not in allowed_set]
