from typing import Any

from blacksmoke.tasks.base import AggregateTask, TaskContext
from blacksmoke.tasks.registry import register_task
from blacksmoke.tasks.schemas import ExtractedTopics, ReviewTaskRequest

DEFAULT_MAX_TOPICS = 19


@register_task
class ExtractUnsupervisedTopics(AggregateTask[ReviewTaskRequest, ExtractedTopics]):
    name = "extract_unsupervised_topics"
    summary = "Discover positive and negative topics in the reviews."
    request_model = ReviewTaskRequest
    output_model = ExtractedTopics
    example_request = {"file_path": "reviews_2025_02_22_210638_com.gardrops.csv"}

    def prompt_vars(self, request: ReviewTaskRequest, context: TaskContext) -> dict[str, Any]:
        return {"max_num_of_topics": context.setting("max_num_of_topics", DEFAULT_MAX_TOPICS)}

    def postprocess(self, output: ExtractedTopics, request: ReviewTaskRequest) -> ExtractedTopics:
        return ExtractedTopics(
            positive_topics=normalize_topics(output.positive_topics),
            negative_topics=normalize_topics(output.negative_topics),
        )


def normalize_topics(topics: list[str]) -> list[str]:
    """snake_case, lowercase, de-duplicated (order preserved)."""
    normalized = ("_".join(topic.strip().lower().split()) for topic in topics)
    return list(dict.fromkeys(topic for topic in normalized if topic))
