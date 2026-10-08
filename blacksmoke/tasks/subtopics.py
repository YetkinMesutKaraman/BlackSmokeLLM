from typing import Any

import pandas as pd

from blacksmoke.data.preprocessing import filter_by_topic
from blacksmoke.tasks.base import AggregateTask, TaskContext
from blacksmoke.tasks.registry import register_task
from blacksmoke.tasks.schemas import PolarTopicRequest, Subtopics
from blacksmoke.tasks.topics import normalize_topics

DEFAULT_MAX_SUBTOPICS = 5


@register_task
class ExtractSubtopics(AggregateTask[PolarTopicRequest, Subtopics]):
    name = "extract_subtopics"
    summary = "Break one topic down into concise sub-topics of the same polarity."
    request_model = PolarTopicRequest
    output_model = Subtopics
    data_format = "tagged"
    example_request = {
        "file_path": "results/reviews_com.gardrops_tagged.csv",
        "topic_polarity": "negative",
        "topic_name": "refund_issues",
    }

    def select_reviews(self, request: PolarTopicRequest, df: pd.DataFrame) -> pd.DataFrame:
        return filter_by_topic(df, request.topic_polarity, request.topic_name)

    def prompt_vars(self, request: PolarTopicRequest, context: TaskContext) -> dict[str, Any]:
        return {
            "topic_polarity": request.topic_polarity,
            "topic_name": request.topic_name,
            "max_num_of_subtopics": context.setting("max_num_of_subtopics", DEFAULT_MAX_SUBTOPICS),
        }

    def postprocess(self, output: Subtopics, request: PolarTopicRequest) -> Subtopics:
        return Subtopics(subtopics=normalize_topics(output.subtopics))
