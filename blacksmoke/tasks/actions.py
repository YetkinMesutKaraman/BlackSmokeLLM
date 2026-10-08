from typing import Any

import pandas as pd

from blacksmoke.data.preprocessing import filter_by_topic
from blacksmoke.tasks.base import AggregateTask, TaskContext
from blacksmoke.tasks.registry import register_task
from blacksmoke.tasks.schemas import RecommendedActions, TopicRequest

DEFAULT_MAX_ACTIONS = 10


@register_task
class RecommendActions(AggregateTask[TopicRequest, RecommendedActions]):
    name = "recommend_actions"
    summary = "Recommend actions that address the issues behind a negative topic."
    request_model = TopicRequest
    output_model = RecommendedActions
    data_format = "tagged"
    example_request = {
        "file_path": "results/reviews_com.gardrops_tagged.csv",
        "topic_name": "frequent_app_crashes",
    }

    def select_reviews(self, request: TopicRequest, df: pd.DataFrame) -> pd.DataFrame:
        return filter_by_topic(df, "negative", request.topic_name)

    def prompt_vars(self, request: TopicRequest, context: TaskContext) -> dict[str, Any]:
        return {
            "topic_name": request.topic_name,
            "max_num_of_actions": context.setting("max_num_of_actions", DEFAULT_MAX_ACTIONS),
        }
