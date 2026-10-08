from typing import Any

import pandas as pd

from blacksmoke.data.preprocessing import filter_by_topic
from blacksmoke.tasks.base import AggregateTask, TaskContext
from blacksmoke.tasks.registry import register_task
from blacksmoke.tasks.schemas import MarketingStrategies, TopicRequest

DEFAULT_MAX_STRATEGIES = 10


@register_task
class RecommendMarketingStrategies(AggregateTask[TopicRequest, MarketingStrategies]):
    name = "recommend_marketing_strategies"
    summary = "Recommend marketing strategies that build on a positive topic."
    request_model = TopicRequest
    output_model = MarketingStrategies
    data_format = "tagged"
    example_request = {
        "file_path": "results/reviews_com.gardrops_tagged.csv",
        "topic_name": "user-friendly_interface",
    }

    def select_reviews(self, request: TopicRequest, df: pd.DataFrame) -> pd.DataFrame:
        return filter_by_topic(df, "positive", request.topic_name)

    def prompt_vars(self, request: TopicRequest, context: TaskContext) -> dict[str, Any]:
        return {
            "topic_name": request.topic_name,
            "max_num_of_marketing_strategies": context.setting(
                "max_num_of_marketing_strategies", DEFAULT_MAX_STRATEGIES
            ),
        }
