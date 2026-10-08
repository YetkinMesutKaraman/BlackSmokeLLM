from typing import Any

from blacksmoke.tasks.base import AggregateTask, TaskContext
from blacksmoke.tasks.registry import register_task
from blacksmoke.tasks.schemas import BugsAndFeatures, ReviewTaskRequest

DEFAULT_MAX_ITEMS = 15


@register_task
class FindBugsAndFeatures(AggregateTask[ReviewTaskRequest, BugsAndFeatures]):
    name = "find_bugs_and_features"
    summary = "Extract reported bugs and feature requests from the reviews."
    request_model = ReviewTaskRequest
    output_model = BugsAndFeatures
    example_request = {"file_path": "reviews_2025_02_22_210638_com.gardrops.csv"}

    def prompt_vars(self, request: ReviewTaskRequest, context: TaskContext) -> dict[str, Any]:
        return {"max_num_of_items": context.setting("max_num_of_items", DEFAULT_MAX_ITEMS)}
