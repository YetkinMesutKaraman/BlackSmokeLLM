from blacksmoke.tasks.base import AggregateTask
from blacksmoke.tasks.registry import register_task
from blacksmoke.tasks.schemas import ReviewsSummary, ReviewTaskRequest


@register_task
class CreateReviewsSummary(AggregateTask[ReviewTaskRequest, ReviewsSummary]):
    name = "create_reviews_summary"
    summary = "Summarize the reviews in two to three English paragraphs."
    request_model = ReviewTaskRequest
    output_model = ReviewsSummary
    example_request = {"file_path": "reviews_2025_02_22_210638_com.gardrops.csv"}
