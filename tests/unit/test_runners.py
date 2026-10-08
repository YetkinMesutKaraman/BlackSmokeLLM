from blacksmoke.llm import errors
from blacksmoke.tasks.schemas import ReviewTaskRequest, TagReviewsRequest
from tests.conftest import RAW_FILE


def tag_by_content(request, target):
    review = request.user_prompt
    if "upload photos" in review:
        raise errors.BadRequestError("rejected")
    if "Crashes again" in review:
        return {
            "positive_topics": [],
            "negative_topics": ["frequent_app_crashes", "frequent_app_crashes"],
        }
    return {"positive_topics": ["affordable_prices"], "negative_topics": []}


async def test_tagging_keeps_review_ids_aligned_when_a_review_in_the_middle_fails(
    container, providers
):
    providers["openai"].default = tag_by_content
    request = TagReviewsRequest(
        file_path=RAW_FILE,
        positive_topics=["affordable_prices"],
        negative_topics=["frequent_app_crashes"],
    )
    report = await container.task_service.run("tag_reviews_with_topics", request)

    assert (report.num_of_reviews, report.succeeded, report.failed) == (4, 3, 1)
    assert report.served_by == {"openai/gpt-4o": 3}
    assert report.usage.input_tokens == 3 * 1000

    output = container.repository.load(report.output_file, "tagged").set_index("review_id")
    assert output.loc[2, "negative_topics"] == []
    assert output.loc[5, "negative_topics"] == ["frequent_app_crashes"]
    assert output.loc[1, "positive_topics"] == ["affordable_prices"]
    assert output.loc[3, "positive_topics"] == ["affordable_prices"]

    raw = container.repository.resolve(report.output_file).read_text(encoding="utf-8")
    header = raw.splitlines()[0].split("¤")
    assert header[:5] == [
        "review_id",
        "date_reviewed",
        "review_text",
        "ai_positive_topics",
        "ai_negative_topics",
    ]
    assert {"status", "served_by", "error", "dropped_topics"} <= set(header)


async def test_aggregate_task_postprocesses_output(container, providers):
    providers["openai"].default = {
        "positive_topics": ["Easy To Use", "easy to use"],
        "negative_topics": ["App Crashes"],
    }
    result = await container.task_service.run(
        "extract_unsupervised_topics", ReviewTaskRequest(file_path=RAW_FILE)
    )
    assert result.output.positive_topics == ["easy_to_use"]
    assert result.output.negative_topics == ["app_crashes"]
    assert result.num_of_reviews == 4
    assert "##" in providers["openai"].calls[0][0].user_prompt
