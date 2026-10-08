import pandas as pd
import pytest

from blacksmoke.core.errors import (
    DatasetFormatError,
    DatasetNotFoundError,
    InvalidDatasetPathError,
)
from blacksmoke.data.preprocessing import ReviewFilter, filter_by_topic, format_for_prompt
from blacksmoke.data.repository import CsvFormat, CsvReviewRepository
from blacksmoke.data.writer import ResultWriter
from tests.conftest import RAW_FILE, TAGGED_FILE

FORMATS = {
    "raw": CsvFormat(sep=";"),
    "tagged": CsvFormat(
        sep="¤", topic_columns={"positive": "ai_positive_topics", "negative": "ai_negative_topics"}
    ),
}


@pytest.fixture
def repository(data_dir) -> CsvReviewRepository:
    return CsvReviewRepository(data_dir, FORMATS)


@pytest.mark.parametrize("path", ["../outside.csv", "/etc/passwd"])
def test_paths_outside_data_dir_are_rejected(repository, path):
    with pytest.raises(InvalidDatasetPathError):
        repository.resolve(path)


def test_missing_file(repository):
    with pytest.raises(DatasetNotFoundError):
        repository.load("nope.csv", "raw")


def test_raw_load_uses_canonical_columns(repository):
    df = repository.load(RAW_FILE, "raw")
    assert list(df.columns) == ["review_id", "review_text", "date_reviewed"]
    assert pd.api.types.is_datetime64_any_dtype(df["date_reviewed"])


def test_tagged_load_parses_topic_lists(repository):
    df = repository.load(TAGGED_FILE, "tagged")
    assert df.loc[df.review_id == 2, "negative_topics"].item() == ["frequent_app_crashes"]
    assert df.loc[df.review_id == 2, "positive_topics"].item() == []


def test_raw_file_is_not_a_tagged_dataset(repository):
    with pytest.raises(DatasetFormatError, match="missing columns"):
        repository.load(RAW_FILE, "tagged")


def test_filter_by_topic(repository):
    df = filter_by_topic(repository.load(TAGGED_FILE, "tagged"), "negative", "frequent_app_crashes")
    assert sorted(df.review_id) == [2, 5]


def test_filter_by_topic_requires_topic_columns(repository):
    with pytest.raises(DatasetFormatError):
        filter_by_topic(repository.load(RAW_FILE, "raw"), "positive", "x")


def test_review_filter_keeps_most_recent_then_drops_short(repository):
    df = ReviewFilter(max_reviews=3, min_length=50).apply(repository.load(RAW_FILE, "raw"))
    assert list(df.review_id) == [5, 3]


def test_format_for_prompt():
    assert format_for_prompt(["a", "b"], "##") == "##a##b##"


def test_writer_output_round_trips_through_tagged_reader(repository, data_dir):
    rows = pd.DataFrame(
        [
            {
                "review_id": 7,
                "date_reviewed": "2025-01-01",
                "review_text": "x",
                "positive_topics": ["a"],
                "negative_topics": [],
                "status": "ok",
            },
        ]
    )
    path = ResultWriter(data_dir / "results").write(rows, FORMATS["tagged"], "src.csv", "tagged")
    loaded = repository.load(str(path.relative_to(data_dir)), "tagged")
    assert loaded.loc[0, "positive_topics"] == ["a"]
    assert path.name.startswith("src_tagged_")
