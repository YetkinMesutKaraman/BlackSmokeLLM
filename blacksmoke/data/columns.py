"""Canonical column names used inside the service, independent of the CSV file layout."""

from typing import Literal

Polarity = Literal["positive", "negative"]

REVIEW_ID = "review_id"
REVIEW_TEXT = "review_text"
DATE_REVIEWED = "date_reviewed"
TOPIC_COLUMNS: dict[Polarity, str] = {
    "positive": "positive_topics",
    "negative": "negative_topics",
}
