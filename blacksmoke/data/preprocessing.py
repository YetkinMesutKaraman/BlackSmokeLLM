from collections.abc import Iterable

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from blacksmoke.core.errors import DatasetFormatError
from blacksmoke.data.columns import DATE_REVIEWED, REVIEW_TEXT, TOPIC_COLUMNS, Polarity


class ReviewFilter(BaseModel):
    """Keeps the most recent `max_reviews` reviews, then drops empty and short ones."""

    model_config = ConfigDict(extra="forbid")

    max_reviews: int | None = Field(1000, ge=1)
    min_length: int = Field(50, ge=0)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if DATE_REVIEWED in df.columns:
            df = df.sort_values(by=DATE_REVIEWED, ascending=False)
        if self.max_reviews is not None:
            df = df.iloc[: self.max_reviews]
        df = df.dropna(subset=[REVIEW_TEXT])
        df = df[df[REVIEW_TEXT].astype(str).str.len() > self.min_length]
        return df.reset_index(drop=True)


def filter_by_topic(df: pd.DataFrame, polarity: Polarity, topic_name: str) -> pd.DataFrame:
    column = TOPIC_COLUMNS[polarity]
    if column not in df.columns:
        raise DatasetFormatError(f"Dataset has no '{polarity}' topic column; use a tagged dataset")
    mask = df[column].map(lambda topics: topic_name in topics)
    return df[mask].reset_index(drop=True)


def format_for_prompt(texts: Iterable[str], delimiter: str) -> str:
    return f"{delimiter}{delimiter.join(texts)}{delimiter}"
