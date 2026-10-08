import ast
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from blacksmoke.core.errors import (
    DatasetFormatError,
    DatasetNotFoundError,
    InvalidDatasetPathError,
)
from blacksmoke.data.columns import DATE_REVIEWED, REVIEW_ID, REVIEW_TEXT, TOPIC_COLUMNS, Polarity

logger = logging.getLogger(__name__)


class CsvFormat(BaseModel):
    """Physical layout of a reviews CSV. Columns are renamed to the canonical names on read
    and back to these names on write."""

    model_config = ConfigDict(extra="forbid")

    sep: str
    id_column: str = "review_id"
    text_column: str = "review_text"
    date_column: str | None = "date_reviewed"
    topic_columns: dict[Polarity, str] = Field(default_factory=dict)

    def to_canonical(self) -> dict[str, str]:
        mapping = {self.id_column: REVIEW_ID, self.text_column: REVIEW_TEXT}
        if self.date_column:
            mapping[self.date_column] = DATE_REVIEWED
        for polarity, column in self.topic_columns.items():
            mapping[column] = TOPIC_COLUMNS[polarity]
        return mapping

    def from_canonical(self) -> dict[str, str]:
        return {canonical: physical for physical, canonical in self.to_canonical().items()}

    @property
    def read_engine(self) -> str:
        return "c" if len(self.sep) == 1 and self.sep.isascii() else "python"


class CsvReviewRepository:
    """Reads review CSVs from inside `data_dir` only."""

    def __init__(self, data_dir: Path, formats: Mapping[str, CsvFormat]) -> None:
        self._data_dir = data_dir.resolve()
        self._formats = dict(formats)

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    def format(self, name: str) -> CsvFormat:
        try:
            return self._formats[name]
        except KeyError:
            raise DatasetFormatError(f"Unknown CSV format '{name}'") from None

    def resolve(self, file_path: str) -> Path:
        candidate = Path(file_path)
        if not candidate.is_absolute():
            candidate = self._data_dir / candidate
        resolved = candidate.resolve()
        if not resolved.is_relative_to(self._data_dir):
            raise InvalidDatasetPathError(
                f"'{file_path}' is outside the data directory; use a path relative to it"
            )
        if not resolved.is_file():
            raise DatasetNotFoundError(f"Dataset '{file_path}' not found")
        return resolved

    def load(self, file_path: str, format_name: str) -> pd.DataFrame:
        fmt = self.format(format_name)
        path = self.resolve(file_path)
        try:
            df = pd.read_csv(path, sep=fmt.sep, engine=fmt.read_engine)
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
            raise DatasetFormatError(
                f"Cannot parse '{file_path}' as '{format_name}': {exc}"
            ) from exc

        mapping = fmt.to_canonical()
        missing = [column for column in mapping if column not in df.columns]
        if missing:
            raise DatasetFormatError(
                f"'{file_path}' is missing columns {missing} required by format '{format_name}'"
            )

        df = df.loc[:, list(mapping)].rename(columns=mapping)
        for column in (TOPIC_COLUMNS[p] for p in fmt.topic_columns):
            df[column] = df[column].map(_parse_list_cell)
        if DATE_REVIEWED in df.columns:
            df[DATE_REVIEWED] = pd.to_datetime(df[DATE_REVIEWED], errors="coerce")
        logger.info("Loaded %d reviews from %s", len(df), path.name)
        return df.reset_index(drop=True)


def _parse_list_cell(value: Any) -> list[str]:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError) as exc:
        raise DatasetFormatError(f"Invalid topic list cell: {value!r}") from exc
    if not isinstance(parsed, (list, tuple)):
        raise DatasetFormatError(f"Topic cell is not a list: {value!r}")
    return [str(item) for item in parsed]
