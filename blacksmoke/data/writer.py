import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from blacksmoke.data.repository import CsvFormat

logger = logging.getLogger(__name__)


class ResultWriter:
    """Writes per-review results as CSV in a given format, so outputs can be read back as
    inputs (e.g. tagging output feeds the topic-scoped tasks)."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir

    def write(self, df: pd.DataFrame, fmt: CsvFormat, source_file: str, suffix: str) -> Path:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        path = self._output_dir / f"{Path(source_file).stem}_{suffix}_{timestamp}.csv"

        out = df.rename(columns=fmt.from_canonical())
        list_columns = [c for c in out.columns if out[c].map(lambda v: isinstance(v, list)).any()]
        for column in list_columns:
            out[column] = out[column].map(repr)
        out.to_csv(path, sep=fmt.sep, index=False)
        logger.info("Wrote %d rows to %s", len(out), path)
        return path
