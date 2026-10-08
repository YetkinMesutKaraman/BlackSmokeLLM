"""Execution strategies for the two task kinds."""

import asyncio
import logging
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from blacksmoke.data.columns import DATE_REVIEWED, REVIEW_ID, REVIEW_TEXT
from blacksmoke.data.repository import CsvFormat
from blacksmoke.data.writer import ResultWriter
from blacksmoke.llm.errors import AllTargetsFailedError
from blacksmoke.llm.gateway import LLMGateway
from blacksmoke.llm.resilience import ResiliencePolicy
from blacksmoke.llm.types import AttemptRecord, LLMResult, ModelTarget
from blacksmoke.services.results import TaggingReport, TaskResult, UsageSummary
from blacksmoke.tasks.base import AggregateTask, PerReviewTask, TaskContext
from blacksmoke.tasks.prompts import PromptBuilder
from blacksmoke.tasks.schemas import ReviewTaskRequest

logger = logging.getLogger(__name__)

DEFAULT_CONCURRENCY = 20


class AggregateRunner:
    """Sends all reviews in one prompt and returns one structured result."""

    def __init__(self, gateway: LLMGateway, prompts: PromptBuilder) -> None:
        self._gateway = gateway
        self._prompts = prompts

    async def run(
        self,
        task: AggregateTask,
        request: ReviewTaskRequest,
        context: TaskContext,
        reviews: pd.DataFrame,
        route: Sequence[ModelTarget],
        policy: ResiliencePolicy,
    ) -> TaskResult:
        texts = reviews[REVIEW_TEXT].astype(str).tolist()
        llm_request = self._prompts.aggregate(task, request, context, texts)
        result = await self._gateway.generate(llm_request, route, policy)
        output = task.postprocess(result.output, request)
        return TaskResult[task.output_model](
            task=task.name,
            output=output,
            served_by=result.served_by,
            num_of_reviews=len(texts),
            usage=UsageSummary.from_attempts(result.attempts),
            latency_s=result.latency_s,
            attempts=result.attempts,
        )


class PerReviewRunner:
    """Calls the model once per review with bounded concurrency and writes one row per review.
    A failed review is written with status 'failed' instead of failing the whole run."""

    def __init__(
        self, gateway: LLMGateway, prompts: PromptBuilder, writer: ResultWriter, data_dir: Path
    ) -> None:
        self._gateway = gateway
        self._prompts = prompts
        self._writer = writer
        self._data_dir = data_dir

    async def run(
        self,
        task: PerReviewTask,
        request: ReviewTaskRequest,
        context: TaskContext,
        reviews: pd.DataFrame,
        route: Sequence[ModelTarget],
        policy: ResiliencePolicy,
        output_format: CsvFormat,
    ) -> TaggingReport:
        started = perf_counter()
        build_request = self._prompts.per_review(task, request, context)
        semaphore = asyncio.Semaphore(int(context.setting("concurrency", DEFAULT_CONCURRENCY)))

        async def process(text: str) -> LLMResult | AllTargetsFailedError:
            async with semaphore:
                try:
                    return await self._gateway.generate(build_request(text), route, policy)
                except AllTargetsFailedError as exc:
                    return exc

        records = reviews.to_dict("records")
        outcomes = await asyncio.gather(*(process(str(r[REVIEW_TEXT])) for r in records))

        rows: list[dict[str, Any]] = []
        attempts: list[AttemptRecord] = []
        served_by: Counter[str] = Counter()
        for record, outcome in zip(records, outcomes, strict=True):
            row = {
                key: record[key] for key in (REVIEW_ID, DATE_REVIEWED, REVIEW_TEXT) if key in record
            }
            attempts.extend(outcome.attempts)
            if isinstance(outcome, AllTargetsFailedError):
                row |= task.failed_row(request) | {
                    "status": "failed",
                    "served_by": None,
                    "error": str(outcome),
                }
            else:
                row |= task.review_row(outcome.output, request) | {
                    "status": "ok",
                    "served_by": str(outcome.served_by),
                    "error": None,
                }
                served_by[str(outcome.served_by)] += 1
            rows.append(row)

        path = await asyncio.to_thread(
            self._writer.write,
            pd.DataFrame(rows),
            output_format,
            request.file_path,
            task.output_suffix,
        )
        succeeded = sum(served_by.values())
        logger.info("%s: %d/%d reviews succeeded", task.name, succeeded, len(rows))
        return TaggingReport(
            task=task.name,
            output_file=self._display_path(path),
            num_of_reviews=len(rows),
            succeeded=succeeded,
            failed=len(rows) - succeeded,
            served_by=dict(served_by),
            usage=UsageSummary.from_attempts(attempts),
            latency_s=round(perf_counter() - started, 4),
        )

    def _display_path(self, path: Path) -> str:
        resolved = path.resolve()
        if resolved.is_relative_to(self._data_dir):
            return str(resolved.relative_to(self._data_dir))
        return str(resolved)
