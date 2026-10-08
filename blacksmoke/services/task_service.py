import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel

from blacksmoke.core.config import TasksConfig
from blacksmoke.core.errors import NoReviewsError, UnknownModelError
from blacksmoke.data.preprocessing import ReviewFilter
from blacksmoke.data.repository import CsvReviewRepository
from blacksmoke.llm.pricing import PricingCatalog
from blacksmoke.llm.types import ModelTarget
from blacksmoke.services.results import TaggingReport, TaskInfo, TaskResult
from blacksmoke.services.runners import AggregateRunner, PerReviewRunner
from blacksmoke.tasks.base import PerReviewTask, Task
from blacksmoke.tasks.registry import TaskRegistry, context_for
from blacksmoke.tasks.schemas import ReviewTaskRequest

logger = logging.getLogger(__name__)

TASKS_PATH_PREFIX = "/v1/tasks"


@dataclass(frozen=True)
class TaskEndpoint:
    """What the API layer needs to expose one task, without knowing task internals."""

    name: str
    summary: str
    request_model: type[ReviewTaskRequest]
    response_model: type[BaseModel]
    example: dict[str, Any]


class TaskService:
    """Application service: resolves the task, its model route and its reviews, then delegates
    execution to the runner that matches the task kind."""

    def __init__(
        self,
        registry: TaskRegistry,
        tasks_config: TasksConfig,
        repository: CsvReviewRepository,
        default_filter: ReviewFilter,
        pricing: PricingCatalog,
        provider_names: list[str],
        aggregate_runner: AggregateRunner,
        per_review_runner: PerReviewRunner,
    ) -> None:
        self._registry = registry
        self._tasks_config = tasks_config
        self._repository = repository
        self._default_filter = default_filter
        self._pricing = pricing
        self._provider_names = provider_names
        self._aggregate = aggregate_runner
        self._per_review = per_review_runner

    def endpoints(self) -> list[TaskEndpoint]:
        return [
            TaskEndpoint(
                name=task.name,
                summary=task.summary,
                request_model=task.request_model,
                response_model=TaggingReport
                if isinstance(task, PerReviewTask)
                else TaskResult[task.output_model],
                example=task.example_request,
            )
            for task in self._registry
        ]

    def list_tasks(self) -> list[TaskInfo]:
        return [
            TaskInfo(
                name=task.name,
                summary=task.summary,
                kind="per_review" if isinstance(task, PerReviewTask) else "aggregate",
                data_format=task.data_format,
                endpoint=f"{TASKS_PATH_PREFIX}/{task.name}",
                route=[t.ref for t in self._tasks_config.route_for(task.name)],
            )
            for task in self._registry
        ]

    async def run(self, task_name: str, request: ReviewTaskRequest) -> TaskResult | TaggingReport:
        task = self._registry.get(task_name)
        request = task.request_model.model_validate(request, from_attributes=True)
        context = context_for(task, self._tasks_config)
        route = self.resolve_route(task_name, request)
        policy = self._tasks_config.resilience_for(task_name)
        reviews = await asyncio.to_thread(self._load_reviews, task, request)
        logger.info(
            "Running %s on %d reviews via %s", task_name, len(reviews), [str(t.ref) for t in route]
        )

        if isinstance(task, PerReviewTask):
            output_format = self._repository.format(task.output_format)
            return await self._per_review.run(
                task, request, context, reviews, route, policy, output_format
            )
        return await self._aggregate.run(task, request, context, reviews, route, policy)

    def resolve_route(self, task_name: str, request: ReviewTaskRequest) -> list[ModelTarget]:
        route = self._tasks_config.route_for(task_name)
        override = request.model_override
        if override is not None:
            if override.provider not in self._provider_names:
                raise UnknownModelError(
                    f"Unknown provider '{override.provider}'; available: {self._provider_names}"
                )
            if not self._pricing.has_model(override.provider, override.model):
                raise UnknownModelError(
                    f"Model '{override.model}' is not in the catalog for '{override.provider}'; "
                    f"available: {self._pricing.models(override.provider)}"
                )
            primary = ModelTarget(
                provider=override.provider,
                model=override.model,
                params=self._tasks_config.params_for(task_name, override.provider, override.model),
            )
            route = [primary, *(t for t in route if t.ref != primary.ref)]
        return route if request.allow_fallback else route[:1]

    def _load_reviews(self, task: Task, request: ReviewTaskRequest) -> pd.DataFrame:
        reviews = self._repository.load(request.file_path, task.data_format)
        reviews = task.select_reviews(request, reviews)
        review_filter = self._tasks_config.task(task.name).review_filter or self._default_filter
        reviews = review_filter.apply(reviews)
        if reviews.empty:
            raise NoReviewsError(f"No reviews left for '{task.name}' after selection and filtering")
        return reviews
