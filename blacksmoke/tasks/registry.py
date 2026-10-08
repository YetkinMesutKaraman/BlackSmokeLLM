import importlib
import logging
import pkgutil
from collections.abc import Iterator
from typing import TypeVar

from blacksmoke.core.config import TasksConfig
from blacksmoke.core.errors import ConfigError, UnknownTaskError
from blacksmoke.llm.pricing import PricingCatalog
from blacksmoke.tasks.base import AggregateTask, PerReviewTask, Task, TaskContext
from blacksmoke.tasks.prompts import PromptBuilder

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=type[Task])

SAMPLE_REVIEWS = ["The app is fast and easy to use.", "It crashes every time I upload a photo."]


class TaskRegistry:
    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    def register(self, task_cls: T) -> T:
        existing = self._tasks.get(task_cls.name)
        if existing is not None and type(existing) is not task_cls:
            raise ConfigError(f"Task name '{task_cls.name}' is registered twice")
        self._tasks[task_cls.name] = task_cls()
        return task_cls

    def get(self, name: str) -> Task:
        try:
            return self._tasks[name]
        except KeyError:
            raise UnknownTaskError(f"Unknown task '{name}'") from None

    def names(self) -> list[str]:
        return sorted(self._tasks)

    def __iter__(self) -> Iterator[Task]:
        return iter(self._tasks[name] for name in self.names())

    def __len__(self) -> int:
        return len(self._tasks)

    def validate(
        self,
        tasks_config: TasksConfig,
        provider_names: list[str],
        pricing: PricingCatalog,
        prompt_builder: PromptBuilder,
    ) -> None:
        """Fail fast at startup if code, YAML, providers, and prompts disagree."""
        problems: list[str] = []
        registered, configured = set(self._tasks), set(tasks_config.tasks)
        problems += [
            f"task '{n}' is registered but missing in tasks.yaml"
            for n in sorted(registered - configured)
        ]
        problems += [
            f"task '{n}' is in tasks.yaml but not registered"
            for n in sorted(configured - registered)
        ]

        for name in sorted(registered & configured):
            task, task_config = self._tasks[name], tasks_config.task(name)
            for target in task_config.route:
                if target.provider not in provider_names:
                    problems.append(f"task '{name}' routes to unknown provider '{target.provider}'")
                elif not pricing.has_model(target.provider, target.model):
                    logger.warning(
                        "Task '%s' routes to %s/%s, which has no pricing in models.yaml",
                        name,
                        target.provider,
                        target.model,
                    )
            try:
                self._render_sample(task, context_for(task, tasks_config), prompt_builder)
            except Exception as exc:
                problems.append(f"task '{name}' prompt check failed: {exc}")

        if problems:
            raise ConfigError("Invalid task setup:\n- " + "\n- ".join(problems))

    @staticmethod
    def _render_sample(task: Task, context: TaskContext, builder: PromptBuilder) -> None:
        request = task.request_model.model_validate(task.example_request)
        if isinstance(task, AggregateTask):
            builder.aggregate(task, request, context, SAMPLE_REVIEWS)
        elif isinstance(task, PerReviewTask):
            builder.per_review(task, request, context)(SAMPLE_REVIEWS[0])
        else:
            raise ConfigError(f"Task '{task.name}' must extend AggregateTask or PerReviewTask")


def context_for(task: Task, tasks_config: TasksConfig) -> TaskContext:
    task_config = tasks_config.task(task.name)
    return TaskContext(prompt=task_config.prompt or task.name, settings=task_config.settings)


default_registry = TaskRegistry()
register_task = default_registry.register


def load_builtin_tasks() -> TaskRegistry:
    """Import every module in `blacksmoke.tasks` so their `@register_task` decorators run."""
    import blacksmoke.tasks as package

    for module in pkgutil.iter_modules(package.__path__):
        importlib.import_module(f"{package.__name__}.{module.name}")
    return default_registry
