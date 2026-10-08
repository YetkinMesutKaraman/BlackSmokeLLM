"""YAML configuration loading. Each layer owns the schema of its own settings; this module
composes them into one validated `Config` object."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from blacksmoke.core.errors import ConfigError
from blacksmoke.data.preprocessing import ReviewFilter
from blacksmoke.data.repository import CsvFormat
from blacksmoke.llm.pricing import ModelsCatalog
from blacksmoke.llm.resilience import ResiliencePolicy
from blacksmoke.llm.types import GenerationParams, ModelTarget

APP_FILE = "app.yaml"
TASKS_FILE = "tasks.yaml"
MODELS_FILE = "models.yaml"
PROMPTS_DIR = "prompts"


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_dir: Path
    output_dir: Path
    log_level: str = "INFO"
    prompt_delimiter: str = "##"
    review_filter: ReviewFilter = Field(default_factory=ReviewFilter)
    csv_formats: dict[str, CsvFormat]


class RouteTargetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    params: dict[str, Any] = Field(default_factory=dict)


class TaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    route: list[RouteTargetConfig] = Field(..., min_length=1)
    resilience: dict[str, Any] = Field(default_factory=dict)
    review_filter: ReviewFilter | None = None
    settings: dict[str, Any] = Field(default_factory=dict)


class TaskDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    params: dict[str, Any] = Field(default_factory=dict)
    resilience: ResiliencePolicy = Field(default_factory=ResiliencePolicy)


class TasksConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    defaults: TaskDefaults = Field(default_factory=TaskDefaults)
    tasks: dict[str, TaskConfig]

    @field_validator("tasks")
    @classmethod
    def _non_empty(cls, value: dict[str, TaskConfig]) -> dict[str, TaskConfig]:
        if not value:
            raise ValueError("at least one task must be configured")
        return value

    def task(self, name: str) -> TaskConfig:
        try:
            return self.tasks[name]
        except KeyError:
            raise ConfigError(f"Task '{name}' has no entry in {TASKS_FILE}") from None

    def params_for(
        self, task_name: str, provider: str, model: str | None = None
    ) -> GenerationParams:
        """Merge defaults <- task params <- params of the matching route entry.

        A route entry matches on provider and model; when `model` is not in the route (e.g. a
        request override), the first entry of the same provider is used.
        """
        task = self.task(task_name)
        entry = next(
            (t for t in task.route if t.provider == provider and t.model == model), None
        ) or next((t for t in task.route if t.provider == provider), None)
        layers = [self.defaults.params, task.params, entry.params if entry else {}]
        return _merge_params(layers)

    def route_for(self, task_name: str) -> list[ModelTarget]:
        task = self.task(task_name)
        return [
            ModelTarget(
                provider=t.provider,
                model=t.model,
                params=self.params_for(task_name, t.provider, t.model),
            )
            for t in task.route
        ]

    def resilience_for(self, task_name: str) -> ResiliencePolicy:
        overrides = self.task(task_name).resilience
        if not overrides:
            return self.defaults.resilience
        try:
            return ResiliencePolicy.model_validate(
                {**self.defaults.resilience.model_dump(), **overrides}
            )
        except ValidationError as exc:
            raise ConfigError(f"Invalid resilience settings for '{task_name}': {exc}") from exc


class Config(BaseModel):
    app: AppConfig
    tasks: TasksConfig
    models: ModelsCatalog
    prompts_dir: Path


def _merge_params(layers: list[dict[str, Any]]) -> GenerationParams:
    merged: dict[str, Any] = {}
    provider_options: dict[str, Any] = {}
    for layer in layers:
        for key, value in layer.items():
            if key == "provider_options":
                provider_options.update(value or {})
            else:
                merged[key] = value
    merged["provider_options"] = provider_options
    try:
        return GenerationParams.model_validate(merged)
    except ValidationError as exc:
        raise ConfigError(f"Invalid generation params: {exc}") from exc


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Missing config file: {path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return data


def _resolve(base: Path, path: Path) -> Path:
    return path if path.is_absolute() else (base / path).resolve()


def load_config(config_dir: Path) -> Config:
    config_dir = config_dir.resolve()
    project_root = config_dir.parent
    try:
        app = AppConfig.model_validate(_read_yaml(config_dir / APP_FILE))
        tasks = TasksConfig.model_validate(_read_yaml(config_dir / TASKS_FILE))
        models = ModelsCatalog.model_validate(_read_yaml(config_dir / MODELS_FILE))
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration in {config_dir}: {exc}") from exc

    app = app.model_copy(
        update={
            "data_dir": _resolve(project_root, app.data_dir),
            "output_dir": _resolve(project_root, app.output_dir),
        }
    )
    for name in tasks.tasks:
        tasks.route_for(name)

    return Config(app=app, tasks=tasks, models=models, prompts_dir=config_dir / PROMPTS_DIR)
