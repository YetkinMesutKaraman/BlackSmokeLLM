import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateError, TemplateNotFound
from pydantic import BaseModel

from blacksmoke.core.errors import ConfigError
from blacksmoke.data.preprocessing import format_for_prompt
from blacksmoke.llm.types import LLMRequest
from blacksmoke.tasks.base import AggregateTask, PerReviewTask, TaskContext
from blacksmoke.tasks.schemas import ReviewTaskRequest

PromptPart = Literal["system", "user"]


class PromptRenderer:
    """Renders `<prompts_dir>/<prompt>/{system,user}.md.j2`. Undefined variables raise."""

    def __init__(self, prompts_dir: Path) -> None:
        self._env = Environment(
            loader=FileSystemLoader(prompts_dir),
            undefined=StrictUndefined,
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render(self, prompt: str, part: PromptPart, variables: dict[str, Any]) -> str:
        path = f"{prompt}/{part}.md.j2"
        try:
            return self._env.get_template(path).render(**variables).strip()
        except TemplateNotFound:
            raise ConfigError(f"Prompt template not found: {path}") from None
        except TemplateError as exc:
            raise ConfigError(f"Cannot render prompt {path}: {exc}") from exc


class PromptBuilder:
    """Turns a task + request into provider-neutral `LLMRequest`s."""

    def __init__(self, renderer: PromptRenderer, delimiter: str) -> None:
        self._renderer = renderer
        self._delimiter = delimiter

    def aggregate(
        self,
        task: AggregateTask,
        request: ReviewTaskRequest,
        context: TaskContext,
        review_texts: Sequence[str],
    ) -> LLMRequest:
        schema = task.output_schema(request)
        variables = {
            **task.prompt_vars(request, context),
            "reviews": format_for_prompt(review_texts, self._delimiter),
            "num_reviews": len(review_texts),
            "delimiter": self._delimiter,
            "output_format": schema_hint(schema),
        }
        return LLMRequest(
            system_prompt=self._renderer.render(context.prompt, "system", variables),
            user_prompt=self._renderer.render(context.prompt, "user", variables),
            output_schema=schema,
        )

    def per_review(
        self, task: PerReviewTask, request: ReviewTaskRequest, context: TaskContext
    ) -> Callable[[str], LLMRequest]:
        schema = task.output_schema(request)
        variables = {**task.prompt_vars(request, context), "output_format": schema_hint(schema)}
        system_prompt = self._renderer.render(context.prompt, "system", variables)

        def build(review_text: str) -> LLMRequest:
            user_prompt = self._renderer.render(
                context.prompt, "user", {**variables, "review": review_text}
            )
            return LLMRequest(system_prompt, user_prompt, schema)

        return build


def schema_hint(schema: type[BaseModel]) -> str:
    """Compact JSON shape of a schema, e.g. {"bugs": ["string"]}, for use inside prompts."""
    json_schema = schema.model_json_schema()
    definitions = json_schema.get("$defs", {})

    def example(node: dict[str, Any]) -> Any:
        if "$ref" in node:
            return example(definitions[node["$ref"].split("/")[-1]])
        kind = node.get("type")
        if kind == "array":
            return [example(node.get("items", {}))]
        if kind == "object":
            return {key: example(value) for key, value in node.get("properties", {}).items()}
        if kind in ("integer", "number"):
            return 0
        if kind == "boolean":
            return True
        return "string"

    return json.dumps(example(json_schema), ensure_ascii=False)
