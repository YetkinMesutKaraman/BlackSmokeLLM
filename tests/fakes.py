from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from blacksmoke.llm.types import LLMRequest, LLMResponse, ModelTarget, Usage

Step = BaseModel | dict[str, Any] | Exception | Callable[[LLMRequest, ModelTarget], Any]

DEFAULT_USAGE = Usage(input_tokens=1000, output_tokens=100)


class FakeProvider:
    """Scripted provider: each call consumes the next step (or `default` once the script is
    empty). A step is an output (model or dict), an exception to raise, or a callable that
    returns either."""

    def __init__(
        self,
        name: str,
        script: list[Step] | None = None,
        default: Step | None = None,
        usage: Usage = DEFAULT_USAGE,
    ) -> None:
        self.name = name
        self.script = list(script or [])
        self.default = default
        self.usage = usage
        self.calls: list[tuple[LLMRequest, ModelTarget]] = []
        self.closed = False

    async def generate(self, request: LLMRequest, target: ModelTarget) -> LLMResponse:
        self.calls.append((request, target))
        step = self.script.pop(0) if self.script else self.default
        if step is None:
            raise AssertionError(f"{self.name}: no scripted response left")
        if callable(step) and not isinstance(step, (BaseModel, Exception)):
            step = step(request, target)
        if isinstance(step, Exception):
            raise step
        output = step if isinstance(step, BaseModel) else request.output_schema.model_validate(step)
        return LLMResponse(
            output=output,
            usage=self.usage,
            provider=self.name,
            model=target.model,
            latency_s=0.0,
            finish_reason="stop",
        )

    async def aclose(self) -> None:
        self.closed = True
