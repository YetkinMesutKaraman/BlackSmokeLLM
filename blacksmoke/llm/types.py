"""Provider-neutral types exchanged between tasks, the gateway, and provider adapters."""

from dataclasses import dataclass, field, replace
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str = Field(..., description="Provider name, e.g. 'openai' or 'google'.")
    model: str = Field(..., description="Model name as known by the provider.")

    def __str__(self) -> str:
        return f"{self.provider}/{self.model}"


class GenerationParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int = 1024
    timeout_s: float = 60.0
    provider_options: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific options passed through by the adapter "
        "(e.g. Gemini 'thinking_budget', OpenAI 'reasoning').",
    )


class ModelTarget(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    params: GenerationParams = Field(default_factory=GenerationParams)

    @property
    def ref(self) -> ModelRef:
        return ModelRef(provider=self.provider, model=self.model)

    def with_max_output_tokens(self, max_output_tokens: int) -> "ModelTarget":
        params = self.params.model_copy(update={"max_output_tokens": max_output_tokens})
        return self.model_copy(update={"params": params})


@dataclass(frozen=True)
class LLMRequest:
    system_prompt: str
    user_prompt: str
    output_schema: type[BaseModel]

    def with_user_prompt(self, user_prompt: str) -> "LLMRequest":
        return replace(self, user_prompt=user_prompt)


class Usage(BaseModel):
    """Token usage. `output_tokens` includes reasoning/thinking tokens, since both are billed
    at the output rate."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )

    @property
    def is_empty(self) -> bool:
        return self.input_tokens == 0 and self.output_tokens == 0


@dataclass(frozen=True)
class LLMResponse:
    output: BaseModel
    usage: Usage
    provider: str
    model: str
    latency_s: float
    finish_reason: str | None = None


AttemptOutcome = Literal[
    "success",
    "transient_error",
    "rate_limited",
    "unavailable",
    "auth_error",
    "bad_request",
    "truncated",
    "invalid_output",
]


class AttemptRecord(BaseModel):
    provider: str
    model: str
    outcome: AttemptOutcome
    error: str | None = None
    latency_s: float = 0.0
    max_output_tokens: int | None = None
    usage: Usage = Field(default_factory=Usage)
    cost_usd: float | None = None


def sum_usage(attempts: list[AttemptRecord]) -> Usage:
    total = Usage()
    for attempt in attempts:
        total = total + attempt.usage
    return total


def sum_cost(attempts: list[AttemptRecord]) -> float | None:
    """Total cost, or None when any attempt that consumed tokens has no known price."""
    total = 0.0
    for attempt in attempts:
        if attempt.cost_usd is None:
            if not attempt.usage.is_empty:
                return None
            continue
        total += attempt.cost_usd
    return round(total, 8)


@dataclass
class LLMResult:
    output: BaseModel
    served_by: ModelRef
    latency_s: float
    attempts: list[AttemptRecord] = field(default_factory=list)

    @property
    def usage(self) -> Usage:
        return sum_usage(self.attempts)

    @property
    def cost_usd(self) -> float | None:
        return sum_cost(self.attempts)
