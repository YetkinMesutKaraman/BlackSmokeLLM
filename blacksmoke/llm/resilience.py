"""Resilience wrappers. Each wrapper implements the `LLMProvider` port and decorates an inner
provider, so the gateway composes them per target:

    StructuredRepair(RetryingProvider(RecordingProvider(adapter)))
"""

import logging
from time import perf_counter

from pydantic import BaseModel, ConfigDict, Field
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt
from tenacity.wait import wait_exponential_jitter

from blacksmoke.llm.errors import (
    LLMError,
    OutputTruncatedError,
    OutputValidationError,
    TransientError,
)
from blacksmoke.llm.pricing import PricingCatalog
from blacksmoke.llm.provider import LLMProvider
from blacksmoke.llm.types import (
    AttemptOutcome,
    AttemptRecord,
    LLMRequest,
    LLMResponse,
    ModelTarget,
    Usage,
)

logger = logging.getLogger(__name__)

MAX_FEEDBACK_CHARS = 1500


class ResiliencePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transient_attempts: int = Field(3, ge=1, description="Calls per target for transient errors.")
    backoff_initial_s: float = Field(1.0, ge=0)
    backoff_max_s: float = Field(20.0, ge=0)
    repair_attempts: int = Field(
        2, ge=0, description="Re-asks per target for bad/truncated output."
    )
    truncation_multiplier: float = Field(1.5, gt=1)
    max_output_tokens_cap: int = Field(4000, ge=1)


class RecordingProvider:
    """Innermost wrapper: records one `AttemptRecord` per physical call, priced against the
    model that actually ran."""

    def __init__(
        self, inner: LLMProvider, pricing: PricingCatalog, sink: list[AttemptRecord]
    ) -> None:
        self.name = inner.name
        self._inner = inner
        self._pricing = pricing
        self._sink = sink

    async def generate(self, request: LLMRequest, target: ModelTarget) -> LLMResponse:
        started = perf_counter()
        try:
            response = await self._inner.generate(request, target)
        except LLMError as exc:
            self._sink.append(self._record(target, exc.outcome, started, exc.usage, str(exc)))
            raise
        self._sink.append(self._record(target, "success", started, response.usage, None))
        return response

    def _record(
        self,
        target: ModelTarget,
        outcome: AttemptOutcome,
        started: float,
        usage: Usage,
        error: str | None,
    ) -> AttemptRecord:
        return AttemptRecord(
            provider=target.provider,
            model=target.model,
            outcome=outcome,
            error=error,
            latency_s=round(perf_counter() - started, 4),
            max_output_tokens=target.params.max_output_tokens,
            usage=usage,
            cost_usd=self._pricing.cost(target.provider, target.model, usage),
        )

    async def aclose(self) -> None:
        return None


class RetryingProvider:
    """Retries transient failures (timeouts, 5xx, rate limits) with jittered backoff."""

    def __init__(self, inner: LLMProvider, policy: ResiliencePolicy) -> None:
        self.name = inner.name
        self._inner = inner
        self._policy = policy

    async def generate(self, request: LLMRequest, target: ModelTarget) -> LLMResponse:
        retrying = AsyncRetrying(
            retry=retry_if_exception_type(TransientError),
            stop=stop_after_attempt(self._policy.transient_attempts),
            wait=wait_exponential_jitter(
                initial=self._policy.backoff_initial_s, max=self._policy.backoff_max_s
            ),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                return await self._inner.generate(request, target)
        raise AssertionError("unreachable")

    async def aclose(self) -> None:
        return None


class StructuredRepair:
    """Re-asks when the output fails schema validation (feeding the error back) and raises the
    output token budget when the output was truncated."""

    def __init__(self, inner: LLMProvider, policy: ResiliencePolicy) -> None:
        self.name = inner.name
        self._inner = inner
        self._policy = policy

    async def generate(self, request: LLMRequest, target: ModelTarget) -> LLMResponse:
        current_request, current_target = request, target
        repairs = 0
        while True:
            try:
                return await self._inner.generate(current_request, current_target)
            except OutputValidationError as exc:
                if repairs >= self._policy.repair_attempts:
                    raise
                repairs += 1
                logger.info("Repairing invalid output from %s (repair %d)", target.ref, repairs)
                current_request = request.with_user_prompt(
                    build_repair_prompt(request.user_prompt, str(exc))
                )
            except OutputTruncatedError:
                current_max = current_target.params.max_output_tokens
                new_max = min(
                    int(current_max * self._policy.truncation_multiplier),
                    self._policy.max_output_tokens_cap,
                )
                if repairs >= self._policy.repair_attempts or new_max <= current_max:
                    raise
                repairs += 1
                logger.info(
                    "Output truncated on %s; raising max_output_tokens %d -> %d",
                    target.ref,
                    current_max,
                    new_max,
                )
                current_target = current_target.with_max_output_tokens(new_max)

    async def aclose(self) -> None:
        return None


def build_repair_prompt(original_user_prompt: str, error: str) -> str:
    """Always derived from the original prompt, so feedback never accumulates across repairs."""
    return (
        f"{original_user_prompt}\n\n"
        "IMPORTANT: Your previous response did not match the required JSON schema.\n"
        f"Validation error: {error[:MAX_FEEDBACK_CHARS]}\n"
        "Return only a JSON object that exactly matches the schema."
    )
