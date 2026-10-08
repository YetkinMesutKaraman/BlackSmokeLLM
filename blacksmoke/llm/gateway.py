import logging
from collections.abc import Sequence
from time import perf_counter

from blacksmoke.llm.errors import (
    AllTargetsFailedError,
    BadRequestError,
    LLMError,
    ProviderUnavailableError,
)
from blacksmoke.llm.factory import ProviderFactory
from blacksmoke.llm.pricing import PricingCatalog
from blacksmoke.llm.resilience import (
    RecordingProvider,
    ResiliencePolicy,
    RetryingProvider,
    StructuredRepair,
)
from blacksmoke.llm.types import AttemptRecord, LLMRequest, LLMResult, ModelTarget

logger = logging.getLogger(__name__)


class LLMGateway:
    """Single entry point for model calls: walks a route (primary target, then fallbacks) and
    applies retry and repair on each target."""

    def __init__(self, factory: ProviderFactory, pricing: PricingCatalog) -> None:
        self._factory = factory
        self._pricing = pricing

    async def generate(
        self, request: LLMRequest, route: Sequence[ModelTarget], policy: ResiliencePolicy
    ) -> LLMResult:
        if not route:
            raise ValueError("route must contain at least one target")

        attempts: list[AttemptRecord] = []
        started = perf_counter()
        for index, target in enumerate(route):
            try:
                provider = self._factory.get(target.provider)
            except ProviderUnavailableError as exc:
                attempts.append(
                    AttemptRecord(
                        provider=target.provider,
                        model=target.model,
                        outcome=exc.outcome,
                        error=str(exc),
                    )
                )
                logger.warning("Skipping %s: %s", target.ref, exc)
                continue

            chain = StructuredRepair(
                RetryingProvider(RecordingProvider(provider, self._pricing, attempts), policy),
                policy,
            )
            try:
                response = await chain.generate(request, target)
            except BadRequestError as exc:
                raise AllTargetsFailedError(
                    f"{target.ref} rejected the request; not trying fallbacks: {exc}", attempts
                ) from exc
            except LLMError as exc:
                has_next = index < len(route) - 1
                logger.warning(
                    "%s failed (%s)%s",
                    target.ref,
                    exc.outcome,
                    "; falling back" if has_next else "",
                )
                continue

            return LLMResult(
                output=response.output,
                served_by=target.ref,
                latency_s=round(perf_counter() - started, 4),
                attempts=attempts,
            )

        tried = ", ".join(str(t.ref) for t in route)
        raise AllTargetsFailedError(f"All targets failed: {tried}", attempts)
