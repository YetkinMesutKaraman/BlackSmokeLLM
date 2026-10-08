import logging

from pydantic import BaseModel, Field

from blacksmoke.llm.types import Usage

logger = logging.getLogger(__name__)

TOKENS_PER_PRICE_UNIT = 1_000_000


class ModelPricing(BaseModel):
    """USD per 1M tokens."""

    input: float
    output: float
    cached_input: float | None = None


class ProviderModels(BaseModel):
    models: dict[str, ModelPricing] = Field(default_factory=dict)


class ModelsCatalog(BaseModel):
    providers: dict[str, ProviderModels] = Field(default_factory=dict)


class PricingCatalog:
    def __init__(self, catalog: ModelsCatalog) -> None:
        self._catalog = catalog
        self._warned: set[tuple[str, str]] = set()

    def has_model(self, provider: str, model: str) -> bool:
        return self._lookup(provider, model) is not None

    def models(self, provider: str) -> list[str]:
        entry = self._catalog.providers.get(provider)
        return sorted(entry.models) if entry else []

    def cost(self, provider: str, model: str, usage: Usage) -> float | None:
        pricing = self._lookup(provider, model)
        if pricing is None:
            if (provider, model) not in self._warned:
                self._warned.add((provider, model))
                logger.warning(
                    "No pricing for %s/%s; cost will be reported as null", provider, model
                )
            return None

        cached = min(usage.cached_input_tokens, usage.input_tokens)
        uncached = usage.input_tokens - cached
        cached_rate = pricing.cached_input if pricing.cached_input is not None else pricing.input
        total = (
            uncached * pricing.input + cached * cached_rate + usage.output_tokens * pricing.output
        ) / TOKENS_PER_PRICE_UNIT
        return round(total, 8)

    def _lookup(self, provider: str, model: str) -> ModelPricing | None:
        entry = self._catalog.providers.get(provider)
        return entry.models.get(model) if entry else None
