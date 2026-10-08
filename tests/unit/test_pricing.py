import pytest

from blacksmoke.llm.pricing import ModelsCatalog, PricingCatalog
from blacksmoke.llm.types import AttemptRecord, Usage, sum_cost

CATALOG = PricingCatalog(
    ModelsCatalog.model_validate(
        {
            "providers": {
                "openai": {
                    "models": {
                        "m": {"input": 2.0, "cached_input": 0.5, "output": 8.0},
                        "no-cache": {"input": 1.0, "output": 1.0},
                    }
                }
            }
        }
    )
)


def test_cost_prices_cached_input_separately():
    usage = Usage(input_tokens=1_000_000, cached_input_tokens=400_000, output_tokens=100_000)
    assert CATALOG.cost("openai", "m", usage) == pytest.approx(0.6 * 2.0 + 0.4 * 0.5 + 0.1 * 8.0)


def test_cached_input_falls_back_to_input_rate():
    usage = Usage(input_tokens=1_000_000, cached_input_tokens=1_000_000)
    assert CATALOG.cost("openai", "no-cache", usage) == pytest.approx(1.0)


def test_unknown_model_has_no_cost():
    assert CATALOG.cost("openai", "missing", Usage(input_tokens=1)) is None
    assert not CATALOG.has_model("google", "m")


def test_sum_cost_is_null_when_a_billed_attempt_is_unpriced():
    priced = AttemptRecord(provider="p", model="m", outcome="success", cost_usd=0.1)
    unpriced = AttemptRecord(
        provider="p", model="x", outcome="success", usage=Usage(input_tokens=5), cost_usd=None
    )
    unbilled = AttemptRecord(provider="p", model="x", outcome="unavailable")
    assert sum_cost([priced, unbilled]) == pytest.approx(0.1)
    assert sum_cost([priced, unpriced]) is None
