import pytest
from pydantic import BaseModel

from blacksmoke.llm import errors
from blacksmoke.llm.factory import ProviderFactory
from blacksmoke.llm.gateway import LLMGateway
from blacksmoke.llm.pricing import ModelsCatalog, PricingCatalog
from blacksmoke.llm.resilience import ResiliencePolicy
from blacksmoke.llm.types import GenerationParams, LLMRequest, ModelTarget, Usage
from tests.fakes import FakeProvider


class Items(BaseModel):
    items: list[str]


OK = {"items": ["a"]}
REQUEST = LLMRequest(system_prompt="sys", user_prompt="original prompt", output_schema=Items)
POLICY = ResiliencePolicy(
    transient_attempts=3,
    backoff_initial_s=0,
    backoff_max_s=0,
    repair_attempts=2,
    truncation_multiplier=1.5,
    max_output_tokens_cap=200,
)
PRIMARY = ModelTarget(
    provider="openai", model="gpt-a", params=GenerationParams(max_output_tokens=100)
)
FALLBACK = ModelTarget(provider="google", model="gem-b")
ROUTE = [PRIMARY, FALLBACK]
PRICING = PricingCatalog(
    ModelsCatalog.model_validate(
        {
            "providers": {
                "openai": {"models": {"gpt-a": {"input": 1.0, "output": 2.0}}},
                "google": {"models": {"gem-b": {"input": 0.5, "output": 1.0}}},
            }
        }
    )
)


def make_gateway(**providers: FakeProvider) -> LLMGateway:
    factory = ProviderFactory({name: (lambda p=p: p) for name, p in providers.items()})
    return LLMGateway(factory, PRICING)


def outcomes(result_or_error) -> list[tuple[str, str]]:
    return [(a.provider, a.outcome) for a in result_or_error.attempts]


async def test_primary_success_records_usage_and_cost():
    openai = FakeProvider("openai", default=OK)
    result = await make_gateway(openai=openai, google=FakeProvider("google")).generate(
        REQUEST, ROUTE, POLICY
    )
    assert result.output == Items(items=["a"])
    assert str(result.served_by) == "openai/gpt-a"
    assert outcomes(result) == [("openai", "success")]
    assert result.usage == Usage(input_tokens=1000, output_tokens=100)
    assert result.cost_usd == pytest.approx((1000 * 1.0 + 100 * 2.0) / 1_000_000)


async def test_transient_error_is_retried_on_same_target():
    openai = FakeProvider("openai", script=[errors.TransientError("timeout")], default=OK)
    result = await make_gateway(openai=openai, google=FakeProvider("google")).generate(
        REQUEST, ROUTE, POLICY
    )
    assert outcomes(result) == [("openai", "transient_error"), ("openai", "success")]


async def test_rate_limit_exhausted_falls_back():
    openai = FakeProvider("openai", default=errors.RateLimitedError("429"))
    google = FakeProvider("google", default=OK)
    result = await make_gateway(openai=openai, google=google).generate(REQUEST, ROUTE, POLICY)
    assert str(result.served_by) == "google/gem-b"
    assert outcomes(result) == [("openai", "rate_limited")] * 3 + [("google", "success")]


async def test_invalid_output_is_repaired_with_error_feedback_that_does_not_accumulate():
    openai = FakeProvider(
        "openai",
        script=[
            errors.OutputValidationError("bad json 1"),
            errors.OutputValidationError("bad json 2"),
        ],
        default=OK,
    )
    result = await make_gateway(openai=openai, google=FakeProvider("google")).generate(
        REQUEST, ROUTE, POLICY
    )
    assert outcomes(result)[-1] == ("openai", "success")
    prompts = [request.user_prompt for request, _ in openai.calls]
    assert prompts[0] == "original prompt"
    assert "bad json 1" in prompts[1]
    assert "bad json 2" in prompts[2] and "bad json 1" not in prompts[2]
    assert all(p.startswith("original prompt") and p.count("IMPORTANT") <= 1 for p in prompts)


async def test_truncated_output_raises_token_budget_up_to_cap():
    openai = FakeProvider("openai", default=errors.OutputTruncatedError("too long"))
    google = FakeProvider("google", default=OK)
    result = await make_gateway(openai=openai, google=google).generate(REQUEST, ROUTE, POLICY)
    budgets = [target.params.max_output_tokens for _, target in openai.calls]
    assert budgets == [100, 150, 200]
    assert str(result.served_by) == "google/gem-b"


async def test_auth_error_falls_back_without_retry():
    openai = FakeProvider("openai", default=errors.AuthError("bad key"))
    google = FakeProvider("google", default=OK)
    result = await make_gateway(openai=openai, google=google).generate(REQUEST, ROUTE, POLICY)
    assert outcomes(result) == [("openai", "auth_error"), ("google", "success")]


async def test_bad_request_fails_fast_without_fallback():
    openai = FakeProvider("openai", default=errors.BadRequestError("invalid param"))
    google = FakeProvider("google", default=OK)
    with pytest.raises(errors.AllTargetsFailedError) as info:
        await make_gateway(openai=openai, google=google).generate(REQUEST, ROUTE, POLICY)
    assert outcomes(info.value) == [("openai", "bad_request")]
    assert google.calls == []


async def test_all_targets_failing_reports_every_attempt_and_its_cost():
    usage = Usage(input_tokens=10, output_tokens=5)
    openai = FakeProvider("openai", default=errors.OutputValidationError("bad", usage=usage))
    google = FakeProvider("google", default=errors.ProviderUnavailableError("down"))
    with pytest.raises(errors.AllTargetsFailedError) as info:
        await make_gateway(openai=openai, google=google).generate(REQUEST, ROUTE, POLICY)
    attempts = info.value.attempts
    assert [a.outcome for a in attempts] == ["invalid_output"] * 3 + ["unavailable"]
    assert attempts[0].cost_usd == pytest.approx((10 * 1.0 + 5 * 2.0) / 1_000_000)


async def test_unconfigured_provider_is_skipped():
    def missing_key():
        raise errors.ProviderUnavailableError("OPENAI_API_KEY is not set")

    google = FakeProvider("google", default=OK)
    gateway = LLMGateway(
        ProviderFactory({"openai": missing_key, "google": lambda: google}), PRICING
    )
    result = await gateway.generate(REQUEST, ROUTE, POLICY)
    assert outcomes(result) == [("openai", "unavailable"), ("google", "success")]


async def test_unknown_model_price_reports_null_cost():
    openai = FakeProvider("openai", default=OK)
    target = ModelTarget(provider="openai", model="unpriced")
    result = await make_gateway(openai=openai).generate(REQUEST, [target], POLICY)
    assert result.cost_usd is None
