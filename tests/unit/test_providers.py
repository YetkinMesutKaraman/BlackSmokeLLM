from types import SimpleNamespace

import httpx
import openai
import pytest
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel

from blacksmoke.llm import errors
from blacksmoke.llm.providers.gemini_provider import GeminiProvider
from blacksmoke.llm.providers.openai_provider import OpenAIProvider
from blacksmoke.llm.types import GenerationParams, LLMRequest, ModelTarget, Usage


class Items(BaseModel):
    items: list[str]


REQUEST = LLMRequest(system_prompt="sys", user_prompt="user", output_schema=Items)
HTTP_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/responses")


class StubCall:
    def __init__(self, result):
        self.result = result
        self.kwargs = None

    async def __call__(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


# ----- OpenAI ---------------------------------------------------------------------------------


def openai_response(status="completed", text='{"items": ["a"]}', reason=None):
    return SimpleNamespace(
        status=status,
        incomplete_details=SimpleNamespace(reason=reason) if reason else None,
        output_text=text,
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            input_tokens_details=SimpleNamespace(cached_tokens=4),
            output_tokens_details=SimpleNamespace(reasoning_tokens=2),
        ),
    )


def openai_provider(result) -> tuple[OpenAIProvider, StubCall]:
    call = StubCall(result)
    return OpenAIProvider(SimpleNamespace(responses=SimpleNamespace(create=call))), call


async def test_openai_success_parses_output_and_usage():
    provider, call = openai_provider(openai_response())
    target = ModelTarget(
        provider="openai",
        model="gpt-4o-mini",
        params=GenerationParams(top_p=0.9, provider_options={"service_tier": "flex"}),
    )
    response = await provider.generate(REQUEST, target)
    assert response.output == Items(items=["a"])
    assert response.usage == Usage(
        input_tokens=10, output_tokens=5, cached_input_tokens=4, reasoning_tokens=2
    )
    assert "temperature" not in call.kwargs
    assert call.kwargs["top_p"] == 0.9
    assert call.kwargs["service_tier"] == "flex"
    assert call.kwargs["text"]["format"]["strict"] is True
    assert call.kwargs["input"][0] == {"role": "system", "content": "sys"}


async def test_openai_incomplete_max_tokens_is_truncation():
    provider, _ = openai_provider(openai_response("incomplete", '{"items": [', "max_output_tokens"))
    with pytest.raises(errors.OutputTruncatedError) as info:
        await provider.generate(REQUEST, ModelTarget(provider="openai", model="m"))
    assert info.value.usage.input_tokens == 10


async def test_openai_schema_mismatch_is_validation_error():
    provider, _ = openai_provider(openai_response(text='{"wrong": 1}'))
    with pytest.raises(errors.OutputValidationError):
        await provider.generate(REQUEST, ModelTarget(provider="openai", model="m"))


@pytest.mark.parametrize(
    ("sdk_error", "expected"),
    [
        (
            openai.RateLimitError(
                "x", response=httpx.Response(429, request=HTTP_REQUEST), body=None
            ),
            errors.RateLimitedError,
        ),
        (
            openai.AuthenticationError(
                "x", response=httpx.Response(401, request=HTTP_REQUEST), body=None
            ),
            errors.AuthError,
        ),
        (
            openai.BadRequestError(
                "x", response=httpx.Response(400, request=HTTP_REQUEST), body=None
            ),
            errors.BadRequestError,
        ),
        (
            openai.NotFoundError(
                "x", response=httpx.Response(404, request=HTTP_REQUEST), body=None
            ),
            errors.ProviderUnavailableError,
        ),
        (
            openai.InternalServerError(
                "x", response=httpx.Response(500, request=HTTP_REQUEST), body=None
            ),
            errors.TransientError,
        ),
        (openai.APITimeoutError(request=HTTP_REQUEST), errors.TransientError),
    ],
)
async def test_openai_error_mapping(sdk_error, expected):
    provider, _ = openai_provider(sdk_error)
    with pytest.raises(expected):
        await provider.generate(REQUEST, ModelTarget(provider="openai", model="m"))


# ----- Gemini ---------------------------------------------------------------------------------


def gemini_response(finish=types.FinishReason.STOP, text='{"items": []}'):
    return SimpleNamespace(
        candidates=[SimpleNamespace(finish_reason=finish)],
        text=text,
        usage_metadata=SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=3,
            thoughts_token_count=2,
            cached_content_token_count=None,
        ),
    )


def gemini_provider(result) -> tuple[GeminiProvider, StubCall]:
    call = StubCall(result)
    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=call)))
    return GeminiProvider(client), call


async def test_gemini_success_counts_thinking_as_output():
    provider, call = gemini_provider(gemini_response())
    target = ModelTarget(
        provider="google",
        model="gemini-2.5-flash",
        params=GenerationParams(temperature=0.0, provider_options={"thinking_budget": 0}),
    )
    response = await provider.generate(REQUEST, target)
    assert response.output == Items(items=[])
    assert response.usage == Usage(input_tokens=10, output_tokens=5, reasoning_tokens=2)
    config = call.kwargs["config"]
    assert config.system_instruction == "sys"
    assert config.temperature == 0.0
    assert config.thinking_config.thinking_budget == 0
    assert config.response_json_schema["required"] == ["items"]
    assert config.http_options.timeout == 60_000


async def test_gemini_max_tokens_is_truncation():
    provider, _ = gemini_provider(gemini_response(types.FinishReason.MAX_TOKENS, '{"items": ['))
    with pytest.raises(errors.OutputTruncatedError):
        await provider.generate(REQUEST, ModelTarget(provider="google", model="g"))


async def test_gemini_empty_text_is_validation_error():
    provider, _ = gemini_provider(gemini_response(types.FinishReason.SAFETY, None))
    with pytest.raises(errors.OutputValidationError):
        await provider.generate(REQUEST, ModelTarget(provider="google", model="g"))


@pytest.mark.parametrize(
    ("sdk_error", "expected"),
    [
        (genai_errors.ClientError(429, {"error": {"message": "quota"}}), errors.RateLimitedError),
        (genai_errors.ClientError(403, {"error": {"message": "denied"}}), errors.AuthError),
        (genai_errors.ClientError(400, {"error": {"message": "bad"}}), errors.BadRequestError),
        (
            genai_errors.ClientError(404, {"error": {"message": "retired"}}),
            errors.ProviderUnavailableError,
        ),
        (genai_errors.ServerError(503, {"error": {"message": "busy"}}), errors.TransientError),
        (httpx.ConnectError("refused"), errors.TransientError),
    ],
)
async def test_gemini_error_mapping(sdk_error, expected):
    provider, _ = gemini_provider(sdk_error)
    with pytest.raises(expected):
        await provider.generate(REQUEST, ModelTarget(provider="google", model="g"))
