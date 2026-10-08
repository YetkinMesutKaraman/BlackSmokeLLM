"""Gemini adapter (native google-genai SDK, JSON-schema structured output)."""

from time import perf_counter
from typing import Any

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, ValidationError

from blacksmoke.llm import errors
from blacksmoke.llm.types import LLMRequest, LLMResponse, ModelTarget, Usage

THINKING_OPTIONS = ("thinking_budget", "thinking_level", "include_thoughts")


class GeminiProvider:
    name = "google"

    def __init__(self, client: genai.Client) -> None:
        self._client = client

    @classmethod
    def create(cls, api_key: str) -> "GeminiProvider":
        return cls(genai.Client(api_key=api_key))

    async def generate(self, request: LLMRequest, target: ModelTarget) -> LLMResponse:
        started = perf_counter()
        try:
            response = await self._client.aio.models.generate_content(
                model=target.model,
                contents=request.user_prompt,
                config=_build_config(request, target),
            )
        except genai_errors.APIError as exc:
            raise _map_api_error(exc) from exc
        except httpx.TransportError as exc:
            raise errors.TransientError(f"Gemini transport error: {exc}") from exc

        usage = _usage(response.usage_metadata)
        finish_reason = _finish_reason(response)
        if finish_reason == types.FinishReason.MAX_TOKENS:
            raise errors.OutputTruncatedError(
                f"Output exceeded max_output_tokens={target.params.max_output_tokens}",
                usage=usage,
            )

        output = _validate(request.output_schema, response.text, usage, finish_reason)
        return LLMResponse(
            output=output,
            usage=usage,
            provider=self.name,
            model=target.model,
            latency_s=round(perf_counter() - started, 4),
            finish_reason=str(finish_reason.value) if finish_reason else None,
        )

    async def aclose(self) -> None:
        await self._client.aio.aclose()


def _build_config(request: LLMRequest, target: ModelTarget) -> types.GenerateContentConfig:
    params = target.params
    options = dict(params.provider_options)
    thinking = {key: options.pop(key) for key in THINKING_OPTIONS if key in options}
    config: dict[str, Any] = {
        "system_instruction": request.system_prompt,
        "max_output_tokens": params.max_output_tokens,
        "response_mime_type": "application/json",
        "response_json_schema": request.output_schema.model_json_schema(),
        "http_options": types.HttpOptions(timeout=int(params.timeout_s * 1000)),
        "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
    }
    if params.temperature is not None:
        config["temperature"] = params.temperature
    if params.top_p is not None:
        config["top_p"] = params.top_p
    if thinking:
        config["thinking_config"] = types.ThinkingConfig(**thinking)
    config.update(options)
    return types.GenerateContentConfig(**config)


def _usage(meta: Any) -> Usage:
    if meta is None:
        return Usage()
    thoughts = meta.thoughts_token_count or 0
    return Usage(
        input_tokens=meta.prompt_token_count or 0,
        output_tokens=(meta.candidates_token_count or 0) + thoughts,
        cached_input_tokens=meta.cached_content_token_count or 0,
        reasoning_tokens=thoughts,
    )


def _finish_reason(response: types.GenerateContentResponse) -> types.FinishReason | None:
    if not response.candidates:
        return None
    return response.candidates[0].finish_reason


def _validate(
    schema: type[BaseModel],
    text: str | None,
    usage: Usage,
    finish_reason: types.FinishReason | None,
) -> BaseModel:
    if not text:
        raise errors.OutputValidationError(
            f"Empty response text (finish_reason={finish_reason})", usage=usage
        )
    try:
        return schema.model_validate_json(text)
    except ValidationError as exc:
        raise errors.OutputValidationError(str(exc), usage=usage) from exc


def _map_api_error(exc: genai_errors.APIError) -> errors.LLMError:
    message = f"Gemini error {exc.code}: {exc.message or exc}"
    if exc.code == 429:
        return errors.RateLimitedError(message)
    if exc.code in (401, 403):
        return errors.AuthError(message)
    if exc.code == 404:
        return errors.ProviderUnavailableError(message)
    if exc.code == 408 or isinstance(exc, genai_errors.ServerError):
        return errors.TransientError(message)
    return errors.BadRequestError(message)
