"""OpenAI adapter (Responses API, strict JSON-schema structured output)."""

from time import perf_counter
from typing import Any

import openai
from openai import AsyncOpenAI
from openai.lib._parsing._responses import type_to_text_format_param
from pydantic import BaseModel, ValidationError

from blacksmoke.llm import errors
from blacksmoke.llm.types import LLMRequest, LLMResponse, ModelTarget, Usage


class OpenAIProvider:
    name = "openai"

    def __init__(self, client: AsyncOpenAI) -> None:
        self._client = client

    @classmethod
    def create(cls, api_key: str, organization: str | None = None) -> "OpenAIProvider":
        return cls(AsyncOpenAI(api_key=api_key, organization=organization, max_retries=0))

    async def generate(self, request: LLMRequest, target: ModelTarget) -> LLMResponse:
        started = perf_counter()
        try:
            response = await self._client.responses.create(**_build_kwargs(request, target))
        except openai.APIError as exc:
            raise _map_error(exc) from exc

        usage = _usage(response.usage)
        if response.status == "incomplete":
            reason = getattr(response.incomplete_details, "reason", None)
            if reason == "max_output_tokens":
                raise errors.OutputTruncatedError(
                    f"Output exceeded max_output_tokens={target.params.max_output_tokens}",
                    usage=usage,
                )
            raise errors.OutputValidationError(f"Incomplete response: {reason}", usage=usage)

        output = _validate(request.output_schema, response.output_text, usage)
        return LLMResponse(
            output=output,
            usage=usage,
            provider=self.name,
            model=target.model,
            latency_s=round(perf_counter() - started, 4),
            finish_reason=response.status,
        )

    async def aclose(self) -> None:
        await self._client.close()


def _build_kwargs(request: LLMRequest, target: ModelTarget) -> dict[str, Any]:
    params = target.params
    kwargs: dict[str, Any] = {
        "model": target.model,
        "input": [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_prompt},
        ],
        "text": {"format": type_to_text_format_param(request.output_schema)},
        "max_output_tokens": params.max_output_tokens,
        "timeout": params.timeout_s,
    }
    if params.temperature is not None:
        kwargs["temperature"] = params.temperature
    if params.top_p is not None:
        kwargs["top_p"] = params.top_p
    kwargs.update(params.provider_options)
    return kwargs


def _usage(raw: Any) -> Usage:
    if raw is None:
        return Usage()
    cached = getattr(getattr(raw, "input_tokens_details", None), "cached_tokens", 0) or 0
    reasoning = getattr(getattr(raw, "output_tokens_details", None), "reasoning_tokens", 0) or 0
    return Usage(
        input_tokens=raw.input_tokens or 0,
        output_tokens=raw.output_tokens or 0,
        cached_input_tokens=cached,
        reasoning_tokens=reasoning,
    )


def _validate(schema: type[BaseModel], text: str | None, usage: Usage) -> BaseModel:
    if not text:
        raise errors.OutputValidationError("Empty response text", usage=usage)
    try:
        return schema.model_validate_json(text)
    except ValidationError as exc:
        raise errors.OutputValidationError(str(exc), usage=usage) from exc


def _map_error(exc: openai.APIError) -> errors.LLMError:
    message = f"OpenAI error: {exc}"
    if isinstance(exc, openai.RateLimitError):
        return errors.RateLimitedError(message)
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
        return errors.TransientError(message)
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return errors.AuthError(message)
    if isinstance(exc, openai.NotFoundError):
        return errors.ProviderUnavailableError(message)
    if isinstance(exc, openai.APIStatusError):
        if exc.status_code >= 500 or exc.status_code in (408, 409):
            return errors.TransientError(message)
        return errors.BadRequestError(message)
    return errors.TransientError(message)
