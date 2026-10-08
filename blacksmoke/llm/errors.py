"""Error taxonomy for LLM calls. The class of an error decides what the resilience layer does:

- TransientError (incl. RateLimitedError): retried on the same target, then fall back.
- OutputValidationError / OutputTruncatedError: repaired on the same target, then fall back.
- AuthError / ProviderUnavailableError: fall back immediately.
- BadRequestError: fail fast, the request itself is wrong.
"""

from typing import ClassVar

from blacksmoke.core.errors import BlackSmokeError
from blacksmoke.llm.types import AttemptOutcome, AttemptRecord, Usage


class LLMError(BlackSmokeError):
    outcome: ClassVar[AttemptOutcome]

    def __init__(self, message: str, *, usage: Usage | None = None) -> None:
        super().__init__(message)
        self.usage = usage or Usage()


class TransientError(LLMError):
    outcome = "transient_error"


class RateLimitedError(TransientError):
    outcome = "rate_limited"


class ProviderUnavailableError(LLMError):
    """The provider is unknown or not configured (e.g. missing API key)."""

    outcome = "unavailable"


class AuthError(LLMError):
    outcome = "auth_error"


class BadRequestError(LLMError):
    outcome = "bad_request"


class OutputTruncatedError(LLMError):
    outcome = "truncated"


class OutputValidationError(LLMError):
    outcome = "invalid_output"


class AllTargetsFailedError(BlackSmokeError):
    def __init__(self, message: str, attempts: list[AttemptRecord]) -> None:
        super().__init__(message)
        self.attempts = attempts
