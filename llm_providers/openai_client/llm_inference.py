import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import openai
from pydantic import BaseModel, ValidationError

# retry logic
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from configs.llm_task_config import MODEL_CHOICES

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Configure logging
logger = logging.getLogger(__name__)


# Constants
MAX_RETRIES = 5
MAX_OUTPUT_TOKEN_MULTIPLIER = 1.2


@dataclass
class APICallCounters:
    """Data class to track API call attempts."""

    total_attempts: int = 0
    validation_retry_attempts: int = 0

    def increment_total(self) -> None:
        self.total_attempts += 1

    def increment_validation(self) -> None:
        self.validation_retry_attempts += 1

    def to_tuple(self) -> Tuple[int, int]:
        return (self.total_attempts, self.validation_retry_attempts)


@dataclass
class TrackTokenCounts:
    """Data class to track token counts."""

    input_tokens: int = 0
    output_tokens: int = 0

    def increment_input(self, count: int) -> None:
        self.input_tokens += count

    def increment_output(self, count: int) -> None:
        self.output_tokens += count

    def to_dict(self) -> Dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


@dataclass
class JsonValidationResult:
    """Result of LLM output validation."""

    is_valid: bool
    json_output: Optional[Dict[str, Any]]
    input_tokens: int
    output_tokens: int
    counters: APICallCounters


def retry_error_callback(retry_state: RetryCallState):
    """Callback function for retry errors."""
    exception = retry_state.outcome.exception()
    logger.error(f"Max retries ({MAX_RETRIES}) exceeded: {exception}")
    raise exception


def _handle_token_limit_exceeded(
    response: Optional[openai.types.responses.ParsedResponse],
    kwargs: Dict[str, Any],
    provider_name: str,
    validation_attempt_count: int,
) -> None:
    """Handle max output tokens exceeded scenario."""

    current_max_output_tokens = kwargs.get("max_output_tokens", 100)
    final_retry = validation_attempt_count >= (MAX_RETRIES - 1)

    if response is None:
        logger.warning(
            f"Response is None, it's likely due to max output tokens exceeded, ValidationError occurred. Output token limit: {current_max_output_tokens}"
        )

        # Increase token limit
        new_token_limit = int(current_max_output_tokens * MAX_OUTPUT_TOKEN_MULTIPLIER)
        kwargs["max_output_tokens"] = new_token_limit
        logger.info(
            f"Increasing token limit from {current_max_output_tokens} to {new_token_limit}"
        )

        # Switch model if available
        model_names = MODEL_CHOICES.get(provider_name, {}).get("llm", [])
        if len(model_names) >= 2:
            current_model = kwargs["model"]
            # Switch to the other model for next retry, on final retry, use the largest model
            if current_model == model_names[0] and not final_retry:
                logger.info(f"Switching model from {current_model} to {model_names[1]}")
                kwargs["model"] = model_names[1]
            elif final_retry:
                logger.info(f"Final retry with largest model: {model_names[-1]}")
                kwargs["model"] = model_names[-1]
    elif (
        response.status == "incomplete"
        and response.incomplete_details.reason == "max_output_tokens"
    ):
        logger.warning(f"Max output tokens exceeded: {current_max_output_tokens}")


def _build_retry_prompt(
    initial_prompt: str, error: ValidationError, validation_attempt_count: int
) -> str:
    """Build the retry prompt with error information."""
    remaining_attempts = MAX_RETRIES - validation_attempt_count
    return (
        f"{initial_prompt}\n\n"
        f"Previous attempt resulted in Pydantic Validation Error: {error}. "
        f"Retry attempt: {validation_attempt_count}, remaining attempts: {remaining_attempts}. "
        f"Please ensure the response follows the exact schema format."
    )


def _extract_initial_prompt(kwargs: Dict[str, Any], counters: APICallCounters) -> str:
    """Extract and clean the initial user prompt."""
    prompt = kwargs["input"][1]["content"]

    # Clean previous retry messages if this is a retry
    if counters.total_attempts > 0 and counters.validation_retry_attempts > 0:
        prompt = prompt.split("\n\n Previous attempt")[0]

    return prompt


@retry(
    retry=retry_if_exception_type(
        (
            openai.InternalServerError,
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
        )
    ),
    wait=wait_exponential(multiplier=2, min=4, max=30) + wait_random(0, 2),
    stop=stop_after_attempt(MAX_RETRIES),
    retry_error_callback=retry_error_callback,
)
def api_responses_with_backoff(
    client: openai.OpenAI,
    provider_name: str,
    api_call_counters: tuple,
    error_message_prompt: str | ValidationError,
    **kwargs,
) -> tuple[tuple[int, int], Optional[openai.types.responses.ParsedResponse], dict]:
    """
    Make OpenAI API call with retry logic and error handling.

    Args:
        client: OpenAI client instance
        provider_name: Name of the provider
        api_call_counters: Tuple of (total_attempts, validation_retry_attempts)
        error_message_prompt: Error message or ValidationError from previous attempt
        **kwargs: Additional arguments for the API call

    Returns:
        Tuple of updated counters, response, and kwargs
    """
    counters = APICallCounters(*api_call_counters)
    initial_prompt = _extract_initial_prompt(kwargs, counters)

    logger.info("Sending request to OpenAI API")

    response = None

    try:
        response = client.responses.parse(**kwargs)
        if response:
            logger.debug(f"Response received: {response.output_text}...")
    except ValidationError as e:
        logger.warning(f"Validation error during API call: {e}")
        error_message_prompt = e
    except Exception as e:
        logger.error(f"Unexpected error during API call: {e}")
        raise

    # Handle validation errors and token limits
    if isinstance(error_message_prompt, ValidationError):
        counters.increment_validation()

        modified_prompt = _build_retry_prompt(
            initial_prompt, error_message_prompt, counters.validation_retry_attempts
        )

        _handle_token_limit_exceeded(
            response, kwargs, provider_name, counters.validation_retry_attempts
        )
    else:
        modified_prompt = initial_prompt

    # Update prompt and counters
    kwargs["input"][1]["content"] = modified_prompt
    counters.increment_total()

    return counters.to_tuple(), response, kwargs


def _validate_single_response(
    response: Optional[openai.types.responses.ParsedResponse],
    schema: BaseModel,
    counters: APICallCounters,
) -> JsonValidationResult:
    """
    Validate a single response against the schema.

    Returns:
        JsonValidationResult

    """

    if response is None:
        logger.error("Response is None, cannot be validated")
        return JsonValidationResult(
            is_valid=False,
            json_output=None,
            input_tokens=0,
            output_tokens=0,
            counters=counters,
        )

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    try:
        validated_output = schema.model_validate_json(response.output_text).model_dump()
        logger.info("JSON output is valid according to the schema")

        return JsonValidationResult(
            is_valid=True,
            json_output=validated_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            counters=counters,
        )

    except ValidationError as e:
        logger.warning(f"Validation failed: {e}")
        return JsonValidationResult(
            is_valid=False,
            json_output=None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            counters=counters,
        )
    except Exception as e:
        logger.error(f"Unexpected error during validation: {e}")
        return JsonValidationResult(
            is_valid=False,
            json_output=None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            counters=counters,
        )


def _calculate_total_input_tokens(total_count_of_api: int, input_tokens: int) -> int:
    """
    Calculate the total number of input tokens based on the number of API calls and input tokens per call.

    Args:
        total_count_of_api (int): The total number of API calls made.
        input_tokens (int): The number of input tokens per API call.

    Returns:
        int: The total number of input tokens.
    """
    if total_count_of_api <= 0 or input_tokens <= 0:
        return 0
    total_input_tokens = total_count_of_api * input_tokens
    return total_input_tokens


def _calculate_total_output_tokens(
    total_count_of_api: int,
    output_tokens: int,
    current_max_output_tokens: int,
) -> int:
    """
    Calculate the total number of output tokens based on the number of API calls and output tokens per call.

    Args:
        total_count_of_api (int): The total number of API calls made.
        output_tokens (int): The number of output tokens per API call.

    Returns:
        int: The total number of output tokens.
    """
    # increment output tokens backwards to the first attempt, considering the MAX_OUTPUT_TOKEN_MULTIPLIER
    if total_count_of_api <= 0 or output_tokens <= 0:
        return 0
    total_output_tokens = output_tokens
    for _ in range(total_count_of_api - 1):
        # inverse calculation to get the output tokens for each attempt
        current_max_output_tokens = (
            current_max_output_tokens / MAX_OUTPUT_TOKEN_MULTIPLIER
        )
        total_output_tokens += int(current_max_output_tokens)
    return total_output_tokens


def validate_llm_output(
    response: Optional[openai.types.responses.ParsedResponse],
    output_pydantic_json_schema: BaseModel,
    client: openai.OpenAI,
    provider_name: str,
    llm_input_args: dict,
    api_call_counters=(0, 0),
    error_message_prompt="",
) -> JsonValidationResult:
    """
    Validate LLM output against Pydantic schema with retry logic.

    Args:
        response: The response from the LLM API
        output_pydantic_json_schema: Pydantic schema for validation
        client: OpenAI client instance
        provider_name: Name of the provider
        llm_input_args: Arguments for LLM calls
        api_call_counters: Tuple of attempt counters
        error_message_prompt: Error message from previous attempts

    Returns:
        JsonValidationResult: Result of the validation process
    """
    counters = APICallCounters(*api_call_counters)

    # Try initial validation
    initial_validation_result = _validate_single_response(
        response, output_pydantic_json_schema, counters
    )
    is_valid = initial_validation_result.is_valid
    counters = initial_validation_result.counters

    if is_valid:
        logger.info("Initial validation successful")
        return initial_validation_result

    # Retry validation if initial attempt failed
    logger.info("Starting validation retry attempts")
    # Prepare error message for retry
    if isinstance(error_message_prompt, ValidationError):
        error_message = str(error_message_prompt)
    else:
        error_message = (
            "Initial validation failed. Retrying with modified prompt and parameters."
        )
    logger.warning(f"Validation error: {error_message}")
    # track token counts
    token_counts = TrackTokenCounts()

    for attempt in range(1, MAX_RETRIES + 1):
        logger.info(f"Validation retry attempt {attempt}/{MAX_RETRIES}")
        logger.debug(f"Model: {llm_input_args.get('model')}")
        logger.debug(f"Max output tokens: {llm_input_args.get('max_output_tokens')}")

        try:
            # Make new API call with error context
            updated_counters, retry_response, updated_args = api_responses_with_backoff(
                client=client,
                provider_name=provider_name,
                api_call_counters=counters.to_tuple(),
                error_message_prompt=error_message,
                **llm_input_args,
            )

            counters = APICallCounters(*updated_counters)
            llm_input_args = updated_args

            # Validate the retry response
            retry_validation_result = _validate_single_response(
                retry_response, output_pydantic_json_schema, counters
            )
            is_valid = retry_validation_result.is_valid
            # Update token counts
            total_input_tokens = _calculate_total_input_tokens(
                counters.total_attempts, retry_response.usage.input_tokens
            )
            token_counts.increment_input(total_input_tokens)
            # calculate total output tokens
            total_output_tokens = _calculate_total_output_tokens(
                counters.total_attempts,
                retry_response.usage.output_tokens,
                llm_input_args.get("max_output_tokens", 100),
            )
            token_counts.increment_output(total_output_tokens)

            if is_valid:
                logger.info(f"Validation successful on attempt {attempt}")
                return JsonValidationResult(
                    is_valid=True,
                    json_output=retry_validation_result.json_output,
                    input_tokens=token_counts.input_tokens,
                    output_tokens=token_counts.output_tokens,
                    counters=counters,
                )
            else:
                error_message = f"Validation failed on attempt {attempt}"

        except Exception as e:
            logger.error(f"Error during retry attempt {attempt}: {e}")
            error_message = f"API error on attempt {attempt}: {str(e)}"
            continue

    logger.error(f"All {MAX_RETRIES} validation attempts failed")
    return JsonValidationResult(
        is_valid=False,
        json_output=None,
        input_tokens=0,
        output_tokens=0,
        counters=counters,
    )
