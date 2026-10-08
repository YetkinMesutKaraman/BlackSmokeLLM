import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from asyncio import sleep
from time import perf_counter
from fastapi import APIRouter, HTTPException
import pandas as pd
import openai

# for environment variables
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
import traceback
import aio_pika  # Import the aio_pika library for RabbitMQ
import json  # Import the json library for JSON operations


# retry logic
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from configs.llm_task_config import COST_PER_TOKEN, MODEL_CHOICES

# import local modules
from llm_providers.openai_client.prep_input_args_for_llm import (
    create_user_prompt,
    prepare_llm_input_args,
)
from schema_models.pydantic_models import (
    AllReviewsTagResponse,
    ReviewTags,
    TagReviewResponse,
    TagReviewsRequest,
)

# Load environment variables
load_dotenv()

## create AirBrake notifier
# NOTIFIER = pybrake.Notifier(**AIRBRAKE_CONFIG)
# set max retries
MAX_RETRIES = 3  # Reduced from 5 to 3 for faster processing
LLM_API_TIMEOUT = 30  # seconds - reduced from 60 to 30 for faster failures

# Configure logging
logger = logging.getLogger(__name__)
# Responses API tuning
MAX_OUTPUT_TOKEN_MULTIPLIER = 1.2


@dataclass
class APICallCounters:
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
    is_valid: bool
    json_output: Optional[Dict[str, Any]]
    input_tokens: int
    output_tokens: int
    counters: APICallCounters


def _handle_token_limit_exceeded(
    response: Optional[openai.types.responses.ParsedResponse],
    kwargs: Dict[str, Any],
    provider_name: str,
    validation_attempt_count: int,
) -> None:
    current_max_output_tokens = kwargs.get("max_output_tokens", 100)
    final_retry = validation_attempt_count >= (MAX_RETRIES - 1)

    if response is None:
        logger.warning(
            f"Response is None, it's likely due to max output tokens exceeded. Output token limit: {current_max_output_tokens}"
        )

        new_token_limit = int(current_max_output_tokens * MAX_OUTPUT_TOKEN_MULTIPLIER)
        kwargs["max_output_tokens"] = new_token_limit
        logger.info(
            f"Increasing token limit from {current_max_output_tokens} to {new_token_limit}"
        )

        model_names = MODEL_CHOICES.get(provider_name, {}).get("llm", [])
        if len(model_names) >= 2:
            current_model = kwargs["model"]
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
    remaining_attempts = MAX_RETRIES - validation_attempt_count
    return (
        f"{initial_prompt}\n\n"
        f"Previous attempt resulted in Pydantic Validation Error: {error}. "
        f"Retry attempt: {validation_attempt_count}, remaining attempts: {remaining_attempts}. "
        f"Please ensure the response follows the exact schema format."
    )


def _extract_initial_prompt(kwargs: Dict[str, Any], counters: APICallCounters) -> str:
    prompt = kwargs["input"][1]["content"]
    if counters.total_attempts > 0 and counters.validation_retry_attempts > 0:
        prompt = prompt.split("\n\n Previous attempt")[0]
    return prompt


def retry_error_callback(retry_state: RetryCallState):
    exception = retry_state.outcome.exception()
    # Log the exception using Airbrake logger
    # NOTIFIER.notify(f"Max retries: {MAX_RETRIES} exceeded!!! {exception}")
    # airbrake_logger.error("Max retries: {MAX_RETRIES} exceeded!!!", exc_info=1)
    # Optionally, you can also notify Airbrake directly
    # Re-raise the exception if you want the program to halt or handle it elsewhere
    raise exception


# apply retry logic for OpenAI Responses API (async)
@retry(
    retry=retry_if_exception_type(
        (
            openai.InternalServerError,
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
        )
    ),
    wait=wait_exponential(multiplier=1, min=4, max=30) + wait_random(0, 2),
    stop=stop_after_attempt(MAX_RETRIES),
    retry_error_callback=retry_error_callback,
)
async def async_completions_with_backoff(
    async_client,
    provider_name: str,
    api_call_counters: tuple,
    error_message_prompt: str | ValidationError,
    **kwargs,
) -> tuple[tuple[int, int], Optional[openai.types.responses.ParsedResponse], dict]:
    """Async Responses API call with retry and prompt augmentation on validation errors."""
    counters = APICallCounters(*api_call_counters)
    initial_user_prompt = _extract_initial_prompt(kwargs, counters)

    response: Optional[openai.types.responses.ParsedResponse] = None
    print(f"kwargs: {kwargs}")
    try:
        response = await async_client.responses.parse(**kwargs)
        if response:
            logger.debug("Received async response from Responses API")
    except ValidationError as e:
        logger.debug(f"Validation error during API call: {e}")
        error_message_prompt = e
    except Exception as e:
        logger.error(f"Unexpected error during API call: {e}")
        raise

    if isinstance(error_message_prompt, ValidationError):
        counters.increment_validation()
        modified_prompt = _build_retry_prompt(
            initial_user_prompt,
            error_message_prompt,
            counters.validation_retry_attempts,
        )
        _handle_token_limit_exceeded(
            response, kwargs, provider_name, counters.validation_retry_attempts
        )
    else:
        modified_prompt = initial_user_prompt

    # Update prompt and counters
    kwargs["input"][1]["content"] = modified_prompt
    counters.increment_total()

    return counters.to_tuple(), response, kwargs


def _validate_single_response(
    response: Optional[openai.types.responses.ParsedResponse],
    schema: BaseModel,
    counters: APICallCounters,
) -> JsonValidationResult:
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
        logger.debug("JSON output is valid according to the schema")
        return JsonValidationResult(
            is_valid=True,
            json_output=validated_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            counters=counters,
        )
    except ValidationError as e:
        logger.debug(f"Validation failed: {e}")
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
    if total_count_of_api <= 0 or input_tokens <= 0:
        return 0
    total_input_tokens = total_count_of_api * input_tokens
    return total_input_tokens


def _calculate_total_output_tokens(
    total_count_of_api: int, output_tokens: int, current_max_output_tokens: int
) -> int:
    if total_count_of_api <= 0 or output_tokens <= 0:
        return 0
    total_output_tokens = output_tokens
    for _ in range(total_count_of_api - 1):
        current_max_output_tokens = (
            current_max_output_tokens / MAX_OUTPUT_TOKEN_MULTIPLIER
        )
        total_output_tokens += int(current_max_output_tokens)
    return total_output_tokens


async def async_validate_llm_output(
    async_client,
    response: Optional[openai.types.responses.ParsedResponse],
    output_pydantic_json_schema: BaseModel,
    provider_name: str,
    llm_input_args: dict,
    api_call_counters=(0, 0),
    error_message_prompt="",
) -> JsonValidationResult:
    counters = APICallCounters(*api_call_counters)

    initial_validation_result = _validate_single_response(
        response, output_pydantic_json_schema, counters
    )
    is_valid = initial_validation_result.is_valid
    counters = initial_validation_result.counters

    if is_valid:
        logger.debug("Initial validation successful")
        return initial_validation_result

    logger.debug("Starting validation retry attempts (async)")
    if isinstance(error_message_prompt, ValidationError):
        error_message = str(error_message_prompt)
    else:
        error_message = (
            "Initial validation failed. Retrying with modified prompt and parameters."
        )
    token_counts = TrackTokenCounts()

    for attempt in range(1, MAX_RETRIES + 1):
        logger.debug(
            f"Validation retry attempt {attempt}/{MAX_RETRIES} | model={llm_input_args.get('model')} | max_output_tokens={llm_input_args.get('max_output_tokens')}"
        )

        try:
            (
                updated_counters,
                retry_response,
                updated_args,
            ) = await async_completions_with_backoff(
                async_client=async_client,
                provider_name=provider_name,
                api_call_counters=counters.to_tuple(),
                error_message_prompt=error_message,
                **llm_input_args,
            )

            counters = APICallCounters(*updated_counters)
            llm_input_args = updated_args

            retry_validation_result = _validate_single_response(
                retry_response, output_pydantic_json_schema, counters
            )
            is_valid = retry_validation_result.is_valid

            total_input_tokens = _calculate_total_input_tokens(
                counters.total_attempts, retry_response.usage.input_tokens
            )
            token_counts.increment_input(total_input_tokens)

            total_output_tokens = _calculate_total_output_tokens(
                counters.total_attempts,
                retry_response.usage.output_tokens,
                llm_input_args.get("max_output_tokens"),
            )
            token_counts.increment_output(total_output_tokens)

            if is_valid:
                logger.debug(f"Validation successful on attempt {attempt}")
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

    logger.error(f"All {MAX_RETRIES} validation attempts failed (async)")
    return JsonValidationResult(
        is_valid=False,
        json_output=None,
        input_tokens=0,
        output_tokens=0,
        counters=counters,
    )


async def async_make_request_llm(
    async_client,
    provider_name,
    customer_review: str,
    output_json_format: str,
    task_config: dict,
    positive_topics: list,
    negative_topics: list,
    empty_json_format: str = None,
    model_name: str = None,
    use_supervised_topics=False,
) -> TagReviewResponse:
    """Make an asynchronous request to the LLM API for tagging a single review.

    Args:
        async_client (AsyncOpenAI): The asynchronous OpenAI client.
        provider_name (str): The name of the LLM provider.
        customer_review (str): The customer review to be tagged.
        output_json_format (str): The desired output JSON format.
        task_config (dict): Configuration parameters for the LLM task.
        positive_topics (list): A list of positive topics for sentiment analysis.
        negative_topics (list): A list of negative topics for sentiment analysis.
        empty_json_format (str, optional): The empty JSON format. Defaults to None.
        model_name (str, optional): The name of the LLM model. Defaults to None.
        use_supervised_topics (bool, optional): Flag indicating whether to use supervised topics. Defaults to False.

    Returns:
        TagReviewResponse: The response from the LLM API for tagging the customer review.
    """
    api_call_counters = (0, 0)  # Reset API call counter for each experiment
    response_completion = None
    # use supervised topics, else use unsupervised topics that are generated from the LLM model
    if use_supervised_topics:
        positive_topics = task_config["default_supervised_topics"]["positive_topics"]
        negative_topics = task_config["default_supervised_topics"]["negative_topics"]

    # create user prompt
    user_message = create_user_prompt(
        task_type="tag_reviews_with_topics",
        customer_review=customer_review,
        current_positive_topics=positive_topics,
        current_negative_topics=negative_topics,
        output_json_format=output_json_format,
        empty_json_format=empty_json_format,
    )
    # print(f"\nUser message: {user_message}\n")
    # create LLM input args
    llm_input_args = prepare_llm_input_args(user_message, task_config["llm_input_args"])
    # update model name in llm input args
    llm_input_args["model"] = model_name

    # Make the initial OpenAI API request
    # print("Requesting LLM API...")
    start_time_initial = perf_counter()
    (
        api_call_counters,
        response_completion,
        llm_input_args,
    ) = await async_completions_with_backoff(
        async_client,
        provider_name,
        api_call_counters,
        error_message_prompt="",
        **llm_input_args,
    )
    end_time_initial = perf_counter()

    # validate openai output
    start_time_validation = perf_counter()
    validation_result = await async_validate_llm_output(
        async_client=async_client,
        response=response_completion,
        output_pydantic_json_schema=ReviewTags,
        provider_name=provider_name,
        llm_input_args=llm_input_args,
        api_call_counters=api_call_counters,
    )
    # If the output is not valid, raise an error
    # if is_valid_output == 0:
    #     raise HTTPException(
    #         status_code=500,
    #         detail="Internal Server Error: The output from the LLM model is not valid.",
    #     )

    end_time_validation = perf_counter()
    # log outputs
    total_attempts, validation_attempts = validation_result.counters.to_tuple()
    # calculate the latency of the LLM response and validation, do not return scientific notation
    # use max 6 decimal places
    latency_of_llm_response = end_time_initial - start_time_initial
    latency_of_llm_response = round(float(latency_of_llm_response), 4)
    latency_of_validation = end_time_validation - start_time_validation
    latency_of_validation = round(float(latency_of_validation), 4)
    # get the number of tokens used for the prompt and completion (Responses API)
    input_tokens = validation_result.input_tokens
    output_tokens = validation_result.output_tokens
    # calculate the cost of the run
    expected_cost = (
        input_tokens * COST_PER_TOKEN[provider_name][model_name]["input"]
    ) + (output_tokens * COST_PER_TOKEN[provider_name][model_name]["output"])
    # round the cost to 6 decimal places
    expected_cost = round(float(expected_cost), 6)
    # get the topics that are not given but generated and used when tagging the review
    not_given_topics = []
    # get the false positive and false negative topics
    false_positive_topics = []
    false_negative_topics = []

    llm_output_payload = {
        "task_type": "tag_reviews_with_topics",
        "llm_backend_provider": provider_name,
        "llm_model_name": llm_input_args["model"],
        "is_valid_output": 1 if validation_result.is_valid else 0,
        "total_count_of_api": total_attempts,
        "validation_attempts": validation_attempts,
        "latency_of_llm_response": latency_of_llm_response,
        "latency_of_validation": latency_of_validation,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_of_run": expected_cost,
        "valid_json_output": validation_result.json_output,
        "not_given_topics": not_given_topics,
        "false_positive_topics": false_positive_topics,
        "false_negative_topics": false_negative_topics,
    }

    return TagReviewResponse(**llm_output_payload)


rabbitmq_connection = None
rabbitmq_channel = None


async def get_rabbitmq_channel():
    """RabbitMQ bağlantısını ve kanalını yönetir."""
    global rabbitmq_connection, rabbitmq_channel

    max_retries = 3
    retry_delay = 2  # saniye

    for attempt in range(max_retries):
        try:
            if rabbitmq_connection is None or rabbitmq_connection.is_closed:
                print(
                    f"RabbitMQ bağlantısı kuruluyor... Deneme {attempt + 1}/{max_retries}"
                )
                rabbitmq_connection = await aio_pika.connect_robust(
                    "amqp://guest:guest@172.28.0.10",
                    timeout=10,  # 10 saniye timeout
                )
                print("RabbitMQ bağlantısı başarılı")

            if rabbitmq_channel is None or rabbitmq_channel.is_closed:
                print(
                    f"RabbitMQ kanalı oluşturuluyor... Deneme {attempt + 1}/{max_retries}"
                )
                rabbitmq_channel = await rabbitmq_connection.channel()
                print("RabbitMQ kanalı başarılı")

            return rabbitmq_channel

        except Exception as e:
            print(f"RabbitMQ bağlantı hatası (Deneme {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"{retry_delay} saniye sonra tekrar denenecek...")
                await sleep(retry_delay)
                retry_delay *= 2  # Her denemede bekleme süresini iki katına çıkar
            else:
                print(
                    "Maksimum deneme sayısına ulaşıldı. RabbitMQ bağlantısı kurulamadı."
                )
                raise HTTPException(
                    status_code=503, detail="RabbitMQ servisine bağlanılamıyor"
                )


# post process the tagging response: STEP 1
# LLM can tag topics multiple times, so we need to remove the duplicates
def remove_duplicate_topics(
    tagging_response: AllReviewsTagResponse,
) -> AllReviewsTagResponse:
    """
    Removes duplicate topics from the positive_topics and negative_topics lists in each review response.

    Parameters:
    tagging_response (AllReviewsTagResponse): The tagging response containing a list of review responses.

    Returns:
    AllReviewsTagResponse: The tagging response with duplicate topics removed from each review response.
    """
    for response in tagging_response.all_reviews:
        response.valid_json_output.positive_topics = list(
            set(response.valid_json_output.positive_topics)
        )
        response.valid_json_output.negative_topics = list(
            set(response.valid_json_output.negative_topics)
        )
    return tagging_response


# post process the tagging response: STEP 2
# LLM can create extra topics that are not in the given topics set(positive and negative topics),
# so we need to add the not given topics to TaggingResponse object with "not_given_topics" key
# after logging the topics, we need to filter them
def check_tagged_topics_only_belong_to_the_given_set(
    tagging_response: AllReviewsTagResponse, positive_topics, negative_topics
) -> AllReviewsTagResponse:
    all_topics_set = positive_topics + negative_topics
    # for each response, check the tagged topics only belong to the given all topics set(positive+negative)
    for response in tagging_response.all_reviews:
        # add the not given topics to TaggingResponse object with "not_given_topics" key
        # first get the tagged topics set
        tagged_topics_set = (
            response.valid_json_output.positive_topics
            + response.valid_json_output.negative_topics
        )
        # second get the not given topics
        response.not_given_topics = [
            topic for topic in tagged_topics_set if topic not in all_topics_set
        ]

        # check the tagged topics only belong to the given set and filter them
        response.valid_json_output.positive_topics = [
            topic
            for topic in response.valid_json_output.positive_topics
            if topic in all_topics_set
        ]
        response.valid_json_output.negative_topics = [
            topic
            for topic in response.valid_json_output.negative_topics
            if topic in all_topics_set
        ]
    return tagging_response


# post process the tagging response: STEP 3
# LLM can tag topics falsely,
# so we need to add the false positive and false negative topics to TaggingResponse object with "false_positive_topics" and "false_negative_topics" keys
# after logging the topics, we need to filter them
def remove_false_positive_and_false_negative_topics(
    tagging_response: AllReviewsTagResponse, positive_topics, negative_topics
) -> AllReviewsTagResponse:
    # for each responce, check the each topic is labelled correctly or not and add the false positive and false negative topics
    for response in tagging_response.all_reviews:
        # add the false positive and false negative topics to TaggingResponse object with "false_positive_topics" and "false_negative_topics" keys
        response.false_positive_topics = [
            topic
            for topic in response.valid_json_output.positive_topics
            if topic not in positive_topics
        ]
        response.false_negative_topics = [
            topic
            for topic in response.valid_json_output.negative_topics
            if topic not in negative_topics
        ]

        # check the each topic is labelled correctly or not and filter them
        response.valid_json_output.positive_topics = [
            topic
            for topic in response.valid_json_output.positive_topics
            if topic in positive_topics
        ]
        response.valid_json_output.negative_topics = [
            topic
            for topic in response.valid_json_output.negative_topics
            if topic in negative_topics
        ]
    return tagging_response


# post process the tagging response: STEP 4
# add review_id column from the input DataFrame to the tagging response
def add_review_id_info_to_tagging_response(
    tagging_response: AllReviewsTagResponse, df_review_id_series: pd.Series
) -> AllReviewsTagResponse:
    """
    Eşleşen review_id'leri tagging response'a ekler.

    Args:
        tagging_response: Etiketleme yanıtları
        df_review_id_series: Review ID'leri içeren pandas Series
    """
    # Uzunlukları kontrol et
    if len(tagging_response.all_reviews) != len(df_review_id_series):
        print(
            f"Uyarı: Yanıt sayısı ({len(tagging_response.all_reviews)}) ile review ID sayısı ({len(df_review_id_series)}) eşleşmiyor"
        )
        # Daha kısa olan listeyi baz al
        length = min(len(tagging_response.all_reviews), len(df_review_id_series))
        df_review_id_series = df_review_id_series[:length]
        tagging_response.all_reviews = tagging_response.all_reviews[:length]

    # Review ID'leri eşle
    for idx, review_id in enumerate(df_review_id_series):
        tagging_response.all_reviews[idx].review_id = review_id

    return tagging_response


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        return super().default(obj)


async def send_message_to_queue(
    queue: str,
    all_reviews_tag_response: AllReviewsTagResponse,
    payload: TagReviewsRequest,
):
    max_retries = 3
    retry_delay = 2  # saniye

    for attempt in range(max_retries):
        try:
            channel = await get_rabbitmq_channel()

            # Kuyruk tanımlaması
            queue_object = await channel.declare_queue(
                queue, durable=True, auto_delete=False
            )

            # Mesaj içeriği hazırlama
            message_data = {
                "payload": payload.json(),
                "topics": all_reviews_tag_response.json(),
            }
            message_body = json.dumps(message_data)

            # Mesaj gönderme
            message = aio_pika.Message(
                body=message_body.encode(),
                content_type="application/json",
                content_encoding="utf-8",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            )

            await channel.default_exchange.publish(message, routing_key=queue)

            print(f"Mesaj başarıyla '{queue}' kuyruğuna gönderildi")
            return  # Başarılı olursa fonksiyondan çık

        except Exception as error:
            print(
                f"RabbitMQ'ya mesaj gönderme hatası (Deneme {attempt + 1}/{max_retries}): {error}"
            )
            traceback.print_exc()

            if attempt < max_retries - 1:
                print(f"{retry_delay} saniye sonra tekrar denenecek...")
                await sleep(retry_delay)
                retry_delay *= 2  # Her denemede bekleme süresini iki katına çıkar
            else:
                print("Maksimum deneme sayısına ulaşıldı. Mesaj gönderilemedi.")
                raise HTTPException(
                    status_code=500,
                    detail=f"RabbitMQ'ya mesaj gönderilemedi: {str(error)}",
                )


# Uygulama kapatılırken bağlantıyı temizleme
async def cleanup_rabbitmq():
    """RabbitMQ bağlantısını temizler."""
    global rabbitmq_connection, rabbitmq_channel

    if rabbitmq_channel and not rabbitmq_channel.is_closed:
        await rabbitmq_channel.close()

    if rabbitmq_connection and not rabbitmq_connection.is_closed:
        await rabbitmq_connection.close()

    rabbitmq_channel = None
    rabbitmq_connection = None


# FastAPI uygulamanıza cleanup işlemini eklemek için:


async def gather_request_llm_api(
    async_client,
    provider_name,
    reviews_data_list: list[str],
    task_config: dict,
    positive_topics: list,
    negative_topics: list,
    model_name: str = None,
    use_supervised_topics=True,
    batch_size=100,  # Reduced from 1000 to 100 for better performance
    delay_seconds=0,
    payload: TagReviewsRequest = None,
    data_reviews=None,
    max_concurrent_batches=3,  # New parameter to control parallel batch processing
) -> None:
    from time import perf_counter

    start_time_total = perf_counter()
    print(f"🚀 Starting processing of {len(reviews_data_list)} reviews")
    print(
        f"📊 Configuration: batch_size={batch_size}, max_concurrent_batches={max_concurrent_batches}"
    )

    # Create all batches upfront
    batches = []
    for i in range(0, len(reviews_data_list), batch_size):
        batch = reviews_data_list[i : i + batch_size]
        batch_review_ids = data_reviews["review_id"][i : i + batch_size]
        batches.append((i, batch, batch_review_ids))

    print(f"📦 Created {len(batches)} batches with batch_size={batch_size}")

    # Process batches in parallel groups
    total_processed = 0
    total_failed = 0

    for batch_group_start in range(0, len(batches), max_concurrent_batches):
        batch_group_time = perf_counter()
        batch_group = batches[
            batch_group_start : batch_group_start + max_concurrent_batches
        ]

        # Create tasks for parallel batch processing
        batch_tasks = [
            process_single_batch(
                async_client,
                provider_name,
                (batch_data, batch_review_ids),
                task_config,
                positive_topics,
                negative_topics,
                model_name,
                use_supervised_topics,
                payload,
                batch_index,
            )
            for batch_index, batch_data, batch_review_ids in batch_group
        ]

        # Process batches in parallel
        try:
            results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Count results
            group_processed = len([r for r in results if not isinstance(r, Exception)])
            group_failed = len([r for r in results if isinstance(r, Exception)])
            total_processed += group_processed
            total_failed += group_failed

            batch_group_duration = perf_counter() - batch_group_time
            current_group = batch_group_start // max_concurrent_batches + 1
            total_groups = (len(batches) - 1) // max_concurrent_batches + 1

            print(
                f"⚡ Batch group {current_group}/{total_groups} completed in {batch_group_duration:.2f}s"
            )
            print(f"📈 Progress: {total_processed} processed, {total_failed} failed")

        except Exception as group_error:
            print(f"❌ Batch group processing error: {group_error}")
            traceback.print_exc()

        # Delay between batch groups to respect rate limits
        if batch_group_start + max_concurrent_batches < len(batches):
            await sleep(delay_seconds)

    # Final performance summary
    total_duration = perf_counter() - start_time_total
    reviews_per_second = (
        len(reviews_data_list) / total_duration if total_duration > 0 else 0
    )

    print("🏁 Processing completed!")
    print(f"⏱️  Total duration: {total_duration:.2f} seconds")
    print(f"🚀 Processing rate: {reviews_per_second:.2f} reviews/second")
    print(f"✅ Successfully processed: {total_processed}")
    print(f"❌ Failed: {total_failed}")
    print(
        f"📊 Success rate: {(total_processed / (total_processed + total_failed) * 100):.1f}%"
        if (total_processed + total_failed) > 0
        else "N/A"
    )


async def process_single_batch(
    async_client,
    provider_name,
    batch,
    task_config,
    positive_topics,
    negative_topics,
    model_name,
    use_supervised_topics,
    payload,
    batch_index,
):
    """Process a single batch of reviews with optimized error handling."""
    from time import perf_counter

    batch_reviews, batch_review_ids = batch
    batch_start_time = perf_counter()
    batch_num = batch_index // 100 + 1

    try:
        print(f"🔄 Processing batch {batch_num} with {len(batch_reviews)} reviews")

        # Create semaphore to limit concurrent requests within batch
        semaphore = asyncio.Semaphore(20)  # Limit to 20 concurrent requests per batch

        async def process_single_review(review):
            async with semaphore:
                return await async_make_request_llm(
                    async_client=async_client,
                    provider_name=provider_name,
                    customer_review=review,
                    output_json_format=task_config["output_json_format"],
                    task_config=task_config,
                    positive_topics=positive_topics,
                    negative_topics=negative_topics,
                    empty_json_format=task_config["empty_json_format"],
                    model_name=model_name,
                    use_supervised_topics=use_supervised_topics,
                )

        # Process reviews in batch with controlled concurrency
        tasks_get_reviews = [process_single_review(review) for review in batch_reviews]
        matched_reviews = await asyncio.gather(
            *tasks_get_reviews, return_exceptions=True
        )

        # Filter out exceptions and keep only successful results
        successful_reviews = []
        failed_count = 0
        for i, result in enumerate(matched_reviews):
            if isinstance(result, Exception):
                print(f"Review {i} failed: {result}")
                failed_count += 1
            else:
                successful_reviews.append(result)

        if failed_count > 0:
            print(
                f"Batch had {failed_count} failed reviews out of {len(batch_reviews)}"
            )

        if not successful_reviews:
            print(
                f"Batch {batch_index // 100 + 1} had no successful reviews, skipping..."
            )
            return

        all_reviews_tag_response = AllReviewsTagResponse(all_reviews=successful_reviews)
        print(
            f"Batch {batch_index // 100 + 1} - Reviews after tagging: {len(all_reviews_tag_response.all_reviews)}"
        )

        # Optimized post-processing - combine operations
        all_reviews_tag_response = optimize_post_processing(
            all_reviews_tag_response, payload.positive_topics, payload.negative_topics
        )

        # Add review IDs only for successful reviews
        successful_review_ids = batch_review_ids[: len(successful_reviews)]
        all_reviews_tag_response = add_review_id_info_to_tagging_response(
            all_reviews_tag_response, successful_review_ids
        )

        # Send to queue asynchronously
        try:
            await send_message_to_queue(
                "tag_reviews_with_topics", all_reviews_tag_response, payload
            )

            batch_duration = perf_counter() - batch_start_time
            reviews_per_sec = (
                len(batch_reviews) / batch_duration if batch_duration > 0 else 0
            )

            print(
                f"✅ Batch {batch_num} completed in {batch_duration:.2f}s ({reviews_per_sec:.1f} reviews/s)"
            )

        except Exception as queue_error:
            print(f"❌ RabbitMQ error for batch {batch_num}: {queue_error}")
            traceback.print_exc()

    except Exception as batch_error:
        batch_duration = perf_counter() - batch_start_time
        print(f"❌ Batch {batch_num} failed after {batch_duration:.2f}s: {batch_error}")
        traceback.print_exc()


def optimize_post_processing(
    all_reviews_tag_response, positive_topics, negative_topics
):
    """Combine post-processing steps for better performance."""
    # Apply all post-processing steps in sequence (these are likely fast operations)
    all_reviews_tag_response = remove_duplicate_topics(all_reviews_tag_response)
    all_reviews_tag_response = check_tagged_topics_only_belong_to_the_given_set(
        all_reviews_tag_response, positive_topics, negative_topics
    )
    all_reviews_tag_response = remove_false_positive_and_false_negative_topics(
        all_reviews_tag_response, positive_topics, negative_topics
    )
    return all_reviews_tag_response
