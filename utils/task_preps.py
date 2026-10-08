
import json
import os
from time import perf_counter
import ast
import pandas as pd

# import tiktoken
from dotenv import load_dotenv

# Load OpenAI API client
from openai import AsyncOpenAI, OpenAI

# Load and configure LLM API settings
from configs.llm_task_config import (
    COST_PER_TOKEN,
    LLM_API_TIMEOUT,
    MODEL_CHOICES,
    VALID_SUBTASKS,
)

# Load environment variables
load_dotenv(dotenv_path="../.env", override=True)


def configure_backend(
    provider_name: str, model_name: str, task_type: str, subtasks_model_name: str = None
) -> dict:
    """
    Configure the backend provider and client based on the given provider name.

    Args:
        provider_name (str): The name of the backend provider (e.g., "openai", "google", "anthropic", "groq").
        model_name (str): The name of the model to use.
        task_type (str): The type of task to perform.
        subtasks_model_name (str, optional): JSON-formatted string specifying models for subtasks.

    Returns:
        dict: A dictionary containing the backend configuration.

    Raises:
        ValueError: If the provider name or subtask model name is invalid.
    """
    # Import necessary modules
    from llm_providers.openai_client import (
        async_llm_inference_tagging,
        data_prep_for_llm,
        llm_inference,
        prep_input_args_for_llm,
    )

    # Map provider names to their respective environment variables and client configurations
    provider_configs = {
        "openai": {
            "client": OpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                organization=os.environ["OPENAI_ORG_ID"],
                timeout=LLM_API_TIMEOUT,
                max_retries=0,  # Disable retries for OpenAI client
            ),
            "async_client": AsyncOpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                organization=os.environ["OPENAI_ORG_ID"],
                timeout=LLM_API_TIMEOUT,
            ),
        },
        "google": {
            "client": OpenAI(
                api_key=os.environ["GOOGLE_API_KEY"],
                base_url=os.environ["GOOGLE_API_BASE_URL"],
                timeout=LLM_API_TIMEOUT,
            ),
            "async_client": AsyncOpenAI(
                api_key=os.environ["GOOGLE_API_KEY"],
                base_url=os.environ["GOOGLE_API_BASE_URL"],
                timeout=LLM_API_TIMEOUT,
            ),
        },
    }

    if provider_name not in provider_configs:
        raise ValueError(f"Invalid backend provider: {provider_name}")

    client_config = provider_configs[provider_name]

    backend_config = {
        "provider_name": provider_name,
        "model_name": model_name,
        "subtasks_model_name": {},
        "client": client_config["client"],
        "data_prep": data_prep_for_llm,
        "input_args": prep_input_args_for_llm,
        "inference": llm_inference,
        "async_client": client_config["async_client"],
        "async_inference": async_llm_inference_tagging,
    }

    if subtasks_model_name:
        try:
            subtask_model_dict = json.loads(subtasks_model_name)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format for --subtasks_model_name.")

        task_valid_subtasks = VALID_SUBTASKS.get(task_type, [])
        for subtask, model in subtask_model_dict.items():
            if subtask not in task_valid_subtasks:
                raise ValueError(
                    f"Invalid subtask '{subtask}' for task '{task_type}'. Valid subtasks are: {task_valid_subtasks}."
                )
            if model not in MODEL_CHOICES[provider_name]["llm"]:
                raise ValueError(
                    f"Model '{model}' is not valid for subtask '{subtask}' with provider '{provider_name}'."
                )

        backend_config["subtasks_model_name"] = subtask_model_dict

    return backend_config


# Load reviews data from JSON or CSV
def load_reviews_data(
    reviews_file_path, topic_type: str = None, topic_name: str = None
) -> pd.DataFrame:
    if reviews_file_path:
        if topic_name:
            df_reviews = pd.read_csv(reviews_file_path, sep="¤", engine="python")

            # Convert comma-separated strings to lists
            # Handle NaN values by replacing them with empty strings, then split by comma
            # df_reviews["ai_positive_topics"] = (
            #     df_reviews["ai_positive_topics"]
            #     .fillna("")
            #     .apply(lambda x: x.split(",") if x else [])
            # )
            # df_reviews["ai_negative_topics"] = (
            #     df_reviews["ai_negative_topics"]
            #     .fillna("")
            #     .apply(lambda x: x.split(",") if x else [])
            # )
            # Convert string representations of lists to actual lists
            df_reviews["ai_positive_topics"] = df_reviews["ai_positive_topics"].apply(
                ast.literal_eval
            )
            df_reviews["ai_negative_topics"] = df_reviews["ai_negative_topics"].apply(
                ast.literal_eval
            )
            # preprocess the csv data for the LLM task

            target_columns = [
                "review_id",
                # "title",
                "date_reviewed",
                "review_text",
                "ai_positive_topics",
                "ai_negative_topics",
            ]

            df_reviews.reset_index(drop=True, inplace=True)
            df_reviews = df_reviews.loc[:, target_columns]
            if topic_type == "ai_positive_topics":
                df_reviews = df_reviews[
                    df_reviews["ai_positive_topics"].apply(lambda x: topic_name in x)
                ]
            elif topic_type == "ai_negative_topics":
                df_reviews = df_reviews[
                    df_reviews["ai_negative_topics"].apply(lambda x: topic_name in x)
                ]
            else:
                raise ValueError(f"Invalid topic type: {topic_type}")
            print(f"Number of reviews after filtering by topic name: {len(df_reviews)}")
            # sort dataframe by date reviewed column, first convert it to datetime
            df_reviews["date_reviewed"] = pd.to_datetime(df_reviews["date_reviewed"])
            df_reviews = df_reviews.sort_values(by="date_reviewed")

        else:
            df_reviews = pd.read_csv(reviews_file_path, sep=";")

            # preprocess the csv data for the LLM task

            target_columns = ["review_id", "date_reviewed", "review_text"]

            df_reviews.reset_index(drop=True, inplace=True)
            df_reviews = df_reviews.loc[:, target_columns]
            # sort dataframe by date reviewed column, first convert it to datetime
            df_reviews["date_reviewed"] = pd.to_datetime(df_reviews["date_reviewed"])
            df_reviews = df_reviews.sort_values(by="date_reviewed")
            print(f"\nNumber of reviews before filtering: {len(df_reviews)}")

        return df_reviews
    else:
        raise ValueError("You must provide either --reviews_data or --reviews_file.")


# Write results to file in specified format (CSV or JSON)
def write_results_to_file(args, df):
    if args.write_output_to_file:
        os.makedirs(args.output_file_path, exist_ok=True)
        file_name = (
            args.reviews_file.split("/")[-1].split(".")[0]
            if args.reviews_file
            else f"output_{args.task_type}"
        )

        if args.output_file_format == "csv":
            df.to_csv(f"{args.output_file_path}/{file_name}.csv", index=False, sep=";")
        elif args.output_file_format == "json":
            df.to_json(
                f"{args.output_file_path}/{file_name}.json",
                orient="records",
                force_ascii=False,
            )
        else:
            raise ValueError(
                f"Unsupported output file format: {args.output_file_format}"
            )


# # Calculate the number of input tokens for a given text and model
# def calculate_input_tokens(input_text: str, model_name: str) -> int:
#     """
#     Calculate the number of input tokens for a given text and model.

#     Args:
#         input_text (str): The input text to tokenize.
#         model_name (str): The name of the model to use for tokenization.

#     Returns:
#         int: The number of input tokens.
#     """
#     encoding = tiktoken.encoding_for_model(model_name)
#     num_tokens = len(encoding.encode(input_text))
#     return num_tokens


# LLM API request with backoff and output validation
def make_request_llm_api(
    backend_config: dict,
    client,
    task_type: str,
    task_config: dict,
    num_of_reviews: int = None,
    input_data_for_llm: str = None,
    model_name: str = None,
    topic_type: str = None,
    topic_name: str = None,
) -> dict:
    api_call_counters = (0, 0)  # Reset API call counter for each experiment
    response_completion = None

    # Access the required functionalities from the global config
    llm_backend_provider = backend_config["provider_name"]
    create_user_prompt = backend_config["input_args"].create_user_prompt
    prepare_llm_input_args = backend_config["input_args"].prepare_llm_input_args
    api_responses_with_backoff = backend_config["inference"].api_responses_with_backoff
    validate_llm_output = backend_config["inference"].validate_llm_output

    user_prompt = create_user_prompt(
        task_type=task_type,
        output_json_format=task_config["output_json_format"],
        num_of_reviews=num_of_reviews,
        input_data_for_llm=input_data_for_llm,
        topic_type=topic_type,
        topic_name=topic_name,
    )

    # print(f"\nuser_prompt: {user_prompt}\n")
    # Prepare the input arguments for the LLM model

    llm_input_args = prepare_llm_input_args(user_prompt, task_config["llm_input_args"])
    # Update "model"
    llm_input_args["model"] = model_name

    # Make the initial OpenAI API request
    # print("Requesting LLM API...")
    start_time_initial = perf_counter()
    api_call_counters, response, llm_input_args = api_responses_with_backoff(
        client=client,
        provider_name=llm_backend_provider,
        api_call_counters=api_call_counters,
        error_message_prompt="",
        **llm_input_args,
    )
    end_time_initial = perf_counter()
    # print("LLM API request completed.")
    # validate openai output
    start_time_validation = perf_counter()
    validation_result = validate_llm_output(
        client=client,
        provider_name=llm_backend_provider,
        response=response,
        output_pydantic_json_schema=task_config["output_pydantic_json_schema"],
        llm_input_args=llm_input_args,
        api_call_counters=api_call_counters,
    )
    end_time_validation = perf_counter()

    # log outputs
    total_attempts, validation_attempts = validation_result.counters.to_tuple()
    # calculate the latency of the LLM response and validation, use max 6 decimal points
    latency_of_llm_response = end_time_initial - start_time_initial
    latency_of_llm_response = round(latency_of_llm_response, 4)
    latency_of_validation = end_time_validation - start_time_validation
    latency_of_validation = round(latency_of_validation, 4)

    if llm_backend_provider == "openai":
        input_tokens = validation_result.input_tokens
        output_tokens = validation_result.output_tokens
        expected_cost = (
            input_tokens * COST_PER_TOKEN[llm_backend_provider][model_name]["input"]
        ) + (output_tokens * COST_PER_TOKEN[llm_backend_provider][model_name]["output"])
    elif llm_backend_provider == "google":
        input_tokens = response_completion.usage.prompt_tokens
        output_tokens = response_completion.usage.completion_tokens
        expected_cost = (
            input_tokens * COST_PER_TOKEN[llm_backend_provider][model_name]["input"]
        ) + (output_tokens * COST_PER_TOKEN[llm_backend_provider][model_name]["output"])
    else:
        raise ValueError(f"Invalid backend provider: {llm_backend_provider}")

    # round expected cost to 6 decimal points
    expected_cost = round(expected_cost, 6)

    llm_output_payload = {
        "task_type": task_type,
        "llm_backend_provider": llm_backend_provider,
        "llm_model_name": llm_input_args["model"],
        "num_of_reviews": num_of_reviews,
        "is_valid_output": validation_result.is_valid,
        "total_count_of_api": total_attempts,
        "validation_attempts": validation_attempts,
        "latency_of_llm_response": latency_of_llm_response,
        "latency_of_validation": latency_of_validation,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_of_run": expected_cost,
        "valid_json_output": validation_result.json_output,
    }

    return llm_output_payload


def count_topic_frequency(
    given_unique_topics: dict[str, list[str]],
    df_reviews: pd.DataFrame,
    target_topic_columns: list[str],
) -> dict:
    """
    Counts the frequency of each topic in the given_unique_topics dictionary within the specified target_topic_columns of the df_reviews DataFrame.

    Args:
        given_unique_topics (dict[str, list[str]]): A dictionary containing the unique topics to count the frequency of. The keys represent the topic types (e.g., "positive_topics", "negative_topics"), and the values are lists of topics.
        df_reviews (pd.DataFrame): The DataFrame containing the reviews and the target_topic_columns.
        target_topic_columns (list[str]): A list of column names in the df_reviews DataFrame that contain the target topics.

    Returns:
        dict: A dictionary containing the frequency counts of each topic. The keys represent the topic types, and the values are dictionaries where the keys are the topics and the values are the frequency counts.

    Example:
        given_unique_topics = {"positive_topics": ["topic1", "topic2", "topic3"], "negative_topics": ["topic4", "topic5", "topic6"]}
        target_topic_columns = ["previous_positive_topics", "previous_negative_topics", "new_positive_topics", "new_negative_topics"]
        count_topic_frequency(given_unique_topics, df_reviews, target_topic_columns)
    """
    # we need to provide unqiue topics set to make sure that we count the frequency of each topic
    # for example if a topic is not mentioned in the reviews, we need to count it as 0
    # since our df_reviews DataFrame includes multiple tagging columns, e.g. previous_positive_topics, previous_negative_topics and
    # and new_positive_topics, new_negative_topics, we need to provide the target_topic_columns list
    # example:
    # given_unique_topics = {"positive_topics": ["topic1", "topic2", "topic3"], "negative_topics": ["topic4", "topic5", "topic6"]}
    # target_topic_columns = ["previous_positive_topics", "previous_negative_topics", "new_positive_topics", "new_negative_topics"]
    # thus, we need to detect topic_type based on column name
    topic_value_counts_dict = {}
    for target_column in target_topic_columns:
        topic_value_counts = {}
        # if the column includes None, handle it
        df_reviews[target_column] = df_reviews[target_column].apply(
            lambda x: [] if x is None else x
        )
        # flatten the list of topics
        topic_labels = [
            topic
            for topic_list in df_reviews[target_column].values
            for topic in topic_list
        ]
        # count the frequency of each topic in the given_unique_topics
        if "positive" in target_column:
            topic_type = "positive_topics"
        elif "negative" in target_column:
            topic_type = "negative_topics"
        else:
            raise ValueError(
                f"Invalid target column name: {target_column}. The target column name must include 'positive' or 'negative'."
            )
        for topic in given_unique_topics[topic_type]:
            topic_value_counts[topic] = topic_labels.count(topic)

        # sort the dictionary by value counts
        topic_value_counts = dict(
            sorted(topic_value_counts.items(), key=lambda x: x[1], reverse=True)
        )
        # add to the dictionary
        topic_value_counts_dict[topic_type] = topic_value_counts

    return topic_value_counts_dict


def calculate_coverage(value_counts: dict[str, int], subset: list[str]) -> float:
    """
    Calculate the coverage of a subset of topics based on their mentions.

    Parameters:
    - value_counts (dict): A dictionary containing the counts of mentions for each topic.
    - subset (list): A list of topics to calculate the coverage for.

    Returns:
    - coverage (float): The coverage of the subset as a percentage.

    Example:
    >>> value_counts = {'topic1': 10, 'topic2': 5, 'topic3': 3, 'topic4': 2}
    >>> subset = ['topic1', 'topic2']
    >>> calculate_coverage(value_counts, subset)
    0.75
    """
    subset_mentions = sum(value_counts[topic] for topic in subset)
    total_mentions = sum(value_counts.values())
    coverage = (subset_mentions / total_mentions) if total_mentions > 0 else 0
    return coverage
