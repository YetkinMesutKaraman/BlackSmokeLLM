import traceback

from fastapi import APIRouter, HTTPException

from configs.llm_task_config import LLM_TASKS_CONFIG
from schema_models.pydantic_models import (
    UnsupervisedTopicExtractionRequest,
    UnsupervisedTopicExtractionResponse,
)
from utils.task_preps import configure_backend, load_reviews_data, make_request_llm_api

router = APIRouter(
    prefix="/extract_unsupervised_topics", tags=["Extract Unsupervised Topics"]
)


# Post-processing steps
def normalize_topic_format(
    response: UnsupervisedTopicExtractionResponse,
) -> UnsupervisedTopicExtractionResponse:
    """Normalize topic names by replacing spaces with underscores and converting to lowercase."""
    response.valid_json_output.positive_topics = [
        topic.replace(" ", "_").lower()
        for topic in response.valid_json_output.positive_topics
    ]
    response.valid_json_output.negative_topics = [
        topic.replace(" ", "_").lower()
        for topic in response.valid_json_output.negative_topics
    ]
    return response


# write topics to a json file
def write_topics_to_json_file(
    response: UnsupervisedTopicExtractionResponse, file_path: str
) -> None:
    """Write the extracted topics to a JSON file."""
    print(f"Writing topics to {file_path}...")
    print("Valid JSON Output:", response)

    with open(file_path, "w") as f:
        f.write(response.valid_json_output.model_dump_json(indent=2))
    print(f"Topics written to {file_path}")


@router.post("/", response_model=UnsupervisedTopicExtractionResponse)
async def extract_unsupervised_topics(
    payload: UnsupervisedTopicExtractionRequest,
):
    """
    Endpoint to extract unsupervised topics from reviews data.
    It takes in a file path to the reviews data and the LLM model name to use for the extraction.
    """
    try:
        print(payload)
        # print(payload)
        # Configure the backend
        backend_config = configure_backend(
            provider_name=payload.provider_name,
            model_name=payload.llm_model_name,
            task_type="extract_unsupervised_topics",
        )

        # Get task configuration
        task_config = LLM_TASKS_CONFIG["extract_unsupervised_topics"][
            payload.provider_name
        ]

        # Load reviews data
        data_reviews = load_reviews_data(reviews_file_path=payload.file_path)

        # filter reviews data
        data_reviews, num_of_reviews = backend_config["data_prep"].filter_reviews_data(
            data_reviews
        )
        print(f"Number of reviews after filtering {num_of_reviews}")

        # prepare input data for llm
        data_reviews = backend_config["data_prep"].prepare_reviews_data_for_llm(
            data_reviews["review_text"]
        )

        # make request to llm api
        # Make LLM API request
        llm_output_payload = make_request_llm_api(
            backend_config=backend_config,
            client=backend_config["client"],
            task_type="extract_unsupervised_topics",
            task_config=task_config,
            num_of_reviews=num_of_reviews,
            input_data_for_llm=data_reviews,
            model_name=payload.llm_model_name,
        )
        # from output dict to pydantic model
        unsupervised_topic_extraction_response = UnsupervisedTopicExtractionResponse(
            **llm_output_payload
        )

        # Normalize topic format
        unsupervised_topic_extraction_response = normalize_topic_format(
            unsupervised_topic_extraction_response
        )
        print("Extracted topics is done successfully")
        # # Write topics to a JSON file
        # # create the destination directory as a child of the given file path with the name of the "results" folder
        # # get dirname of the file path
        # input_file_dir = os.path.dirname(payload.file_path)
        # result_dir = os.path.join(input_file_dir, f"results/{payload.llm_model_name}")
        # os.makedirs(result_dir, exist_ok=True)
        # # get the file name from the file path
        # file_name = os.path.basename(payload.file_path)
        # # remove the file extension from the file name
        # file_name_without_extension = os.path.splitext(file_name)[0]
        # # create the file name with the model name
        # file_name = f"{file_name_without_extension}_{payload.llm_model_name}"
        # file_path = os.path.join(result_dir, f"{file_name}_topics.json")
        # print(f"File path: {file_path}")
        # # Write the topics to a JSON file
        # write_topics_to_json_file(
        #     unsupervised_topic_extraction_response, file_path=file_path
        # )
        # Return the response

        return unsupervised_topic_extraction_response

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
