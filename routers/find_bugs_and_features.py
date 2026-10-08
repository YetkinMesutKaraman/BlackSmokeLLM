import traceback

from fastapi import APIRouter, HTTPException

from configs.llm_task_config import LLM_TASKS_CONFIG
from schema_models.pydantic_models import (
    FindBugsAndFeaturesRequest,
    FindBugsAndFeaturesResponse,
)
from utils.task_preps import configure_backend, load_reviews_data, make_request_llm_api

router = APIRouter(prefix="/find_bugs_and_features", tags=["Find Bugs and Features"])


@router.post("/", response_model=FindBugsAndFeaturesResponse)
async def find_bugs_and_features(
    payload: FindBugsAndFeaturesRequest,
):
    """
    Endpoint to find bugs and features from reviews data.
    It takes in a file path to the reviews data and the LLM model name to use for the extraction.
    """
    try:
        # Configure the backend
        backend_config = configure_backend(
            provider_name=payload.provider_name,
            model_name=payload.llm_model_name,
            task_type="find_bugs_and_features",
        )

        # Get task configuration
        task_config = LLM_TASKS_CONFIG["find_bugs_and_features"][payload.provider_name]

        # Load reviews data
        data_reviews = load_reviews_data(reviews_file_path=payload.file_path)

        # filter reviews data
        data_reviews, num_of_reviews = backend_config["data_prep"].filter_reviews_data(
            data_reviews
        )
        print(f"Number of reviews after filtering {num_of_reviews}")

        # Prepare the reviews data for LLM
        input_data_for_llm = backend_config["data_prep"].prepare_reviews_data_for_llm(
            data_reviews["review_text"]
        )
        # Make a request to the LLM API
        llm_output_payload = make_request_llm_api(
            backend_config=backend_config,
            client=backend_config["client"],
            task_type="find_bugs_and_features",
            task_config=task_config,
            num_of_reviews=num_of_reviews,
            input_data_for_llm=input_data_for_llm,
            model_name=payload.llm_model_name,
        )
        # from output dict to pydantic model
        find_bugs_and_features_response = FindBugsAndFeaturesResponse(
            **llm_output_payload
        )
        print("Find Bugs and Features is done")
        return find_bugs_and_features_response
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the request: {str(e)}",
        )
