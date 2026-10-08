import traceback

from fastapi import APIRouter, HTTPException

from configs.llm_task_config import LLM_TASKS_CONFIG
from schema_models.pydantic_models import ReviewsSummaryRequest, ReviewsSummaryResponse
from utils.task_preps import configure_backend, load_reviews_data, make_request_llm_api

router = APIRouter(prefix="/create_reviews_summary", tags=["Create Reviews Summary"])


@router.post("/", response_model=ReviewsSummaryResponse)
async def create_reviews_summary_endpoint(
    payload: ReviewsSummaryRequest,
):
    """
    Endpoint to generate a summary of reviews using the specified LLM model and backend provider.
    """
    try:
        # print(payload)
        # Configure the backend
        backend_config = configure_backend(
            provider_name=payload.provider_name,
            model_name=payload.llm_model_name,
            task_type="create_reviews_summary",
        )

        # Get task configuration
        task_config = LLM_TASKS_CONFIG["create_reviews_summary"][payload.provider_name]

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

        # Make LLM API request
        llm_output_payload = make_request_llm_api(
            backend_config=backend_config,
            client=backend_config["client"],
            task_type="create_reviews_summary",
            task_config=task_config,
            num_of_reviews=num_of_reviews,
            input_data_for_llm=data_reviews,
            model_name=payload.llm_model_name,
        )

        print(f"llm_output_payload: {llm_output_payload}")

        # Format response
        llm_output_payload["valid_json_output"] = "<br><br>".join(
            llm_output_payload["valid_json_output"]["summary_text"]
        )
        # print(f"pydantic response: {ReviewsSummaryResponse(**llm_output_payload)}")

        return ReviewsSummaryResponse(**llm_output_payload)
    # except RetryException as e:
    #     # reconfigure backend to switch to another provider
    #     backend_config = configure_backend(
    #         provider_name=payload.backend_provider, model_name=payload.llm_model_name, task_type="create_reviews_summary"
    #     )

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
