import traceback

from fastapi import APIRouter, HTTPException
from pprint import pprint
from configs.llm_task_config import LLM_TASKS_CONFIG
from utils.task_preps import configure_backend, load_reviews_data, make_request_llm_api
from schema_models.pydantic_models import (
    ExtractSubTopicsRequest,
    ExtractSubTopicsResponse,
)


router = APIRouter(prefix="/extract_subtopics", tags=["Extract SubTopics"])


@router.post("/", response_model=ExtractSubTopicsResponse)
async def recommend_actions(payload: ExtractSubTopicsRequest):
    # steps:
    # 1. configure the backend
    # 2. load the reviews data
    # 3. filter the reviews data based on the topic name
    # 3. make the request to the LLM API
    # 4. return the response
    try:
        task_type = "extract_subtopics"
        backend_config = configure_backend(
            provider_name=payload.provider_name,
            model_name=payload.llm_model_name,
            task_type=task_type,
        )
        # Get task configuration
        task_config = LLM_TASKS_CONFIG[task_type][payload.provider_name]

        print(f"payload: {payload}")
        # Load reviews data
        data_reviews = load_reviews_data(
            reviews_file_path=payload.file_path,
            topic_type=payload.topic_type,
            topic_name=payload.topic_name,
        )

        # filter reviews data
        data_reviews, num_of_reviews = backend_config["data_prep"].filter_reviews_data(
            data_reviews
        )

        print(f"data_reviews: {data_reviews}")

        # prepare input data for llm
        data_reviews = backend_config["data_prep"].prepare_reviews_data_for_llm(
            data_reviews["review_text"]
        )
        # make request to llm api
        llm_output_payload = make_request_llm_api(
            backend_config=backend_config,
            client=backend_config["client"],
            task_type=task_type,
            task_config=task_config,
            num_of_reviews=num_of_reviews,
            input_data_for_llm=data_reviews,
            model_name=payload.llm_model_name,
            topic_type=payload.topic_type,
            topic_name=payload.topic_name,
        )

        extract_subtopics_response = ExtractSubTopicsResponse(**llm_output_payload)
        # from output dict to pydantic model
        pprint(extract_subtopics_response.valid_json_output, indent=2)
        # return the response
        return extract_subtopics_response
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
