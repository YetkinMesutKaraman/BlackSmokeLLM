import traceback

from fastapi import APIRouter, HTTPException
from pprint import pprint
from configs.llm_task_config import LLM_TASKS_CONFIG
from utils.task_preps import configure_backend, load_reviews_data, make_request_llm_api
from schema_models.pydantic_models import (
    RecommendActionsRequest,
    RecommendActionsResponse,
    MarketingStrategiesResponse,
)


router = APIRouter(prefix="/recommend_actions", tags=["Recommend Actions"])


@router.post("/", response_model=RecommendActionsResponse | MarketingStrategiesResponse)
async def recommend_actions(payload: RecommendActionsRequest):
    """
    Endpoint to recommend actions for a given topic from reviews data.
    It takes in the following parameters:
    - backend_provider: The backend provider to use for the task.
    - llm_model_name: The LLM model name to use for the task.
    - file_path: The path to the reviews file.
    - topic_type: The type of topic to recommend actions for.
    - topic_name: The topic name to recommend actions for.
    """
    # steps:
    # 1. configure the backend
    # 2. load the reviews data
    # 3. filter the reviews data based on the topic name
    # 3. make the request to the LLM API
    # 4. return the response
    try:
        # Determine task type based on topic type
        actual_task_type = (
            "marketing_strategies"
            if payload.topic_type == "positive"
            else "recommend_actions"
        )

        backend_config = configure_backend(
            provider_name=payload.provider_name,
            model_name=payload.llm_model_name,
            task_type=actual_task_type,
        )
        # Get task configuration
        task_config = LLM_TASKS_CONFIG[actual_task_type][payload.provider_name]

        print(f"payload: {payload}")
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
            task_type=actual_task_type,
            task_config=task_config,
            num_of_reviews=num_of_reviews,
            input_data_for_llm=data_reviews,
            model_name=payload.llm_model_name,
            topic_type=payload.topic_type,
            topic_name=payload.topic_name,
        )
        if payload.topic_type == "positive":
            recommend_actions_response = MarketingStrategiesResponse(
                **llm_output_payload
            )
        else:
            recommend_actions_response = RecommendActionsResponse(**llm_output_payload)
        # from output dict to pydantic model
        pprint(recommend_actions_response.valid_json_output, indent=2)
        # return the response
        return recommend_actions_response
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
