# Description: This file contains the FastAPI router for tagging reviews with topics using LLM model.
import json  # Import the json library for JSON operations
import os  # Import the os library for file operations
import traceback

import aio_pika  # Import the aio_pika library for RabbitMQ
import pandas as pd
from fastapi import APIRouter, HTTPException, Depends


from configs.llm_task_config import LLM_TASKS_CONFIG
from schema_models.pydantic_models import (
    AllReviewsTagResponse,
    TagReviewsRequest,
)
from utils.task_preps import configure_backend, load_reviews_data

import traceback
import aio_pika  # Import the aio_pika library for RabbitMQ
import json  # Import the json library for JSON operations
import asyncio  # Yeni import ekliyoruz
import time
from typing import Dict

router = APIRouter(prefix="/tag_reviews_with_topics", tags=["Tag Reviews With Topics"])

# Global bağlantı ve kanal değişkenleri

# Global semaphore değişkeni
SEMAPHORE = asyncio.Semaphore(8)


@router.post("/")
async def tag_reviews_with_topics(
    payload: TagReviewsRequest,
):
    """
    Endpoint to tag reviews with topics using LLM model.
    It takes in the following parameters:
    - backend_provider: The backend provider to use for the task.
    - llm_model_name: The LLM model name to use for the task.
    - file_path: The path to the reviews file.
    """

    print(f"payload: {payload}")
    try:
        async with SEMAPHORE:  # Sadece semaphore kontrolü kalıyor
            # Configure the backend
            backend_config = configure_backend(
                provider_name=payload.provider_name,
                model_name=payload.llm_model_name,
                task_type="tag_reviews_with_topics",
            )

        # Get task configuration
        task_config = LLM_TASKS_CONFIG["tag_reviews_with_topics"][payload.provider_name]
        # Get LLM API client, which is async
        async_client = backend_config["async_client"]
        # Get provider name
        provider_name = backend_config["provider_name"]
        # Get async inference function
        async_inference = backend_config["async_inference"].gather_request_llm_api

        # Load reviews data
        data_reviews = load_reviews_data(reviews_file_path=payload.file_path)

        # filter reviews data
        data_reviews, num_of_reviews = backend_config["data_prep"].filter_reviews_data(
            data_reviews
        )

        print(f"Number of reviews after filtering: {num_of_reviews}")

        # prepare input data for tagging
        reviews_data_list = data_reviews["review_text"].to_list()

        print(f"Number of reviews after filtering done")
        # async_inference fonksiyonunu çağır ve tamamlanmasını bekle
        await async_inference(
            async_client,
            provider_name,
            reviews_data_list,
            task_config,
            positive_topics=payload.positive_topics,
            negative_topics=payload.negative_topics,
            model_name=payload.llm_model_name,
            use_supervised_topics=False,
            payload=payload,
            data_reviews=data_reviews,
        )

        print("İşlem başarıyla tamamlandı")

        return {"status": "success", "message": "İşlem başarıyla tamamlandı"}

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
