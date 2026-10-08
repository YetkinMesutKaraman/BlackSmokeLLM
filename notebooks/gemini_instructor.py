import instructor
from google import genai
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.task_preps import load_reviews_data
from llm_providers.google_client.data_prep_for_llm import (
    filter_reviews_data,
    prepare_reviews_data_for_llm,
)

load_dotenv(dotenv_path=os.path.join(project_root, ".env"), override=True)

file_path = "./data/reviews_2025_02_15_123926_1919.csv"

# Load reviews data
data_reviews = load_reviews_data(reviews_file_path=file_path)

# filter reviews data
data_reviews, num_of_reviews = filter_reviews_data(data_reviews)
print(f"Number of reviews after filtering {num_of_reviews}")

# prepare input data for llm
data_reviews = prepare_reviews_data_for_llm(data_reviews["review_text"])

delimiter = "###"
output_json_format = """{"positive_topics": list[str], "negative_topics": list[str]}"""
user_message = f"""
You are provided with {num_of_reviews} user reviews for a Mobile Application. 
Each review is enclosed between {delimiter} characters.

### Task:
Extract **positive** and **negative** main topic phrases from all {num_of_reviews} reviews.  
- **Language**: Output topics in English.  
- **Output Format**: Return a JSON dictionary in this format: {output_json_format}.

### Instructions:
1. **Consolidate related topics:**  
   - Merge synonymous or closely related phrases into a single topic.  
   - **Example:**  
     - Input: ["bad_experience_with_room_cleaning", "poor_cleaning"]  
     - Output: ["poor_cleaning"]  

2. **Avoid redundant prefixes and structures:**  
   - **Example:**  
     - Input: ["bad_experience_with_room_noise", "noisy_rooms"]  
     - Output: ["noisy_rooms"]  

3. **Ensure topic diversity:**  
   - Extract semantically diverse topics to maximize coverage without redundancy.  

4. **Split combined topics (unless synonymous):**  
   - If a topic contains "and" or "or," split them unless they represent the same concept.  
   - **Example:**  
     - Input: "spa_and_gym_facilities"  
     - Output: ["spa_facilities", "gym_facilities"]  

5. **Output constraints:**  
   - **Do not hallucinate** topics that are not present in the reviews.  
   - **Topic limit:** Do not exceed {num_of_reviews // 10 + 1} topics.

### Data:
{data_reviews}
"""

# user_message = f"""I am providing you a dataset of user reviews for a Mobile Application. \
# Each review is placed in between two {delimiter} characters. There are {num_of_reviews} reviews. \
# Your task is to identify the topic phrases using all {num_of_reviews} user reviews. \
# Output topic phrases should be in English. \
# STEP 1: Identify the positive topics phrases \
# STEP 2: Identify the negative topics phrases \
# STEP 3: Output the topics in JSON format. \
# It is crucial to consolidate synonymous or closely related topic phrases to provide a cleaner output. Ensure that related topics boil down into a single topic. \
# Ensure that topic phrases are consolidated to avoid redundancy. \


# Be semantically diverse as much as possible since diversity richens the output in the sense of topic coverage. \
# Total number of topics must be less than {num_of_reviews//10 +1}, do not exceed this maximum limit. \
# Here is the data: \n{data_reviews}\n."""

sys_instruct = """You are an expert AI language model specialized in extracting concise, high-quality topics from customer reviews."""


# output data schema for unsupervised topic extraction from LLM
class ReviewsUnsupervisedTopics(BaseModel):
    positive_topics: list[str] = Field(
        description="List of positive topics which are extracted from the reviews."
    )
    negative_topics: list[str] = Field(
        description="List of negative topics which are extracted from the reviews."
    )


genai.Client()

client = instructor.from_gemini(
    client=genai.Generative,
    mode=instructor.Mode.GEMINI_JSON,
)

# note that client.chat.completions.create will also work
resp = client.messages.create(
    messages=[
        {
            "role": "system",
            "content": sys_instruct,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ],
    response_model=ReviewsUnsupervisedTopics,
    max_tokens=1000,
    temperature=0.0,
)
