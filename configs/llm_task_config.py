from schema_models.pydantic_models import (
    BugsAndFeatures,
    ReviewsSummary,
    ReviewsUnsupervisedTopics,
    ReviewTags,
    RecommendActions,
    MarketingStrategies,
    ExtractSubTopics,
)

LLM_API_TIMEOUT = 60  # seconds

# AIRBRAKE_CONFIG = {
#     "project_id": 509604,
#     "project_key": "a123d7e2da6b3fbc933cacce7a47280d",
#     "environment": os.environ["PYTHON_ENV"],  # development, staging, production
# }

# Updated MODEL_CHOICES dictionary to include both LLM and embedding models
MODEL_CHOICES = {
    "openai": {
        "llm": [
            "gpt-4.1-nano",
            "gpt-4.1-mini",
            "gpt-4.1, gpt-5-nano, gpt-5-mini, gpt-5",
        ],
        "embedding": ["text-embedding-3-small", "text-embedding-3-large"],
    },
    "google": {"llm": ["gemini-2.0-flash", "gemini-2.5-pro-preview-03-25"]},
}
# Define valid subtask names
VALID_SUBTASKS = {
    "create_reviews_summary": [],
    "extract_unsupervised_topics": ["tag_reviews_with_topics"],
    "tag_reviews_with_topics": [],
    "update_topics": ["extract_unsupervised_topics", "tag_reviews_with_topics"],
}

# define the cost per token for each model in a provider
COST_PER_TOKEN = {
    "openai": {
        "gpt-4.1-nano": {
            "input": 0.10 * 0.000001,
            "output": 0.40 * 0.000001,
        },
        "gpt-4.1-mini": {
            "input": 0.15 * 0.000001,
            "output": 0.60 * 0.000001,
        },
        "gpt-4.1": {
            "input": 1.5 * 0.000001,
            "output": 6.0 * 0.000001,
        },
        "gpt-4o-mini": {"input": 0.15 * 0.000001, "output": 0.60 * 0.000001},
        "gpt-4o": {"input": 2.5 * 0.000001, "output": 10 * 0.000001},
        "text-embedding-3-small": {"input": 0.02 * 0.000001},
        "text-embedding-3-large": {"input": 0.13 * 0.000001},
    },
    "google": {
        "gemini-2.0-flash": {"input": 0.10 * 0.000001, "output": 0.40 * 0.000001},
        "gemini-2.5-flash-preview-05-20": {
            "input": 0.15 * 0.000001,
            "output": 0.60 * 0.000001,
        },
    },
}


### LLM Tasks Configuration
LLM_TASKS_CONFIG = {
    "create_reviews_summary": {
        "openai": {
            "llm_input_args": {
                "model": "gpt-4.1-mini",
                "timeout": 60,  # seconds
                "max_output_tokens": 500,
                "temperature": 0.2,
                "top_p": 0.95,
                "system_message": """You are a trained AI Language model specialized on creating a summary text from customer reviews. Respond with only JSON format.""",
                "text_format": ReviewsSummary,
            },
            "output_json_format": """{"reviews_summary": ["summary_paragraph_1", "summary_paragraph_2", "summary_paragraph_3"]}""",
            "output_pydantic_json_schema": ReviewsSummary,
        },
        "google": {
            "llm_input_args": {
                "model": "gemini-2.0-flash",
                "timeout": 60,  # seconds
                "max_output_tokens": 500,
                "temperature": 0.1,
                "system_message": """You are a trained AI Language model specialized on creating a summary text from customer reviews. Respond with only JSON format.""",
                "response_format": {"type": "json_object"},
            },
            "output_json_format": """{"reviews_summary": ["summary_paragraph_1", "summary_paragraph_2", "summary_paragraph_3"]}""",
            "output_pydantic_json_schema": ReviewsSummary,
        },
        "Retry": {
            "max_retries": 3,
        },
    },
    "extract_unsupervised_topics": {
        "openai": {
            "llm_input_args": {
                "model": "gpt-4o-mini",
                "timeout": 60,  # seconds
                "max_output_tokens": 500,
                "temperature": 0.2,
                "top_p": 0.99,
                "system_message": """You are an expert AI language model specialized in extracting concise, high-quality topics from customer reviews. Always respond with English language.""",
                "text_format": ReviewsUnsupervisedTopics,
            },
            "output_json_format": """{"positive_topics": list[string], "negative_topics": list[string]}""",
            "output_pydantic_json_schema": ReviewsUnsupervisedTopics,
            "max_num_of_topics": 19,  # Maximum number of topics to extract
            "coverage_th": 0.95,  # to eliminate low frequency topics
        },
        "google": {
            "llm_input_args": {
                "model": "gemini-2.0-flash",
                "timeout": 60,  # seconds
                "max_output_tokens": 800,
                "temperature": 0.0,
                "system_message": """You are an expert AI language model specialized in extracting concise, high-quality topics from customer reviews. Always respond with English language.""",
                "text_format": ReviewsUnsupervisedTopics,
            },
            "output_json_format": """{"positive_topics": list[string], "negative_topics": list[string]}""",
            "output_pydantic_json_schema": ReviewsUnsupervisedTopics,
            "max_num_of_topics": 19,  # Maximum number of topics to extract
            "coverage_th": 0.95,  # to eliminate low frequency topics
        },
        "Retry": {
            "max_retries": 3,
        },
    },
    "tag_reviews_with_topics": {
        "openai": {
            "llm_input_args": {
                "model": "gpt-4o",
                "timeout": 60,  # seconds
                "max_output_tokens": 200,
                "temperature": 0.0,
                "top_p": 0.9,
                "system_message": """You are a trained AI Language model specialized on labelling user reviews with given topics. You are an expert multilabel classifier.""",
                "text_format": ReviewTags,
            },
            "output_json_format": """{"positive_topics": list[string], "negative_topics": list[string]}""",
            "empty_json_format": """{"positive_topics": [], "negative_topics": []}""",
            "output_pydantic_json_schema": ReviewTags,
        },
        "google": {
            "llm_input_args": {
                "model": "gemini-2.0-flash",
                "timeout": 60,  # seconds
                "max_output_tokens": 200,
                "temperature": 0.0,
                "top_p": 0.9,
                "system_message": """You are a trained AI Language model specialized on labelling user reviews with given topics. You are an expert multilabel classifier.""",
                "text_format": ReviewTags,
            },
            "output_json_format": """{"positive_topics": list[string], "negative_topics": list[string]}""",
            "empty_json_format": """{"positive_topics": [], "negative_topics": []}""",
            "output_pydantic_json_schema": ReviewTags,
        },
        "Retry": {
            "max_retries": 3,
        },
    },
    "update_topics": {
        "openai": {
            "embedding_input_args": {
                "model": "text-embedding-3-large",
            },
            "deprecated_coverage_th": 0.3,  # to decide if a topic is covered in the mention space
            "emergent_coverage_th": 0.3,  # to decide if a topic is emergent based on the previous topics mention space
            "similarity_th_upper": 0.7,  # to decide if a topic is similar to the existing topics, if yes, it is merged
            "similarity_th_lower": 0.4,  # if a topic is not similar to the existing topics, it is considered as a new topic
            "adjusted_co_occurrence_th": 0.7,  # to decide if a topic is co-occurred with another topic
            "relaxed_similarity_th": 0.5,  # it is used when previous base topics and new topics have common phrases
        },
    },
    "find_bugs_and_features": {
        "openai": {
            "llm_input_args": {
                "model": "gpt-4o-mini",
                "timeout": 60,  # seconds
                "max_output_tokens": 500,
                "temperature": 0.2,
                "top_p": 0.98,
                "system_message": """You are an expert AI language model specialized in finding bugs and feature requests from customer reviews.""",
                "text_format": BugsAndFeatures,
            },
            "output_json_format": """{"bugs": list[string], "feature_requests": list[string]}""",
            "output_pydantic_json_schema": BugsAndFeatures,
        },
    },
    "recommend_actions": {
        "openai": {
            "llm_input_args": {
                "model": "gpt-4o-mini",
                "timeout": 60,  # seconds
                "max_output_tokens": 300,
                "temperature": 0.3,
                "top_p": 0.98,
                "system_message": """You are an expert AI language model specialized in recommending actions from customer reviews.""",
                "text_format": RecommendActions,
            },
            "output_json_format": """{"recommended_actions": list[string]}""",
            "output_pydantic_json_schema": RecommendActions,
            "max_num_of_actions": 10,  # Maximum number of actions to recommend
        },
    },
    "marketing_strategies": {
        "openai": {
            "llm_input_args": {
                "model": "gpt-4o-mini",
                "timeout": 60,  # seconds
                "max_output_tokens": 300,
                "temperature": 0.5,
                "top_p": 0.98,
                "system_message": """You are an expert AI language model specialized in recommending marketing strategies for customer reviews.""",
                "text_format": MarketingStrategies,
            },
            "output_json_format": """{"marketing_strategies": list[string]}""",
            "output_pydantic_json_schema": MarketingStrategies,
            "max_num_of_marketing_strategies": 10,  # Maximum number of marketing strategies to recommend
        },
    },
    "extract_subtopics": {
        "openai": {
            "llm_input_args": {
                "model": "gpt-4o-mini",
                "timeout": 60,  # seconds
                "max_output_tokens": 50,
                "temperature": 0.0,
                "top_p": 0.95,
                "system_message": """You are an expert AI language model specialized in extracting concise, high-quality sub-topics from customer reviews to give meaningfull insights to the app owner. Always respond with English language.""",
                "text_format": ExtractSubTopics,
            },
            "output_json_format": """{"subtopics": list[string]}""",
            "output_pydantic_json_schema": ExtractSubTopics,
        }
    },
}
