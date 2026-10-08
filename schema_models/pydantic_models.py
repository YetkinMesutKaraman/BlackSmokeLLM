from decimal import Decimal

import pandas as pd
from pydantic import BaseModel, Field


# file_path,
# backend_provider: str = "openai",
# model_name: str = "gpt-4o-mini",
# task_type: str = "create_reviews_summary",
### Create Reviews Summary endpoint ###
# input data for reviews summary, we need to provide number of reviews, input data for LLM and model name
class ReviewsSummaryRequest(BaseModel):
    file_path: str = Field(..., description="The path to the file containing reviews.")
    provider_name: str = Field(..., description="The provider for the LLM API.")
    llm_model_name: str = Field(
        ..., description="The model name to use for the given provider and task"
    )


# output schema for reviews summary
class ReviewsSummary(BaseModel):
    # it should be 3 paragraphs
    summary_text: list[str] = Field(
        ...,
        min_length=2,
        max_length=3,
        description="List of paragraphs extracted from customer reviews for summary text.",
    )


# output json payload for reviews summary
class ReviewsSummaryResponse(BaseModel):
    task_type: str = Field(..., description="The type of LLM task.")
    llm_backend_provider: str = Field(
        ..., description="Which provider is used for LLM API."
    )
    llm_model_name: str = Field(..., description="The LLM model name.")
    num_of_reviews: int = Field(..., description="The number of reviews")
    is_valid_output: int = Field(
        ...,
        description="Flag indicating if the output is valid. 1 for valid, 0 for invalid.",
    )
    total_count_of_api: int = Field(
        ..., description="Total count of API requests made to LLM."
    )
    validation_attempts: int = Field(
        ..., description="Number of validation attempts made."
    )
    latency_of_llm_response: float = Field(
        ..., description="Latency of the initial LLM response in seconds."
    )
    latency_of_validation: float = Field(
        ..., description="Latency of the output schema validation in seconds."
    )
    input_tokens: int = Field(
        ..., description="True input tokens used for the LLM request."
    )
    output_tokens: int = Field(
        ..., description="Output tokens used for the LLM request."
    )
    cost_of_run: float = Field(
        ..., description="Cost of running the LLM model based on tokens."
    )
    valid_json_output: str = Field(
        ...,
        description="The valid JSON output from the LLM model that contains the reviews summary text. It is formatted for HTML friendly display.",
    )


### Extract Unsupervised Topics endpoint ###
# input data for extract unsupervised topics, same as the input data for reviews summary
class UnsupervisedTopicExtractionRequest(BaseModel):
    file_path: str = Field(..., description="The path to the file containing reviews.")
    provider_name: str = Field(..., description="The provider for the LLM API.")
    llm_model_name: str = Field(
        ..., description="The model name to use for the given provider and task"
    )


# output data schema for unsupervised topic extraction from LLM
class ReviewsUnsupervisedTopics(BaseModel):
    positive_topics: list[str] = Field(
        ..., description="List of positive topics which are extracted from the reviews."
    )
    negative_topics: list[str] = Field(
        ..., description="List of negative topics which are extracted from the reviews."
    )


# output json payload for unsupervised topic extraction
class UnsupervisedTopicExtractionResponse(BaseModel):
    task_type: str = Field(..., description="The type of LLM task.")
    llm_backend_provider: str = Field(
        ..., description="Which provider is used for LLM API."
    )
    llm_model_name: str = Field(..., description="The LLM model name.")
    num_of_reviews: int = Field(..., description="The number of reviews")
    is_valid_output: int = Field(
        ...,
        description="Flag indicating if the output is valid. 1 for valid, 0 for invalid.",
    )
    total_count_of_api: int = Field(
        ..., description="Total count of API requests made to LLM."
    )
    validation_attempts: int = Field(
        ..., description="Number of validation attempts made."
    )
    latency_of_llm_response: float = Field(
        ..., description="Latency of the initial LLM response in seconds."
    )
    latency_of_validation: float = Field(
        ..., description="Latency of the output schema validation in seconds."
    )
    input_tokens: int = Field(
        ..., description="True input tokens used for the LLM request."
    )
    output_tokens: int = Field(
        ..., description="Output tokens used for the LLM request."
    )
    cost_of_run: float = Field(
        ..., description="Cost of running the LLM model based on tokens."
    )
    valid_json_output: ReviewsUnsupervisedTopics = Field(
        ...,
        description="The valid JSON output from the LLM model for unsupervised topics extraction.",
    )


### Tag Reviews with Topics endpoint ###
# input data for tagging reviews with topics
class TagReviewsRequest(BaseModel):
    """
    Represents a request to tag reviews.

    Args:
        reviews_dataframe (pd.DataFrame): The reviews data as a pandas DataFrame which includes 'review_text' and 'review_id' columns.
        positive_topics (List[str]): List of positive topics.
        negative_topics (List[str]): List of negative topics.
        llm_model_name (str): The model name to use.
    """

    file_path: str = Field(..., description="The path to the file containing reviews.")
    provider_name: str = Field(..., description="The provider for the LLM API.")
    llm_model_name: str = Field(
        ..., description="The model name to use for the given provider and task"
    )
    app_id: int = Field(..., description="The app ID.")
    is_last_page: bool = Field(
        ..., description="Flag indicating if this is the last page of reviews."
    )
    positive_topics: list[str] = Field(..., description="List of positive topics.")
    negative_topics: list[str] = Field(..., description="List of negative topics.")

    class Config:
        arbitrary_types_allowed = True


# output data schema for tagging reviews with topics from LLM
class ReviewTags(BaseModel):
    positive_topics: list[str] = Field(
        default=[],
        title="List of positive topics.",
        description="Provide the positive topics that are relevant to the customer review.",
    )
    negative_topics: list[str] = Field(
        default=[],
        title="List of negative topics.",
        description="Provide the negative topics that are relevant to the customer review.",
    )


# individual tagging response, output json payload for a single review
class TagReviewResponse(BaseModel):
    task_type: str = Field(..., description="The type of LLM task.")
    llm_backend_provider: str = Field(
        ..., description="Which provider is used for LLM API."
    )
    llm_model_name: str = Field(..., description="The name of the LLM model.")
    review_id: str | int = Field(default=1, description="The review ID.")
    is_valid_output: int = Field(
        ...,
        description="Flag indicating if the output is valid. 1 for valid, 0 for invalid.",
    )
    total_count_of_api: int = Field(
        ..., description="Total count of API requests made to LLM."
    )
    validation_attempts: int = Field(
        ..., description="Number of validation attempts made."
    )
    latency_of_llm_response: float = Field(
        ..., description="Latency of the initial LLM response in seconds."
    )
    latency_of_validation: float = Field(
        ..., description="Latency of the output schema validation in seconds."
    )
    input_tokens: int = Field(
        ..., description="True input tokens used for the LLM request."
    )
    output_tokens: int = Field(
        ..., description="Output tokens used for the LLM request."
    )
    cost_of_run: float = Field(
        ..., description="Cost of running the LLM model based on tokens."
    )
    valid_json_output: ReviewTags = Field(
        ...,
        title="The valid JSON output from the LLM model.",
        description="Provide the positive and negative topics that are relevant to the customer review.",
    )
    not_given_topics: list[str] = Field(
        ...,
        description="List of topics that are not given but generated and used when tagging the review.",
    )
    false_positive_topics: list[str] = Field(
        ..., description="List of false positive topics that are generated for tagging."
    )
    false_negative_topics: list[str] = Field(
        ..., description="List of false negative topics that are generated for tagging."
    )


# output data for all reviews
class AllReviewsTagResponse(BaseModel):
    all_reviews: list[TagReviewResponse] = Field(
        ..., description="List of tagging responses for each customer review."
    )


# input data for update topics
class UpdateTopicsRequest(BaseModel):
    """
    The request object for detecting deprecated and emergent topics.

    Attributes:
        unsupervised_topic_extraction_request (UnsupervisedTopicExtractionRequest): The request for unsupervised topic extraction.
        df_reviews (pd.DataFrame): The reviews data as a pandas DataFrame which includes 'review_text' and 'review_id' columns.
        previous_base_topics (ReviewsUnsupervisedTopics): The previous base topics.
    """

    # this should meet the requirements of extract_unsupervised_topics_endpoint():
    # 1-) UnsuperivsedTopicExtractionRequest
    # 2-) df_reviews
    # and should have previous topics set: previous_base_positive_topics and previous_base_negative_topics
    unsupervised_topic_extraction_request: UnsupervisedTopicExtractionRequest = Field(
        ..., description="The request for unsupervised topic extraction."
    )
    df_reviews: pd.DataFrame = Field(
        ...,
        description="The reviews data as a pandas DataFrame which includes 'review_text','review_id', 'positive_topics', and 'negative_topics' columns.",
    )
    previous_base_topics: ReviewsUnsupervisedTopics = Field(
        ..., description="The previous base topics."
    )

    # define custom class for pandas dataframe
    class Config:
        arbitrary_types_allowed = True


class TagsForEmergentTopics(BaseModel):
    review_id: str | int = Field(default=None, description="The review ID.")
    valid_json_output: ReviewTags = Field(
        ..., description="The valid JSON output from the LLM model."
    )


class TopicDetails(BaseModel):
    similarity_score: float | None = Field(
        None, description="Similarity score between candidate and base topics."
    )
    n_gram_match: bool | None = Field(
        None, description="Indicates if there was an n-gram overlap."
    )
    co_occurrence_score: float | None = Field(
        None,
        title="Co-occurrence score",
        description="Optional co-occurrence score if applicable.",
    )
    adjusted_co_occurrence_score: float | None = Field(
        None,
        title="Adjusted co-occurrence score",
        description="Optional adjusted co-occurrence score if applicable.",
    )


# Consolidated topics with nested topic mappings for positive and negative topics.
class ConsolidatedTopics(BaseModel):
    positive_topics: dict[str, dict[str, TopicDetails]] = Field(
        default={},
        description="Consolidated positive topics when updating topics based on similarity check to detect emergent topics.",
    )
    negative_topics: dict[str, dict[str, TopicDetails]] = Field(
        default={},
        description="Consolidated negative topics when updating topics based on similarity check to detect emergent topics.",
    )


# output json payload for updating topics advanced
class UpdateTopicsResponse(BaseModel):
    task_type: str = Field(..., description="The type of LLM task.")
    llm_backend_provider: str = Field(
        ..., description="Which provider is used for LLM API."
    )
    llm_model_name: str = Field(..., description="The LLM model name.")
    num_of_reviews: int = Field(..., description="The number of reviews")
    deprecated_topics: ReviewTags = Field(..., description="Deprecated topics")
    emergent_topics: ReviewTags = Field(..., description="Emergent topics")
    consolidated_topics: ConsolidatedTopics = Field(
        ...,
        description="Consolidated topics when we apply similarity check to detect emergent topics.",
    )
    cost_of_run: dict[str, float | Decimal] = Field(
        ..., description="Cost of running the LLM model for this task."
    )
    latency: dict[str, float] = Field(
        ..., description="Latency of the LLM model for this task."
    )
    tags_for_emergent_topics: list[TagsForEmergentTopics] = Field(
        ..., description="List of tagging responses for emergent topics."
    )


### Find endpoint ###
# input data for find bugs and feature requests
class FindBugsAndFeaturesRequest(BaseModel):
    file_path: str = Field(..., description="The path to the file containing reviews.")
    provider_name: str = Field(..., description="The provider for the LLM API.")
    llm_model_name: str = Field(
        ..., description="The model name to use for the given provider and task"
    )


# output data schema for find bugs and feature requests
class BugsAndFeatures(BaseModel):
    bugs: list[str] = Field(..., description="List of bugs extracted from the reviews.")
    feature_requests: list[str] = Field(
        ..., description="List of feature requests extracted from the reviews."
    )


# output json payload for find bugs and feature requests
class FindBugsAndFeaturesResponse(BaseModel):
    task_type: str = Field(..., description="The type of LLM task.")
    llm_backend_provider: str = Field(
        ..., description="Which provider is used for LLM API."
    )
    llm_model_name: str = Field(..., description="The LLM model name.")
    num_of_reviews: int = Field(..., description="The number of reviews")
    is_valid_output: int = Field(
        ...,
        description="Flag indicating if the output is valid. 1 for valid, 0 for invalid.",
    )
    total_count_of_api: int = Field(
        ..., description="Total count of API requests made to LLM."
    )
    validation_attempts: int = Field(
        ..., description="Number of validation attempts made."
    )
    latency_of_llm_response: float = Field(
        ..., description="Latency of the initial LLM response in seconds."
    )
    latency_of_validation: float = Field(
        ..., description="Latency of the output schema validation in seconds."
    )
    input_tokens: int = Field(
        ..., description="True input tokens used for the LLM request."
    )
    output_tokens: int = Field(
        ..., description="Output tokens used for the LLM request."
    )
    cost_of_run: float = Field(
        ..., description="Cost of running the LLM model based on tokens."
    )
    valid_json_output: BugsAndFeatures = Field(
        ...,
        description="The valid JSON output from the LLM model for bugs and feature requests extraction.",
    )


### Recommend Actions endpoint ###
# input data for recommend actions
class RecommendActionsRequest(BaseModel):
    file_path: str = Field(..., description="The path to the file containing reviews.")
    provider_name: str = Field(..., description="The provider for the LLM API.")
    llm_model_name: str = Field(
        ..., description="The model name to use for the given provider and task"
    )
    topic_type: str = Field(
        ..., description="The type of topic to recommend actions for."
    )
    topic_name: str = Field(..., description="The topic name to recommend actions for.")


# output data schema for recommend actions
class RecommendActions(BaseModel):
    recommended_actions: list[str] = Field(
        ..., description="List of actions to recommend for the reviews."
    )


# output data schema for recommend marketing strategies
class MarketingStrategies(BaseModel):
    marketing_strategies: list[str] = Field(
        ..., description="List of marketing strategies from the reviews."
    )


# output json payload for recommend actions
class RecommendActionsResponse(BaseModel):
    task_type: str = Field(..., description="The type of LLM task.")
    llm_backend_provider: str = Field(
        ..., description="Which provider is used for LLM API."
    )
    llm_model_name: str = Field(..., description="The LLM model name.")
    num_of_reviews: int = Field(..., description="The number of reviews")
    is_valid_output: int = Field(
        ...,
        description="Flag indicating if the output is valid. 1 for valid, 0 for invalid.",
    )
    total_count_of_api: int = Field(
        ..., description="Total count of API requests made to LLM."
    )
    validation_attempts: int = Field(
        ..., description="Number of validation attempts made."
    )
    latency_of_llm_response: float = Field(
        ..., description="Latency of the initial LLM response in seconds."
    )
    latency_of_validation: float = Field(
        ..., description="Latency of the output schema validation in seconds."
    )
    input_tokens: int = Field(
        ..., description="True input tokens used for the LLM request."
    )
    output_tokens: int = Field(
        ..., description="Output tokens used for the LLM request."
    )
    cost_of_run: float = Field(
        ..., description="Cost of running the LLM model based on tokens."
    )
    valid_json_output: RecommendActions = Field(
        ...,
        description="The valid JSON output from the LLM model for recommend actions.",
    )


# output json payload for recommend actions
class MarketingStrategiesResponse(BaseModel):
    task_type: str = Field(..., description="The type of LLM task.")
    llm_backend_provider: str = Field(
        ..., description="Which provider is used for LLM API."
    )
    llm_model_name: str = Field(..., description="The LLM model name.")
    num_of_reviews: int = Field(..., description="The number of reviews")
    is_valid_output: int = Field(
        ...,
        description="Flag indicating if the output is valid. 1 for valid, 0 for invalid.",
    )
    total_count_of_api: int = Field(
        ..., description="Total count of API requests made to LLM."
    )
    validation_attempts: int = Field(
        ..., description="Number of validation attempts made."
    )
    latency_of_llm_response: float = Field(
        ..., description="Latency of the initial LLM response in seconds."
    )
    latency_of_validation: float = Field(
        ..., description="Latency of the output schema validation in seconds."
    )
    input_tokens: int = Field(
        ..., description="True input tokens used for the LLM request."
    )
    output_tokens: int = Field(
        ..., description="Output tokens used for the LLM request."
    )
    cost_of_run: float = Field(
        ..., description="Cost of running the LLM model based on tokens."
    )
    valid_json_output: MarketingStrategies = Field(
        ...,
        description="The valid JSON output from the LLM model for recommend marketing strategies.",
    )


### Extract Subtopics endpoint ###
# input data for extract subtopics
class ExtractSubTopicsRequest(BaseModel):
    file_path: str = Field(..., description="The path to the file containing reviews.")
    provider_name: str = Field(..., description="The provider for the LLM API.")
    llm_model_name: str = Field(
        ..., description="The model name to use for the given provider and task"
    )
    topic_type: str = Field(
        ..., description="The type of topic to recommend actions for."
    )
    topic_name: str = Field(..., description="The topic name to recommend actions for.")


# output data schema for recommend actions
class ExtractSubTopics(BaseModel):
    subtopics: list[str] = Field(..., description="List of subtopic phrases.")


# output json payload for subtopic extraction
class ExtractSubTopicsResponse(BaseModel):
    task_type: str = Field(..., description="The type of LLM task.")
    llm_backend_provider: str = Field(
        ..., description="Which provider is used for LLM API."
    )
    llm_model_name: str = Field(..., description="The LLM model name.")
    num_of_reviews: int = Field(..., description="The number of reviews")
    is_valid_output: int = Field(
        ...,
        description="Flag indicating if the output is valid. 1 for valid, 0 for invalid.",
    )
    total_count_of_api: int = Field(
        ..., description="Total count of API requests made to LLM."
    )
    validation_attempts: int = Field(
        ..., description="Number of validation attempts made."
    )
    latency_of_llm_response: float = Field(
        ..., description="Latency of the initial LLM response in seconds."
    )
    latency_of_validation: float = Field(
        ..., description="Latency of the output schema validation in seconds."
    )
    input_tokens: int = Field(
        ..., description="True input tokens used for the LLM request."
    )
    output_tokens: int = Field(
        ..., description="Output tokens used for the LLM request."
    )
    cost_of_run: float = Field(
        ..., description="Cost of running the LLM model based on tokens."
    )
    valid_json_output: ExtractSubTopics = Field(
        ...,
        description="The valid JSON output from the LLM model for unsupervised topics extraction.",
    )
