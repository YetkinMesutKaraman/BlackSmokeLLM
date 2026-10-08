# Description: This module contains functions to prepare input arguments for the LLM model.


def create_user_prompt(
    task_type: str,
    output_json_format: str,
    delimiter: str = "##",
    num_of_reviews: int = None,
    input_data_for_llm: str = None,
    customer_review: str = None,
    current_positive_topics: list = None,
    current_negative_topics: list = None,
    empty_json_format: str = None,
    topic_type: str = None,
    topic_name: str = None,
) -> str:
    """
    Create user prompt for the LLM model.

    Args:
        task_type (str): The type of task to perform. Valid options are:
            'create_reviews_summary', 'extract_unsupervised_topics', 'update_topics', 'tag_reviews_with_topics'.
        output_json_format (str): The desired format of the output JSON.
        delimiter (str, optional): The delimiter used to separate the customer review. Required for 'tag_reviews_with_topics' task.
            Defaults to "##".
        num_of_reviews (int, optional): The number of reviews. Required for 'create_reviews_summary',
            'extract_unsupervised_topics', and 'recommend_actions' tasks. Defaults to None.
        input_data_for_llm (str, optional): The input data for the LLM model. Required for
            'create_reviews_summary', 'extract_unsupervised_topics', and 'recommend_actions' tasks. Defaults to None.
        customer_review (str, optional): The customer review to tag. Required for 'tag_reviews_with_topics' task.
            Defaults to None.
        positive_topics (list, optional): The list of positive topics. Required for 'tag_reviews_with_topics' task.
            Defaults to None.
        num_of_reviews (int, optional): The number of reviews. Required for 'create_reviews_summary' and
            'extract_unsupervised_topics', and 'recommend_actions' tasks. Defaults to None.
        input_data_for_llm (str, optional): The input data for the LLM model. Required for
            'create_reviews_summary', 'extract_unsupervised_topics', and 'recommend_actions' tasks. Defaults to None.
        customer_review (str, optional): The customer review to tag. Required for 'tag_reviews_with_topics' task.
            Defaults to None.
        positive_topics (list, optional): The list of positive topics. Required for 'tag_reviews_with_topics' task.
            Defaults to None.
        negative_topics (list, optional): The list of negative topics. Required for 'tag_reviews_with_topics' task.
            Defaults to None.
        empty_json_format (str, optional): The format of the empty JSON. Required for 'tag_reviews_with_topics' task.
            Defaults to None.
        delimiter (str, optional): The delimiter used to separate the customer review. Required for 'tag_reviews_with_topics' task.
            Defaults to "##".
        topic_type (str, optional): The type of topic to recommend actions for. Required for 'recommend_actions' task.
            Defaults to None.
        topic_name (str, optional): The name of the topic to recommend actions for. Required for 'recommend_actions' task.
            Defaults to None.
    Returns:
        str: The user prompt message.

    Raises:
        ValueError: If an invalid task type is provided.

    """

    if task_type == "create_reviews_summary":
        user_message = f"""I am providing you a dataset of user reviews for a Mobile Application. \
Each review is placed in between two {delimiter} characters. There are {num_of_reviews} reviews. \
Using all the {num_of_reviews} user reviews, create always an English summary text of the reviews with 3 paragraphs. \
Must use "**" as a delimiter to highlight the key aspects in the summary text. \
Return the output as a JSON dict with the following format: {output_json_format}. \
Here is the data: \n{input_data_for_llm}\n."""

    elif task_type == "extract_unsupervised_topics":
        user_message = f""" \
You are provided with {num_of_reviews} user reviews for a Mobile Application. \
Each review is enclosed between {delimiter} characters. \
# Task: \
- Identify **positive** and **negative** topic phrases from the reviews. \

# Important Notes: \
- Do not use same prefix because it is not good for user experience. \
- Do not use "and" or "or" in topic phrases. \
- If a topic phrase contains "and" or "or", split them into separate topics if they are not synonymous. \
- Consolidate synonymous or closely related topic phrases to provide a cleaner output. \
- Provide semantically diverse topics. \
- Provide high coverage of mentions in the reviews. Thus, provide high coverage of reviews for the given app reviews.
- Topic phrases must not be longer than 5 words. \

# Output Format: \
- Always return English topic phrases. \
- Return a json object with the following format: {output_json_format}. \

# Data: 
\n{input_data_for_llm}\n
"""

    elif task_type == "tag_reviews_with_topics":
        user_message = f"""You are a world leading expert trained AI Language model specialized on labelling user reviews with given topics. Your strongest asset is your ability to classify reviews into the most relevant topics.

# Task: \
- A review about a mobile application will be provided to you. \
- Your task is to analyze the review and tag it with the most relevant topics from the provided schema.

# Steps: \
Let's think and work step by step. \

1. **Understand the review content:** 
   - Read the user review carefully.
2. **Identify relevant topics:**  
   - First: Classify this user review according to the provided schema's positive topics.
   - Second: Classify this user review according to the provided schema's negative topics.
3. **Apply relevance criteria:**
   - Only use the topics from the provided schema. Do not hallucinate topics.
   - First assign the obvious topics that are explicitly mentioned in the review.
   - Some mentions may be vague or ambiguous, so you need to interpret the review content and choose the most relevant topics.
   - If no topics are relevant, return an empty JSON in the following format: {empty_json_format}.
4. **Detect potential incorrect labels:**
   - Check for any topics that may not fit the context of the review.
   - Length of the review and number of topics should be considered.
   - Ensure the number of topics is reasonable (e.g., not too many for a short review).
   - Remove any topics that do not align with the review's content.
5. **Detect potential missing topics:**
    - Look for any relevant topics that may have been overlooked.
    - Ensure all relevant topics are included.
    - Ensure the number of topics is reasonable (e.g., not too few for a long review).
    - Add any missing topics that are relevant to the review.
6. **Double check labels:**  
   - Double check for false positive labels (topics incorrectly assigned) and false negative labels (relevant topics missed).  
   - Make corrections to ensure accurate labeling.

# Quality Assurance Guidelines:

**Edge Cases to Handle:**
- Short reviews (1-2 sentences): Assign only topics explicitly mentioned, typically 0-2 topics.
- Long reviews (5+ sentences): May have multiple topics, but still require explicit evidence for each.
- Mixed sentiment: Carefully separate positive and negative aspects, ensure topics match correct sentiment.


**Common Pitfalls to Avoid:**
- Don't assign opposite sentiment topics (e.g., both "fast_process" and "slow_process").
- Don't over-tag short reviews with many topics.
- Don't under-tag detailed reviews that mention multiple clear issues/praises.
- Don't assign generic topics when specific ones are more appropriate.
- Don't tag unrelated topics just because they appear in the topic list.

# Important Note:
- It is crucial to provide accurate topic labels to the user because it willl be used for further analysis and decision making.

# Examples for Calibration:
**Example 1 (Specific Positive Review):**
    *Input Review:* "I'm so impressed with this app! Listing my vintage dresses was incredibly straightforward, and I made my first sale within 24 hours. The interface is intuitive, and I love that I don't get charged a commission on my sales. Also, the customer support team was very helpful when I had a question about shipping."
    *Correct Output:* {{"positive_topics": ["easy_product_listing", "fast_sales_process", "user-friendly_interface", "no_seller_commission", "good_customer_support"], "negative_topics": []}}

**Example 2 (Mixed Review):**
    *Input Review:* I love how easy it is to list items and the interface is clean. However, my last payment took over a week to show up in my bank account, and the buyer protection fee seems excessively high now.
    *Correct Output:* {{"positive_topics": ["easy_product_listing", "user-friendly_interface"], "negative_topics": ["delayed_payment_transfers", "high_buyer_protection_fees"]}}

**Example 3 (Specific Negative Review):**
    *Input Review:* "This app is unusable. It freezes and closes every time I try to upload photos for my listing. I can't even get past that step."
    *Correct Output:* {{"positive_topics": [], "negative_topics": ["app_crashes_frequently", "inadequate_photo_handling"]}}

**Example 4 (Vague/No Topics):**
    *Input Review:* It's an okay app. Does the job.
    *Correct Output:* {{"positive_topics": [], "negative_topics": []}}

# Input Review:
{customer_review}
"""

    elif task_type == "find_bugs_and_features":
        user_message = f""" \
You are provided with {num_of_reviews} user reviews for a Mobile Application. \
Each review is enclosed between {delimiter} characters. \
# Task: \
- Identify **feature requests** and **bugs** from the reviews. \

# Output Format: \
- Always return English phrases. \
- Return a json object with the following format: {output_json_format}. 

# Data:
\n{input_data_for_llm}\n
"""

    elif task_type == "recommend_actions":
        user_message = f"""
You are provided with {num_of_reviews} user reviews for a Mobile Application. \
Each review is enclosed between {delimiter} characters. \
# Task: \
- Recommend compact set of distinct actions to solve the issues in the reviews that is strictly related to the main topic. \

# Steps: \
Let's think and work step by step. \

1. **Understand the reviews:** \
    - Read the reviews carefully. \
2. **Detect issues:** \
    - Detect issues in the reviews that is related to the main topic, sort them by frequency. \
3. **Recommend actions:** \
    - Ensure the actions must be relevant to the main topic: {topic_name}. \
    - Ensure the actions are to the point, specific and concise. \
   
# Output Format: \
- Always return English texts. \
- Return a json object with the following format: {output_json_format}. \

# Data: \
\n{input_data_for_llm}\n
"""

    elif task_type == "recommend_marketing_strategies":
        user_message = f"""
    You are provided with {num_of_reviews} user reviews for a Mobile Application. \
    Each review is enclosed between {delimiter} characters. \
    # Task: \
    - Recommend compact set of distinct marketing strategies to improve the revenue. \

    # Steps: \
    Let's think and work step by step. \

    1. **Understand the reviews:** \
        - Read the reviews carefully. \
    2. **Detect Potential Marketing Strategies:** \
        - Detect potential marketing strategies to improve the revenue. \
    3. **Recommend marketing strategies:** \
        - Ensure the marketing strategies must be relevant to the main topic: {topic_name}. \
        - Ensure the marketing strategies are to the point, creative,specific and concise. \

    # Output Format: \
    - Always return English texts. \
    - Return a json object with the following format: {output_json_format}. \

    # Data: \
    \n{input_data_for_llm}\n
    """

    elif "extract_subtopics":
        user_message = f"""You are given {num_of_reviews} user reviews. Each review is delimited by the token {delimiter}.

ROLE
- Extract concise sub-topic PHRASES strictly relevant to a single main topic.
- Main topic context: topic_type = "{topic_type}", topic_name = "{topic_name}".
- IMPORTANT: All sub-topics MUST be the SAME TYPE as the main topic (i.e., inherit topic_type = "{topic_type}"). Do NOT propose subtopics that belong to any other type.

OUTPUT
- Return ONLY a JSON object matching exactly: {output_json_format}
- If no valid sub-topics exist, return exactly: []
- Use double quotes for all JSON strings.

RULES
- At most 5 unique, insightful sub-topic phrases.
- Each phrase ≤ 5 words.
- Sub-topics must be strictly relevant to "{topic_name}" AND belong to topic_type "{topic_type}".
- Do NOT repeat the same leading prefix among phrases (e.g., avoid "slow loading", "slow login").
- No duplicates or near-duplicates (plural/singular, tense variants, trivial rewordings).
- Avoid vague/general terms (“bad experience”, “UI issues”) unless they directly and specifically qualify the main topic within the SAME TYPE.

DECISION CRITERIA (apply before output)
- Keep a candidate only if ALL are true:
  1) Type-consistent: belongs to topic_type "{topic_type}"
  2) Specific to "{topic_name}" (not generic app/platform noise)
  3) Appears in multiple reviews OR is highly actionable/impactful
  4) Adds a distinct angle vs. other selected phrases
- If fewer than 1 phrase meets all criteria, output [].

FORMAT EXAMPLES (style only; do NOT include in output)
- GOOD (same type as main): ["refund_policy_confusion", "delayed_payouts", "ID_verification_loops"]
- BAD (wrong type or too generic): ["UI issues", "slow", "customer_support_polite"]

DATA
{input_data_for_llm}

INSTRUCTIONS
- Think through selection privately. Output only the final JSON (or [] if none) in the exact required format.
"""

    else:
        raise ValueError(
            "Invalid task type. Please choose from 'create_reviews_summary', 'extract_unsupervised_topics', 'update_topics', 'tag_reviews_with_topics', 'update_topics_advanced'."
        )

    return user_message


# # Steps: \
# Let's think and work step by step. \
# 1. **Extract the sub-topic phrases**
#     - Ensure sub-topics must be relevant to the main topic: {topic_name}. \
# 2. **Prefix Avoidance**
#     - Do not use same prefix because it is not good for user experience. \
# 3. **Not merge complements**
#     - Do not use "and" or "or" in topic phrases. \
# 4. **Decompose**
#     - If a topic phrase contains "and" or "or", split them into separate topics if they are not synonymous. \
# 5. **Return Distinct**
#     - Consolidate synonymous or closely related topic phrases to provide a cleaner output. \
# 6. **High Coverage**
#     - Provide semantically diverse topics. \
# 7. **Limit number of words**
#     - Topic phrases must not be longer than 5 words. \


def prepare_llm_input_args(user_message: str, llm_input_args_config: dict) -> dict:
    """Prepares the input arguments for the LLM model.

    Args:
        user_message (str): The user's message.
        llm_input_args_config (dict): Configuration for the LLM input arguments.
                                     Expected to contain: model, timeout, max_completion_tokens,
                                     temperature, text_format, system_message.

    Returns:
        dict: The prepared LLM input arguments with messages added.
    """
    # Create a copy to avoid mutating the original config
    llm_input_args = llm_input_args_config.copy()

    # Add the messages to the configuration
    llm_input_args["input"] = [
        {
            "role": "system",
            "content": llm_input_args_config["system_message"],
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]

    # Remove system_message from the final args since it's now in messages
    llm_input_args.pop("system_message", None)

    return llm_input_args
