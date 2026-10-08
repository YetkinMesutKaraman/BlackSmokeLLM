"""
Gemini 3.0 Flash Integration for Review Tagging

This module demonstrates native Google GenAI SDK integration for tagging reviews
with topics. It serves as a proof-of-concept before full provider integration.

Key Features:
- Native google.genai SDK usage
- Async batch processing with controlled concurrency
- Prompt caching for cost optimization
- Type-safe configuration and responses
- Dynamic schema generation with Literal types for constrained outputs
- Comprehensive error handling and logging
"""

import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from time import perf_counter
from typing import Any, List, Literal, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError, create_model

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_providers.openai_client.data_prep_for_llm import filter_reviews_data
from llm_providers.openai_client.prep_input_args_for_llm import create_user_prompt
from schema_models.pydantic_models import ReviewTags
from utils.task_preps import load_reviews_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)


# ============================================================================
# Dynamic Schema Generation
# ============================================================================


def create_classification_model_with_literal(
    positive_topics: List[str],
    negative_topics: List[str],
) -> type[BaseModel]:
    """
    Create dynamic Pydantic model with Literal type hints for constrained outputs.

    This approach ensures the LLM can only output topics from the provided lists,
    improving type safety and reducing hallucination of invalid topics.

    Best for: Gemini/OpenAI/Anthropic API structured outputs (JSON mode)
    Pros:
    - Works perfectly with LLM APIs
    - Cleaner JSON schema
    - Compile-time type checking
    - Prevents invalid topic generation
    - Better IDE autocomplete support

    Args:
        positive_topics: List of valid positive topic strings
        negative_topics: List of valid negative topic strings

    Returns:
        Dynamic Pydantic model class with Literal-constrained fields

    Example:
        >>> positive = ["fast", "reliable", "cheap"]
        >>> negative = ["slow", "buggy", "expensive"]
        >>> Model = create_classification_model_with_literal(positive, negative)
        >>> instance = Model(positive_categories=["fast"], negative_categories=["buggy"])
    """
    # Create Literal types from topics - must be tuple for Literal
    PositiveLiteral = Literal[tuple[str, ...](positive_topics)]
    NegativeLiteral = Literal[tuple[str, ...](negative_topics)]

    # Build dynamic model with constrained fields
    DynamicClassificationModel = create_model(
        "ReviewClassification",
        __doc__="Dynamically generated classification model with topic constraints",
        positive_topics=(
            List[PositiveLiteral],
            Field(
                default=[],
                description=f"The predicted positive topic labels for the review.",
                title="Positive Topics",
            ),
        ),
        negative_topics=(
            List[NegativeLiteral],
            Field(
                default=[],
                description=f"The predicted negative topic labels for the review.",
                title="Negative Topics",
            ),
        ),
    )

    return DynamicClassificationModel


# ============================================================================
# Configuration Dataclasses
# ============================================================================


@dataclass
class GeminiConfig:
    """Configuration for Gemini 3.0 Flash model."""

    model_name: str = "gemini-3-flash-preview"
    max_output_tokens: int = 250
    temperature: float = 0.0
    thinking_level: str = "minimal"
    response_mime_type: str = "application/json"
    cache_ttl_seconds: int = 30  # 1 hour


@dataclass
class BatchConfig:
    """Configuration for batch processing."""

    batch_size: int = 1000
    max_concurrent_requests: int = 1000
    delay_between_batches_seconds: float = 0


@dataclass
class ProcessingMetrics:
    """Metrics for processing performance tracking."""

    total_reviews: int = 0
    successful: int = 0
    failed: int = 0
    total_duration: float = 0.0
    cached_input_tokens: int = 0
    input_tokens: int = 0
    thinking_tokens: int = 0
    output_tokens: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        total = self.successful + self.failed
        return (self.successful / total * 100) if total > 0 else 0.0

    @property
    def reviews_per_second(self) -> float:
        """Calculate processing throughput."""
        return (
            self.total_reviews / self.total_duration if self.total_duration > 0 else 0.0
        )


# ============================================================================
# Gemini Client Wrapper
# ============================================================================


class GeminiClient:
    """
    Wrapper for Google GenAI SDK with caching and async support.

    This class provides a clean interface for Gemini API interactions,
    handling client initialization, caching, and response standardization.
    Supports dynamic schema generation for constrained topic outputs.
    """

    def __init__(
        self,
        api_key: str,
        config: GeminiConfig,
        response_schema: Optional[type[BaseModel]] = None,
    ):
        """
        Initialize Gemini client with configuration.

        Args:
            api_key: Google API key
            config: Gemini configuration
            response_schema: Optional dynamic Pydantic model for response validation.
                           If None, defaults to ReviewTags.
        """
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required")

        self.config = config
        self._client = genai.Client(api_key=api_key)
        self._async_client = self._client.aio
        self._cache: Optional[Any] = None
        self._response_schema = response_schema or ReviewTags

        logger.info(f"Initialized Gemini client with model: {config.model_name}")
        logger.info(f"Using response schema: {self._response_schema.__name__}")

    def create_cache(
        self, system_instruction: str, cache_name: str = "review_tagging"
    ) -> Any:
        """
        Create or update prompt cache for system instruction.

        Caching the system instruction significantly reduces costs by charging
        cache tokens only once instead of per request.

        Args:
            system_instruction: System prompt to cache
            cache_name: Display name for the cache

        Returns:
            Cache object
        """
        try:
            self._cache = self._client.caches.create(
                model=self.config.model_name,
                config=types.CreateCachedContentConfig(
                    display_name=cache_name,
                    system_instruction=system_instruction,
                    ttl=f"{self.config.cache_ttl_seconds}s",
                ),
            )
            logger.info(f"Created prompt cache: {cache_name}")
            return self._cache
        except Exception as e:
            logger.error(f"Failed to create cache: {e}")
            raise

    def _build_generation_config(self, use_cache: bool = True) -> dict[str, Any]:
        """
        Build generation configuration for API calls.

        Args:
            use_cache: Whether to use cached content

        Returns:
            Configuration dict
        """
        config = {
            "max_output_tokens": self.config.max_output_tokens,
            "temperature": self.config.temperature,
            "response_mime_type": self.config.response_mime_type,
            "response_json_schema": self._response_schema.model_json_schema(),  # Use dynamic schema
            "thinking_config": types.ThinkingConfig(
                thinking_level=self.config.thinking_level
            ),
        }

        if use_cache and self._cache is not None:
            config["cached_content"] = self._cache.name

        return config

    async def generate_async(
        self,
        user_content: str,
        system_instruction: Optional[str] = None,
    ) -> types.GenerateContentResponse:
        """
        Generate content asynchronously using Gemini.

        Args:
            user_content: User message/review to process
            system_instruction: System instruction (if not using cache)

        Returns:
            Gemini response

        Raises:
            Exception: If generation fails
        """
        config_kwargs = self._build_generation_config(use_cache=self._cache is not None)

        # Add system instruction if not using cache
        if self._cache is None and system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        try:
            response = await self._async_client.models.generate_content(
                model=self.config.model_name,
                contents=[user_content],
                config=types.GenerateContentConfig(**config_kwargs),
            )
            return response
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise


# ============================================================================
# Review Processing Pipeline
# ============================================================================


class ReviewTaggingPipeline:
    """
    Pipeline for batch processing review tagging with Gemini.

    Handles data loading, batch processing, validation, and metrics collection.
    """

    def __init__(
        self,
        gemini_client: GeminiClient,
        batch_config: BatchConfig,
    ):
        """
        Initialize tagging pipeline.

        Args:
            gemini_client: Configured Gemini client
            batch_config: Batch processing configuration
        """
        self.client = gemini_client
        self.batch_config = batch_config
        self.metrics = ProcessingMetrics()

    def _create_system_instruction(
        self,
        positive_topics: list[str],
        negative_topics: list[str],
    ) -> str:
        """
        Create system instruction with topics embedded.

        Note: Gemini requires minimum 1024 tokens for caching. We add additional
        context to meet this requirement while enhancing classification quality.

        Args:
            positive_topics: List of positive topic labels
            negative_topics: List of negative topic labels

        Returns:
            Formatted system instruction (>= 1024 tokens)
        """
        # Use the shared create_user_prompt from OpenAI client
        # This ensures consistency across providers
        empty_json_format = '{"positive_topics": [], "negative_topics": []}'
        output_json_format = (
            '{"positive_topics": list[string], "negative_topics": list[string]}'
        )

        user_prompt = create_user_prompt(
            task_type="tag_reviews_with_topics",
            customer_review="{review_placeholder}",  # Placeholder for actual review
            current_positive_topics=positive_topics,
            current_negative_topics=negative_topics,
            output_json_format=output_json_format,
            empty_json_format=empty_json_format,
        )

        # Remove the placeholder and keep just the instruction part
        # The actual review will be passed as user content
        instruction = user_prompt.split("# Input Review:")[0].strip()

        return f"{instruction}"

    async def _process_single_review(
        self,
        review: str,
        semaphore: asyncio.Semaphore,
    ) -> Optional[dict[str, Any]]:
        """
        Process a single review with rate limiting.

        Uses the shared dynamic schema (self.client._response_schema) that was
        created once in process_batch(). This ensures consistent validation and
        prevents schema recreation overhead.

        Args:
            review: Review text to process
            semaphore: Semaphore for concurrency control

        Returns:
            Parsed ReviewTags dict (normalized format) or None if failed
        """
        async with semaphore:
            try:
                # Uses self.client._response_schema (set once in process_batch)
                response = await self.client.generate_async(user_content=review)
                print(f"Response: {response}")

                # Validate response
                if response and response.parsed:
                    validated = response.parsed

                    # Track metrics
                    if hasattr(response, "usage_metadata"):
                        self.metrics.cached_input_tokens += getattr(
                            response.usage_metadata, "cached_content_token_count", 0
                        )
                        self.metrics.input_tokens += getattr(
                            response.usage_metadata, "prompt_token_count", 0
                        )
                        self.metrics.thinking_tokens += getattr(
                            response.usage_metadata, "thinking_token_count", 0
                        )
                        self.metrics.output_tokens += getattr(
                            response.usage_metadata, "candidates_token_count", 0
                        )

                    self.metrics.successful += 1
                    return validated
                else:
                    logger.warning("Response missing parsed content")
                    self.metrics.failed += 1
                    return None

            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                self.metrics.failed += 1
                return None
            except Exception as e:
                logger.error(f"Processing error: {e}")
                self.metrics.failed += 1
                return None

    async def process_batch(
        self,
        reviews: list[str],
        positive_topics: list[str],
        negative_topics: list[str],
    ) -> list[dict[str, Any]]:
        """
        Process a batch of reviews with controlled concurrency.

        Creates a dynamic schema with Literal types to constrain outputs to valid topics,
        then processes all reviews using the constrained schema for better accuracy.

        Args:
            reviews: List of review texts
            positive_topics: Positive topic labels
            negative_topics: Negative topic labels

        Returns:
            List of parsed results (None entries for failures)
        """
        start_time = perf_counter()
        self.metrics.total_reviews = len(reviews)

        # ═══════════════════════════════════════════════════════════════
        # ONE-TIME SETUP (Schema + Cache) - Shared across all reviews
        # ═══════════════════════════════════════════════════════════════

        # Create dynamic schema ONCE with Literal types for constrained outputs
        # This schema will be reused for ALL reviews in this batch
        logger.info("Creating dynamic schema with topic constraints...")
        dynamic_schema = create_classification_model_with_literal(
            positive_topics, negative_topics
        )

        # Update client's response schema ONCE
        # All subsequent API calls will use this constrained schema
        self.client._response_schema = dynamic_schema
        logger.info(
            f"✓ Dynamic schema created: {len(positive_topics)} positive, "
            f"{len(negative_topics)} negative topics (reused for all reviews)"
        )

        # Create prompt cache ONCE with system instruction
        # Cache reduces cost by ~99% on repeated system instruction tokens
        system_instruction = self._create_system_instruction(
            positive_topics, negative_topics
        )
        self.client.create_cache(system_instruction)
        logger.info("✓ Prompt cache created (reused for all reviews)")

        # ═══════════════════════════════════════════════════════════════
        # BATCH PROCESSING - All reviews use same schema + cache
        # ═══════════════════════════════════════════════════════════════

        logger.info(f"Processing {len(reviews)} reviews with shared schema + cache...")

        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(self.batch_config.max_concurrent_requests)

        # Process all reviews concurrently with rate limiting
        # Each review uses the SAME dynamic_schema (via self.client._response_schema)
        tasks = [self._process_single_review(review, semaphore) for review in reviews]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        # Calculate metrics
        self.metrics.total_duration = perf_counter() - start_time

        logger.info(
            f"Batch completed: {self.metrics.successful} successful, "
            f"{self.metrics.failed} failed in {self.metrics.total_duration:.2f}s"
        )
        logger.info(f"Throughput: {self.metrics.reviews_per_second:.2f} reviews/sec")
        logger.info(f"Success rate: {self.metrics.success_rate:.1f}%")

        return results


# ============================================================================
# Main Execution
# ============================================================================


async def main() -> None:
    """
    Main execution function demonstrating Gemini integration with dynamic schemas.

    This function showcases:
    - Dynamic Pydantic model generation with Literal type constraints
    - Prompt caching for cost optimization (>1024 tokens)
    - Async batch processing with controlled concurrency
    - Comprehensive metrics tracking
    """

    # Configuration
    gemini_config = GeminiConfig(
        model_name="gemini-3-flash-preview",
        max_output_tokens=100,
        temperature=1.0,
        thinking_level="minimal",
    )

    batch_config = BatchConfig(
        batch_size=50,
        max_concurrent_requests=20,
        delay_between_batches_seconds=1.0,
    )

    # Sample topics (these would come from your database in production)
    topics = {
        "positive_topics": [
            "user-friendly_interface",
            "no_commission_fees",
            "fast_customer_support",
            "wide_product_variety",
            "easy_to_use",
            "affordable_prices",
            "secure_transactions",
            "good_for_selling",
            "helpful_customer_service",
            "great_for_second-hand_items",
        ],
        "negative_topics": [
            "high_buyer_protection_fees",
            "poor_customer_service",
            "frequent_app_crashes",
            "difficult_to_access_account",
            "misleading_shipping_fees",
            "slow_response_times",
            "inaccurate_product_listings",
            "ineffective_search_function",
            "refund_issues",
            "limited_payment_options",
        ],
    }

    try:
        # Initialize client
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY environment variable is not set")

        gemini_client = GeminiClient(api_key=api_key, config=gemini_config)

        # Load and filter data
        file_path = "../tests/data/results/reviews_2025_02_22_210638_com.gardrops.csv"
        data_reviews = load_reviews_data(reviews_file_path=file_path)
        data_reviews, num_reviews = filter_reviews_data(data_reviews)

        logger.info(f"Loaded {num_reviews} reviews")

        # Get review texts
        reviews_list = data_reviews["review_text"].tolist()

        # Process reviews with dynamic schema
        logger.info("\n" + "=" * 80)
        logger.info("DYNAMIC SCHEMA CONFIGURATION")
        logger.info("=" * 80)
        logger.info(f"Positive topics: {len(topics['positive_topics'])}")
        logger.info(f"Negative topics: {len(topics['negative_topics'])}")
        logger.info("Using Literal type constraints to prevent hallucination")

        pipeline = ReviewTaggingPipeline(
            gemini_client=gemini_client,
            batch_config=batch_config,
        )
        num_reviews_to_process = 15

        results = await pipeline.process_batch(
            reviews=reviews_list[:num_reviews_to_process],  # Process first 10 for testing
            positive_topics=topics["positive_topics"],
            negative_topics=topics["negative_topics"],
        )

        # Display results
        logger.info("\n" + "=" * 80)
        logger.info("SAMPLE RESULTS")
        logger.info("=" * 80)

        for idx, (review, result) in enumerate(zip(reviews_list[:num_reviews_to_process], results), 1):
            logger.info(f"\nReview {idx}:")
            logger.info(f"Text: {review[:100]}...")
            if result:
                logger.info(f"Positive: {result.get('positive_topics', [])}")
                logger.info(f"Negative: {result.get('negative_topics', [])}")
            else:
                logger.info("Status: FAILED")

        # Final metrics
        logger.info("\n" + "=" * 80)
        logger.info("FINAL METRICS")
        logger.info("=" * 80)
        logger.info(f"Total Reviews: {pipeline.metrics.total_reviews}")
        logger.info(f"Successful: {pipeline.metrics.successful}")
        logger.info(f"Failed: {pipeline.metrics.failed}")
        logger.info(f"Success Rate: {pipeline.metrics.success_rate:.1f}%")
        logger.info(f"Duration: {pipeline.metrics.total_duration:.2f}s")
        logger.info(
            f"Throughput: {pipeline.metrics.reviews_per_second:.2f} reviews/sec"
        )
        logger.info(f"Cached Input Tokens: {pipeline.metrics.cached_input_tokens:,}")
        logger.info(f"Input Tokens: {pipeline.metrics.input_tokens:,}")
        logger.info(f"Thinking Tokens: {pipeline.metrics.thinking_tokens:,}")
        logger.info(f"Output Tokens: {pipeline.metrics.output_tokens:,}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
