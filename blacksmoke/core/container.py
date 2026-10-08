"""Composition root: builds and wires every component once per application."""

from collections.abc import Mapping
from dataclasses import dataclass

from blacksmoke.core.config import Config, load_config
from blacksmoke.core.settings import Settings
from blacksmoke.data.repository import CsvReviewRepository
from blacksmoke.data.writer import ResultWriter
from blacksmoke.llm.errors import ProviderUnavailableError
from blacksmoke.llm.factory import ProviderBuilder, ProviderFactory
from blacksmoke.llm.gateway import LLMGateway
from blacksmoke.llm.pricing import PricingCatalog
from blacksmoke.llm.providers.gemini_provider import GeminiProvider
from blacksmoke.llm.providers.openai_provider import OpenAIProvider
from blacksmoke.services.runners import AggregateRunner, PerReviewRunner
from blacksmoke.services.task_service import TaskService
from blacksmoke.tasks.prompts import PromptBuilder, PromptRenderer
from blacksmoke.tasks.registry import TaskRegistry, load_builtin_tasks


@dataclass
class Container:
    settings: Settings
    config: Config
    registry: TaskRegistry
    factory: ProviderFactory
    gateway: LLMGateway
    repository: CsvReviewRepository
    task_service: TaskService

    async def aclose(self) -> None:
        await self.factory.aclose()


def default_provider_builders(settings: Settings) -> dict[str, ProviderBuilder]:
    def build_openai() -> OpenAIProvider:
        if settings.openai_api_key is None:
            raise ProviderUnavailableError("OPENAI_API_KEY is not set")
        return OpenAIProvider.create(
            api_key=settings.openai_api_key.get_secret_value(),
            organization=settings.openai_org_id or None,
        )

    def build_gemini() -> GeminiProvider:
        if settings.google_api_key is None:
            raise ProviderUnavailableError("GOOGLE_API_KEY is not set")
        return GeminiProvider.create(api_key=settings.google_api_key.get_secret_value())

    return {OpenAIProvider.name: build_openai, GeminiProvider.name: build_gemini}


def build_container(
    settings: Settings | None = None,
    *,
    config: Config | None = None,
    provider_builders: Mapping[str, ProviderBuilder] | None = None,
    registry: TaskRegistry | None = None,
) -> Container:
    settings = settings or Settings()
    config = config or load_config(settings.config_dir)
    registry = registry or load_builtin_tasks()
    factory = ProviderFactory(
        provider_builders if provider_builders is not None else default_provider_builders(settings)
    )
    pricing = PricingCatalog(config.models)
    prompts = PromptBuilder(PromptRenderer(config.prompts_dir), config.app.prompt_delimiter)
    registry.validate(config.tasks, factory.names, pricing, prompts)

    gateway = LLMGateway(factory, pricing)
    repository = CsvReviewRepository(config.app.data_dir, config.app.csv_formats)
    task_service = TaskService(
        registry=registry,
        tasks_config=config.tasks,
        repository=repository,
        default_filter=config.app.review_filter,
        pricing=pricing,
        provider_names=factory.names,
        aggregate_runner=AggregateRunner(gateway, prompts),
        per_review_runner=PerReviewRunner(
            gateway, prompts, ResultWriter(config.app.output_dir), repository.data_dir
        ),
    )
    return Container(
        settings=settings,
        config=config,
        registry=registry,
        factory=factory,
        gateway=gateway,
        repository=repository,
        task_service=task_service,
    )
