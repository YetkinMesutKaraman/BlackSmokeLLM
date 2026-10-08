import logging
from collections.abc import Callable, Mapping

from blacksmoke.llm.errors import ProviderUnavailableError
from blacksmoke.llm.provider import LLMProvider

logger = logging.getLogger(__name__)

ProviderBuilder = Callable[[], LLMProvider]


class ProviderFactory:
    """Registry of provider builders. Providers are built on first use and cached for the
    lifetime of the application, so clients and connection pools are shared."""

    def __init__(self, builders: Mapping[str, ProviderBuilder] | None = None) -> None:
        self._builders: dict[str, ProviderBuilder] = dict(builders or {})
        self._instances: dict[str, LLMProvider] = {}

    def register(self, name: str, builder: ProviderBuilder) -> None:
        self._builders[name] = builder
        self._instances.pop(name, None)

    @property
    def names(self) -> list[str]:
        return sorted(self._builders)

    def get(self, name: str) -> LLMProvider:
        if name in self._instances:
            return self._instances[name]
        builder = self._builders.get(name)
        if builder is None:
            raise ProviderUnavailableError(f"Unknown provider '{name}'")
        provider = builder()
        self._instances[name] = provider
        logger.info("Initialized provider '%s'", name)
        return provider

    async def aclose(self) -> None:
        for name, provider in self._instances.items():
            try:
                await provider.aclose()
            except Exception:
                logger.exception("Failed to close provider '%s'", name)
        self._instances.clear()
