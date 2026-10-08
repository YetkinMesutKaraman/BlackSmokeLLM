from typing import Protocol, runtime_checkable

from blacksmoke.llm.types import LLMRequest, LLMResponse, ModelTarget


@runtime_checkable
class LLMProvider(Protocol):
    """Port implemented by every provider adapter and every resilience wrapper.

    Adapters must translate SDK failures into `blacksmoke.llm.errors` types and must not
    retry on their own; retries are owned by the resilience layer.
    """

    name: str

    async def generate(self, request: LLMRequest, target: ModelTarget) -> LLMResponse: ...

    async def aclose(self) -> None: ...
