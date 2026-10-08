import inspect
from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import APIRouter, Body
from pydantic import BaseModel

from blacksmoke.api.deps import TaskServiceDep
from blacksmoke.services.results import TaskInfo
from blacksmoke.services.task_service import TASKS_PATH_PREFIX, TaskEndpoint, TaskService


def build_tasks_router(service: TaskService) -> APIRouter:
    """One typed POST route per registered task, so OpenAPI documents each request and
    response schema."""
    router = APIRouter(prefix=TASKS_PATH_PREFIX, tags=["tasks"])

    @router.get("", response_model=list[TaskInfo], summary="List tasks and their model routes")
    async def list_tasks(service: TaskServiceDep) -> list[TaskInfo]:
        return service.list_tasks()

    for endpoint in service.endpoints():
        router.add_api_route(
            f"/{endpoint.name}",
            _make_handler(endpoint),
            methods=["POST"],
            response_model=endpoint.response_model,
            summary=endpoint.summary,
            name=endpoint.name,
            operation_id=endpoint.name,
        )
    return router


def _make_handler(endpoint: TaskEndpoint) -> Callable[..., Awaitable[BaseModel]]:
    async def handler(payload: BaseModel, service: TaskService) -> BaseModel:
        return await service.run(endpoint.name, payload)

    # FastAPI reads the signature to build validation and docs; set it explicitly so each
    # generated route gets its task's concrete request model.
    handler.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [
            inspect.Parameter(
                "payload",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=Annotated[endpoint.request_model, Body(examples=[endpoint.example])],
            ),
            inspect.Parameter(
                "service", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=TaskServiceDep
            ),
        ]
    )
    handler.__name__ = f"run_{endpoint.name}"
    return handler
