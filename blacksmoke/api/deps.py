from typing import Annotated

from fastapi import Depends, Request

from blacksmoke.core.container import Container
from blacksmoke.services.task_service import TaskService


def get_container(request: Request) -> Container:
    return request.app.state.container


def get_task_service(container: Annotated[Container, Depends(get_container)]) -> TaskService:
    return container.task_service


TaskServiceDep = Annotated[TaskService, Depends(get_task_service)]
