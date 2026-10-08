from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from blacksmoke.api.errors import register_error_handlers
from blacksmoke.api.routes import health
from blacksmoke.api.routes.tasks import build_tasks_router
from blacksmoke.core.config import load_config
from blacksmoke.core.container import Container, build_container
from blacksmoke.core.logging import configure_logging
from blacksmoke.core.settings import Settings


def create_app(container: Container | None = None) -> FastAPI:
    if container is None:
        settings = Settings()
        config = load_config(settings.config_dir)
        configure_logging(config.app.log_level)
        container = build_container(settings, config=config)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        await container.aclose()

    app = FastAPI(title="BlackSmoke LLM Review Analyzer", version="0.2.0", lifespan=lifespan)
    app.state.container = container
    register_error_handlers(app)
    app.include_router(health.router)
    app.include_router(build_tasks_router(container.task_service))
    return app
