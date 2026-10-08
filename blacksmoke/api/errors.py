import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from blacksmoke.core.errors import (
    BlackSmokeError,
    ConfigError,
    DatasetFormatError,
    DatasetNotFoundError,
    InvalidDatasetPathError,
    NoReviewsError,
    UnknownModelError,
    UnknownTaskError,
)
from blacksmoke.llm.errors import AllTargetsFailedError

logger = logging.getLogger(__name__)

STATUS_BY_ERROR: list[tuple[type[BlackSmokeError], int]] = [
    (UnknownTaskError, status.HTTP_404_NOT_FOUND),
    (DatasetNotFoundError, status.HTTP_404_NOT_FOUND),
    (InvalidDatasetPathError, status.HTTP_422_UNPROCESSABLE_CONTENT),
    (DatasetFormatError, status.HTTP_422_UNPROCESSABLE_CONTENT),
    (NoReviewsError, status.HTTP_422_UNPROCESSABLE_CONTENT),
    (UnknownModelError, status.HTTP_422_UNPROCESSABLE_CONTENT),
    (AllTargetsFailedError, status.HTTP_502_BAD_GATEWAY),
    (ConfigError, status.HTTP_500_INTERNAL_SERVER_ERROR),
]


def status_for(exc: BlackSmokeError) -> int:
    for error_type, code in STATUS_BY_ERROR:
        if isinstance(exc, error_type):
            return code
    return status.HTTP_500_INTERNAL_SERVER_ERROR


async def handle_domain_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, BlackSmokeError)
    code = status_for(exc)
    error: dict[str, Any] = {"type": type(exc).__name__, "message": str(exc)}
    if isinstance(exc, AllTargetsFailedError):
        error["attempts"] = [attempt.model_dump(mode="json") for attempt in exc.attempts]
    log = logger.error if code >= 500 else logger.info
    log("%s %s -> %d %s: %s", request.method, request.url.path, code, error["type"], exc)
    return JSONResponse(status_code=code, content={"error": error})


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(BlackSmokeError, handle_domain_error)
