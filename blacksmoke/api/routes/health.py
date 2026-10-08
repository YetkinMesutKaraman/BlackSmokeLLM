from fastapi import APIRouter

router = APIRouter(tags=["meta"])


@router.get("/")
async def root() -> dict[str, str]:
    return {"message": "BlackSmoke LLM Review Analyzer", "docs": "/docs", "tasks": "/v1/tasks"}


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
