from __future__ import annotations

from fastapi import APIRouter

from ...db.mongo import ping_database
from ...schemas.common import HealthResponse
from ...services.inference import generation_service

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    try:
        db_ok = await ping_database()
    except Exception:
        db_ok = False
    status = generation_service.status
    return HealthResponse(
        status="ok" if db_ok else "degraded",
        database="ok" if db_ok else "down",
        model_loaded=status["loaded"],
        model_path=status["model_path"],
        device=status["device"],
        version="3.0.0",
    )
