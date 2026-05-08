from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pymongo.errors import PyMongoError

from .api.routes.auth import router as auth_router
from .api.routes.generation import router as generation_router
from .api.routes.system import router as system_router
from .core.config import settings
from .db.mongo import close_database, ensure_indexes, mongo_state
from .services.inference import generation_service

logger = logging.getLogger("cdvae.api")
logging.basicConfig(level=logging.INFO, format="%(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    await ensure_indexes()
     # preload model here
    if settings.model_path:
        try:
            await generation_service.load_model(None)
            print("✅ Model preloaded at startup")
        except Exception as e:
            print("❌ Model preload failed:", e)
            
    yield
    await close_database()



app = FastAPI(title=settings.app_name, version="3.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(system_router)
app.include_router(auth_router)
app.include_router(generation_router)

@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.exception(
            "request_failed request_id=%s route=%s latency_ms=%s model_loaded=%s mongo=%s",
            request_id,
            request.url.path,
            latency_ms,
            generation_service.status["loaded"],
            mongo_state()["status"],
        )
        raise

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["x-request-id"] = request_id
    logger.info(
        "request request_id=%s route=%s status=%s latency_ms=%s model_loaded=%s mongo=%s",
        request_id,
        request.url.path,
        response.status_code,
        latency_ms,
        generation_service.status["loaded"],
        mongo_state()["status"],
    )
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": "Validation failed.", "errors": exc.errors()})


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(PyMongoError)
async def mongo_exception_handler(_: Request, exc: PyMongoError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": "Database operation failed.", "error": str(exc)})


@app.exception_handler(RuntimeError)
async def runtime_exception_handler(_: Request, exc: RuntimeError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})
