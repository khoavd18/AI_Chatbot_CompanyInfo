from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.core.setting_loader import ensure_env_loaded, load_settings
from src.core.logging_setup import setup_logging
from src.core.startup import initialize_rag_components, warmup_embedding_model


ensure_env_loaded()
setup_logging()
logger = logging.getLogger("api")

from src.api.routes import chat_router, health_router


def _parse_allowed_origins() -> list[str]:
    raw_origins = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:8501,http://127.0.0.1:8501",
    )
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    llm_config = settings.get("llm", {})
    logger.info(
        "Starting NVDK Chatbot API provider=%s model=%s",
        llm_config.get("provider", "unknown"),
        llm_config.get("model_name", "unknown"),
    )

    try:
        components = initialize_rag_components()
        if components and components.get("bm25"):
            logger.info("RAG components initialized successfully.")
            warmup_embedding_model()
        else:
            logger.warning(
                "RAG components are not fully initialized. Chat requests will return 503 until the Qdrant index is built."
            )
    except Exception as e:
        logger.error("Failed to initialize RAG components", exc_info=True)
        raise e

    yield
    logger.info("Shutting down NVDK Chatbot API...")


app = FastAPI(
    title="NVDK Chatbot API",
    description="API for NVDK Architecture Chatbot",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_allowed_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def track_response_time(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "request_id=%s method=%s path=%s unhandled_exception",
            request_id,
            request.method,
            request.url.path,
        )
        raise

    duration_ms = (time.perf_counter() - start_time) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"
    logger.info(
        "request_id=%s method=%s path=%s status=%s latency_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


app.include_router(health_router, tags=["health"])
app.include_router(chat_router, prefix="/api", tags=["chat"])


@app.get("/")
async def root():
    settings = load_settings()
    return {
        "message": "NVDK Chatbot API",
        "version": settings.get("app", {}).get("version", "1.0.0"),
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.routes.app:app", host="0.0.0.0", port=8000, reload=True)
