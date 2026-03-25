from __future__ import annotations

import logging
import time

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import chat_router, health_router
from src.core.logging_setup import setup_logging
from src.core.startup import initialize_rag_components   # thêm dòng này


setup_logging()
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting NVDK Chatbot API...")
    try:
        initialize_rag_components()   # thêm dòng này
        logger.info("RAG components initialized successfully.")
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def track_response_time(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    response.headers["X-Response-Time"] = f"{duration:.3f}s"
    logger.info("%s %s took %.3fs", request.method, request.url.path, duration)
    return response


app.include_router(health_router, tags=["health"])
app.include_router(chat_router, prefix="/api", tags=["chat"])


@app.get("/")
async def root():
    return {
        "message": "NVDK Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.routes.app:app", host="0.0.0.0", port=8000, reload=True)