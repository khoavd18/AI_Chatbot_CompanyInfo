from __future__ import annotations

import logging

from fastapi import APIRouter

from src.core.setting_loader import load_settings
from src.rag.embedding.embed_text import get_model
from src.rag.vectorstore.qdrant import get_qdrant_client


logger = logging.getLogger("health")
router = APIRouter()


@router.get("/health")
async def health_check():
    settings = load_settings()

    health_status = {
        "status": "healthy",
        "services": {},
    }

    try:
        client = get_qdrant_client()
        collections = client.get_collections()
        health_status["services"]["qdrant"] = {
            "status": "up",
            "collections": len(collections.collections),
        }
    except Exception as e:
        logger.error("Qdrant health check failed: %s", e)
        health_status["status"] = "unhealthy"
        health_status["services"]["qdrant"] = {
            "status": "down",
            "error": str(e),
        }

    try:
        model = get_model()
        health_status["services"]["embedding"] = {
            "status": "up",
            "model": model.__class__.__name__,
        }
    except Exception as e:
        logger.error("Embedding model health check failed: %s", e)
        if health_status["status"] == "healthy":
            health_status["status"] = "degraded"
        health_status["services"]["embedding"] = {
            "status": "down",
            "error": str(e),
        }

    llm_config = settings.get("llm", {})
    health_status["services"]["llm"] = {
        "provider": llm_config.get("provider", "unknown"),
        "model": llm_config.get("model_name", "unknown"),
        "status": "configured",
    }

    return health_status