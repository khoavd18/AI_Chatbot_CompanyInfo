from __future__ import annotations

import logging
import os

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from src.core.setting_loader import load_settings
from src.core.startup import get_initialization_status
from src.rag.vectorstore.qdrant import get_qdrant_client


logger = logging.getLogger("health")
router = APIRouter()


def _demo_rate_limit_enabled() -> bool:
    return os.getenv("DEMO_RATE_LIMIT_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


def _runtime_snapshot() -> dict:
    return {
        "session_state": "stateless",
        "demo_rate_limit": {
            "enabled": _demo_rate_limit_enabled(),
            "mode": "in_memory_demo" if _demo_rate_limit_enabled() else "disabled",
        },
        "max_query_length": int(os.getenv("MAX_QUERY_LENGTH", "500")),
        "max_retrieval_top_k": int(os.getenv("MAX_RETRIEVAL_TOP_K", "20")),
    }


def _app_snapshot(settings: dict | None) -> dict:
    app_config = (settings or {}).get("app", {})
    return {
        "name": app_config.get("name", "NVDK Chatbot API"),
        "env": app_config.get("env", "unknown"),
    }


def _validate_minimum_config(settings: dict) -> dict:
    vector_config = settings.get("vector_database", {})
    missing: list[str] = []

    has_qdrant_url = bool(vector_config.get("url"))
    has_qdrant_host_port = bool(vector_config.get("host")) and vector_config.get("port") not in {None, ""}

    if not vector_config.get("collection_name"):
        missing.append("vector_database.collection_name")
    if not has_qdrant_url and not has_qdrant_host_port:
        missing.append("vector_database.url or vector_database.host + port")

    connection_mode = "url" if has_qdrant_url else "host_port" if has_qdrant_host_port else "missing"
    endpoint = vector_config.get("url") or (
        f"{vector_config.get('host')}:{vector_config.get('port')}" if has_qdrant_host_port else None
    )

    return {
        "status": "ok" if not missing else "invalid",
        "missing": missing,
        "collection_name": vector_config.get("collection_name"),
        "qdrant_endpoint": endpoint,
        "connection_mode": connection_mode,
    }


def _check_qdrant(settings: dict, *, require_collection: bool) -> dict:
    vector_config = settings.get("vector_database", {})
    collection_name = vector_config.get("collection_name")
    config_check = _validate_minimum_config(settings)

    if config_check["status"] != "ok":
        return {
            "status": "skipped",
            "collection_name": collection_name,
            "collection_exists": False,
            "reason": "qdrant config is incomplete",
        }

    try:
        client = get_qdrant_client()
        collections = client.get_collections().collections
        collection_names = {collection.name for collection in collections}
        collection_exists = bool(collection_name) and collection_name in collection_names

        check = {
            "status": "up",
            "collection_name": collection_name,
            "collections": len(collection_names),
            "collection_exists": collection_exists,
        }
        if require_collection and not collection_exists:
            check["status"] = "down"
            check["error"] = f"Configured collection '{collection_name}' was not found."
        return check
    except Exception as error:
        logger.warning("Qdrant check failed: %s", error)
        return {
            "status": "down",
            "collection_name": collection_name,
            "collection_exists": False,
            "error": str(error),
        }


def _build_rag_readiness() -> dict:
    rag_status = get_initialization_status()
    reasons: list[str] = []

    if not rag_status.get("initialized"):
        reasons.append("startup_not_initialized")
    if not rag_status.get("sparse_embedder"):
        reasons.append("sparse_embedder_not_ready")
    if not rag_status.get("bm25"):
        reasons.append("bm25_not_ready")
    if not rag_status.get("reranker"):
        reasons.append("reranker_not_ready")
    if not rag_status.get("embedding_warmed_up"):
        reasons.append("embedding_not_warmed_up")
    if not rag_status.get("corpus_documents"):
        reasons.append("corpus_empty")

    return {
        "status": "ready" if not reasons else "not_ready",
        "reasons": reasons,
        **rag_status,
    }


def _health_payload(settings: dict | None, *, config_check: dict, qdrant_check: dict) -> dict:
    reasons: list[str] = []
    status_text = "ok"

    if config_check["status"] != "ok":
        status_text = "degraded"
        reasons.append("config_invalid")
    if qdrant_check["status"] == "down":
        status_text = "degraded"
        reasons.append("qdrant_unreachable")

    return {
        "status": status_text,
        "app": _app_snapshot(settings),
        "checks": {
            "app": {"status": "up"},
            "config": config_check,
            "qdrant": qdrant_check,
        },
        "runtime": _runtime_snapshot(),
        "reasons": reasons,
    }


def _readiness_payload(settings: dict | None, *, config_check: dict, qdrant_check: dict, rag_check: dict) -> dict:
    reasons: list[str] = []

    if config_check["status"] != "ok":
        reasons.append("config_invalid")
    if qdrant_check["status"] != "up":
        reasons.append("qdrant_not_ready")
    if rag_check["status"] != "ready":
        reasons.extend(rag_check.get("reasons", []))

    return {
        "status": "ready" if not reasons else "not_ready",
        "app": _app_snapshot(settings),
        "checks": {
            "config": config_check,
            "qdrant": qdrant_check,
            "rag": rag_check,
        },
        "runtime": _runtime_snapshot(),
        "reasons": reasons,
    }


@router.get("/health")
async def health_check():
    try:
        settings = load_settings()
        config_check = _validate_minimum_config(settings)
        qdrant_check = _check_qdrant(settings, require_collection=False)
        return _health_payload(settings, config_check=config_check, qdrant_check=qdrant_check)
    except Exception as error:
        logger.exception("Health check failed unexpectedly")
        return {
            "status": "degraded",
            "app": _app_snapshot(None),
            "checks": {
                "app": {"status": "up"},
                "config": {"status": "invalid", "error": str(error)},
                "qdrant": {"status": "skipped", "reason": "health setup failed"},
            },
            "runtime": _runtime_snapshot(),
            "reasons": ["health_check_failed"],
        }


@router.get("/readiness")
async def readiness_check():
    try:
        settings = load_settings()
        config_check = _validate_minimum_config(settings)
        qdrant_check = _check_qdrant(settings, require_collection=True)
        rag_check = _build_rag_readiness()
        payload = _readiness_payload(
            settings,
            config_check=config_check,
            qdrant_check=qdrant_check,
            rag_check=rag_check,
        )
        status_code = status.HTTP_200_OK if payload["status"] == "ready" else status.HTTP_503_SERVICE_UNAVAILABLE
        return JSONResponse(status_code=status_code, content=payload)
    except Exception as error:
        logger.exception("Readiness check failed unexpectedly")
        payload = {
            "status": "not_ready",
            "app": _app_snapshot(None),
            "checks": {
                "config": {"status": "invalid", "error": str(error)},
                "qdrant": {"status": "skipped", "reason": "readiness setup failed"},
                "rag": {"status": "not_ready", "reasons": ["readiness_check_failed"]},
            },
            "runtime": _runtime_snapshot(),
            "reasons": ["readiness_check_failed"],
        }
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=payload)
