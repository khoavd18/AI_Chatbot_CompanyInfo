import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOTENV_PATH = PROJECT_ROOT / ".env"


@lru_cache(maxsize=1)
def ensure_env_loaded() -> Path | None:
    """Load the project-root .env once, without overriding real environment vars."""
    if not DOTENV_PATH.exists():
        return None

    load_dotenv(DOTENV_PATH, override=False, encoding="utf-8")
    return DOTENV_PATH


def _env_bool(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    return int(raw)


def _env_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    return float(raw)


def load_settings():
    """Load settings from YAML and override with environment variables"""
    ensure_env_loaded()
    src_dir = Path(__file__).resolve().parents[1]   # .../src
    settings_path = src_dir / "config" / "settings.yaml"

    with open(settings_path, "r", encoding="utf-8") as file:
        settings = yaml.safe_load(file)

    # Override với environment variables
    if os.getenv("APP_ENV"):
        settings["app"]["env"] = os.getenv("APP_ENV")

    # Vector database overrides
    if os.getenv("QDRANT_URL"):
        settings["vector_database"]["url"] = os.getenv("QDRANT_URL")
    if os.getenv("QDRANT_API_KEY"):
        settings["vector_database"]["api_key"] = os.getenv("QDRANT_API_KEY")
    if os.getenv("QDRANT_COLLECTION_NAME"):
        settings["vector_database"]["collection_name"] = os.getenv("QDRANT_COLLECTION_NAME")
    qdrant_timeout = _env_int("QDRANT_TIMEOUT")
    if qdrant_timeout is not None:
        settings["vector_database"]["timeout"] = qdrant_timeout

    # Embedding overrides
    if os.getenv("EMBEDDING_MODEL"):
        settings["embedding"]["model"] = os.getenv("EMBEDDING_MODEL")
    if os.getenv("EMBEDDING_DEVICE"):
        settings["embedding"]["device"] = os.getenv("EMBEDDING_DEVICE")
    embedding_batch_size = _env_int("EMBEDDING_BATCH_SIZE")
    if embedding_batch_size is not None:
        settings["embedding"]["batch_size"] = embedding_batch_size

    # LLM overrides
    if os.getenv("LLM_PROVIDER"):
        settings["llm"]["provider"] = os.getenv("LLM_PROVIDER")
    if os.getenv("LLM_MODEL_NAME"):
        settings["llm"]["model_name"] = os.getenv("LLM_MODEL_NAME")
    if os.getenv("LLM_BASE_URL"):
        settings["llm"]["base_url"] = os.getenv("LLM_BASE_URL")
    if os.getenv("LLM_DEVICE"):
        settings["llm"]["device"] = os.getenv("LLM_DEVICE")
    llm_temperature = _env_float("LLM_TEMPERATURE")
    if llm_temperature is not None:
        settings["llm"]["temperature"] = llm_temperature
    llm_max_tokens = _env_int("LLM_MAX_TOKENS")
    if llm_max_tokens is not None:
        settings["llm"]["max_tokens"] = llm_max_tokens
    llm_timeout = _env_int("LLM_TIMEOUT")
    if llm_timeout is not None:
        settings["llm"]["timeout"] = llm_timeout
    llm_trust_remote_code = _env_bool("LLM_TRUST_REMOTE_CODE")
    if llm_trust_remote_code is not None:
        settings["llm"]["trust_remote_code"] = llm_trust_remote_code
    llm_load_in_8bit = _env_bool("LLM_LOAD_IN_8BIT")
    if llm_load_in_8bit is not None:
        settings["llm"]["load_in_8bit"] = llm_load_in_8bit
    llm_enable_fallback = _env_bool("LLM_ENABLE_FALLBACK")
    if llm_enable_fallback is not None:
        settings["llm"]["enable_fallback"] = llm_enable_fallback

    # Retrieval overrides
    retrieval_top_k = _env_int("RETRIEVAL_TOP_K")
    if retrieval_top_k is not None:
        settings["retrieval"]["top_k"] = retrieval_top_k
    retrieval_score_threshold = _env_float("RETRIEVAL_SCORE_THRESHOLD")
    if retrieval_score_threshold is not None:
        settings["retrieval"]["score_threshold"] = retrieval_score_threshold
    dense_weight = _env_float("DENSE_WEIGHT")
    if dense_weight is not None:
        settings["retrieval"]["dense_weight"] = dense_weight
    bm25_weight = _env_float("BM25_WEIGHT")
    if bm25_weight is not None:
        settings["retrieval"]["bm25_weight"] = bm25_weight

    # Reranking overrides
    if "reranking" not in settings:
        settings["reranking"] = {}
    if os.getenv("RERANKING_MODEL"):
        settings["reranking"]["model"] = os.getenv("RERANKING_MODEL")
    if os.getenv("RERANKING_DEVICE"):
        settings["reranking"]["device"] = os.getenv("RERANKING_DEVICE")
    reranking_top_k = _env_int("RERANKING_TOP_K")
    if reranking_top_k is not None:
        settings["reranking"]["top_k"] = reranking_top_k

    return settings
