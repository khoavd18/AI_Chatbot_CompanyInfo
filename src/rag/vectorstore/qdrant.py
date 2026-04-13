from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, SparseVectorParams, SparseIndexParams
import logging

from src.core.setting_loader import load_settings

settings = load_settings()
logger = logging.getLogger("vector_database")

QDRANT_CONFIG = settings["vector_database"]
COLLECTION_NAME = QDRANT_CONFIG["collection_name"]
VECTOR_SIZE = QDRANT_CONFIG["vector_size"]
DISTANCE = QDRANT_CONFIG.get("distance", "cosine")
TIMEOUT = QDRANT_CONFIG.get("timeout", 30)

_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is not None:
        return _client

    try:
        if QDRANT_CONFIG.get("url"):
            logger.info("Connect via URL")
            _client = QdrantClient(
                url=QDRANT_CONFIG["url"],
                api_key=QDRANT_CONFIG.get("api_key"),
                timeout=TIMEOUT,
            )
        else:
            logger.info(f"Connect via: {QDRANT_CONFIG.get('host')}:{QDRANT_CONFIG.get('port')}")
            _client = QdrantClient(
                host=QDRANT_CONFIG.get("host"),
                port=QDRANT_CONFIG.get("port"),
                api_key=QDRANT_CONFIG.get("api_key"),
                timeout=TIMEOUT,
            )

        _client.get_collections()
        logger.info("Successfully connected to Qdrant")
        return _client

    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        raise ConnectionError(f"Cannot connect to Qdrant database: {e}")


def ensure_collection(client: QdrantClient, recreate: bool = False):
    existing_collections = {c.name for c in client.get_collections().collections}

    if COLLECTION_NAME in existing_collections and not recreate:
        logger.info(f"Collection '{COLLECTION_NAME}' already exists.")
        return

    if COLLECTION_NAME in existing_collections and recreate:
        logger.info(f"Recreating collection '{COLLECTION_NAME}' to avoid duplicated points...")
    else:
        logger.info(f"Creating collection '{COLLECTION_NAME}' with hybrid vectors (dense + sparse)...")

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(
                size=VECTOR_SIZE,
                distance=Distance[DISTANCE.upper()]
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams()
            )
        }
    )

    logger.info(
        f"Collection '{COLLECTION_NAME}' is ready with dense vector size {VECTOR_SIZE}, "
        f"distance '{DISTANCE}', and sparse vectors."
    )