import logging
import torch
from sentence_transformers import SentenceTransformer

from src.core.setting_loader import load_settings

settings = load_settings()
logger = logging.getLogger("embedding")

EMBEDDING_CONFIG = settings["embedding"]
EMBEDDING_MODEL = EMBEDDING_CONFIG["model"]
EMBEDDING_BATCH_SIZE = int(EMBEDDING_CONFIG.get("batch_size", 32))

_model = None


def _resolve_device() -> str:
    device = EMBEDDING_CONFIG.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("Embedding device is set to cuda, but CUDA is not available. Falling back to cpu.")
        return "cpu"
    return device


def get_model() -> SentenceTransformer:
    global _model # ghi vao bien toan cuc
    if _model is None: # neu chua co model thi load, chi load 1 lan
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL, device=_resolve_device())
    return _model

def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        logger.warning("No texts provided for embedding.")
        return []
    
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_tensor=False,
        show_progress_bar=False,
    ).tolist() # chuyen thanh list de luu vao qdrant BAT BUOC
    logger.info(f"Completed embedding texts {len(texts)}.")
    return embeddings
    
