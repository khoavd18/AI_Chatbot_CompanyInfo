import logging
from src.core.setting_loader import load_settings
from src.rag.embedding.embed_text import embed_texts

settings = load_settings()
logger = logging.getLogger("embedding")
EMBEDDING_BATCH_SIZE = settings["embedding"].get("batch_size", 32)

def batch_embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        logger.warning("No texts provided for embedding.")
        return []
    all_embeddings = []

    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i:i+EMBEDDING_BATCH_SIZE]
        batch_embeddings = embed_texts(batch)
        all_embeddings.extend(batch_embeddings)
        logger.info(f"Processed batch {i//EMBEDDING_BATCH_SIZE + 1} with {len(batch)} texts.")
    return all_embeddings
