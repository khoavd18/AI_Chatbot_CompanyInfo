import logging
import uuid

from src.rag.embedding.embed_text import embed_texts

logger = logging.getLogger("embedding")


def build_qdrant_points(chunks: list[dict]) -> list[dict]:
    if not chunks:
        logger.warning("No chunks provided to build Qdrant points.")
        return []

    valid_chunks = [chunk for chunk in chunks if chunk.get("text")]
    if not valid_chunks:
        logger.warning("No text found in the provided chunks.")
        return []

    texts = [chunk["text"] for chunk in valid_chunks]
    embeddings = embed_texts(texts)

    if embeddings is None or len(embeddings) == 0:
        logger.warning("No embeddings generated for the provided texts.")
        return []

    points = []

    for chunk, vector in zip(valid_chunks, embeddings):
        points.append({
            "id": chunk.get("metadata", {}).get("chunk_id", str(uuid.uuid4())),
            "vector": vector.tolist() if hasattr(vector, "tolist") else vector,
            "payload": {
                "text": chunk["text"],
                **chunk.get("metadata", {})
            }
        })

    logger.info(f"Built {len(points)} Qdrant points.")
    return points