import logging
import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger("reranking")

class CrossEncoderModel:
    def __init__(self, model_name: str, device: str = "cpu"):
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("Reranker device is set to cuda, but CUDA is not available. Falling back to cpu.")
            device = "cpu"

        logger.info(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name, device=device)

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        return self.model.predict(pairs).tolist()
