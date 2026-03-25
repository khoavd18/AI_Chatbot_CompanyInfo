# import logging
# from typing import List

# from qdrant_client import QdrantClient
# from qdrant_client.models import ScoredPoint
# from qdrant_client.http.exceptions import ResponseHandlingException

# from src.core.setting_loader import load_settings
# from src.core.schema import RetrievedDocument
# from src.rag.vectorstore.qdrant import get_qdrant_client
# from src.rag.embedding.embed_text import embed_texts
# from src.rag.retrieval.scoring.bm25 import BM25

# settings = load_settings()
# logger = logging.getLogger("retrieval")

# COLLECTION_NAME = settings["vector_database"]["collection_name"]
# RETRIEVAL_CONFIG = settings["retrieval"]
# TOP_K = RETRIEVAL_CONFIG.get("top_k", 10)
# SCORE_THRESHOLD = RETRIEVAL_CONFIG.get("score_threshold", 0.0)
# DENSE_WEIGHT = RETRIEVAL_CONFIG.get("dense_weight", 0.6)
# BM25_WEIGHT = RETRIEVAL_CONFIG.get("bm25_weight", 0.4)

# def hybrid_retrieve(query: str, bm25: BM25) -> List[RetrievedDocument]:
#     if not query or not query.strip():
#         logger.warning("Empty query received for hybrid retrieval.")
#         return []

#     try:
#         client: QdrantClient = get_qdrant_client()
#         dense_vectors = embed_texts([query])
#         if not dense_vectors:
#             logger.error("Failed to embed query.")
#             return []

#         query_vector = dense_vectors[0]

#         response = client.query_points(
#             collection_name=COLLECTION_NAME,
#             query=query_vector,
#             using="dense",  # specify named vector for hybrid search
#             limit=TOP_K * 3,  # lấy dư để rerank
#             with_payload=True,
#             score_threshold=SCORE_THRESHOLD,
#         )

#         points: list[ScoredPoint] = response.points
#         documents: list[RetrievedDocument] = []

#         for point in points:
#             payload = point.payload or {}
#             text = payload.get("text", "")

#             if not text:
#                 continue

#             bm25_score = bm25.score(query, text)
#             hybrid_score = (DENSE_WEIGHT * point.score + BM25_WEIGHT * bm25_score)

#             documents.append(
#                 RetrievedDocument(
#                     id=str(point.id),
#                     score=hybrid_score,
#                     text=text,
#                     metadata={
#                         **{k: v for k, v in payload.items() if k != "text"},
#                         "dense_score": point.score,
#                         "bm25_score": bm25_score,
#                     },
#                 )
#             )

#         documents.sort(key=lambda d: d.score, reverse=True)
#         return documents[:TOP_K]
    
#     except ResponseHandlingException as e:
#         logger.error(f"Qdrant connection error: {e}")
#         raise ConnectionError("Cannot connect to vector database")
#     except Exception as e:
#         logger.error(f"Error during retrieval: {e}", exc_info=True)
#         return []


import logging
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint
from qdrant_client.http.exceptions import ResponseHandlingException

from src.core.schema import RetrievedDocument
from src.core.setting_loader import load_settings
from src.core.startup import get_corpus_documents
from src.rag.embedding.embed_text import embed_texts
from src.rag.retrieval.scoring.bm25 import BM25
from src.rag.vectorstore.qdrant import get_qdrant_client

settings = load_settings()
logger = logging.getLogger("retrieval")

COLLECTION_NAME = settings["vector_database"]["collection_name"]
RETRIEVAL_CONFIG = settings["retrieval"]
TOP_K = RETRIEVAL_CONFIG.get("top_k", 10)
SCORE_THRESHOLD = RETRIEVAL_CONFIG.get("score_threshold", 0.0)
DENSE_WEIGHT = RETRIEVAL_CONFIG.get("dense_weight", 0.6)
BM25_WEIGHT = RETRIEVAL_CONFIG.get("bm25_weight", 0.4)

RRF_K = 60


def _infer_intent(query: str) -> str:
    q = query.lower()

    company_keywords = ["công ty", "hotline", "email", "địa chỉ", "website", "giờ làm việc", "liên hệ"]
    project_keywords = ["dự án", "công trình", "địa điểm", "chủ đầu tư", "diện tích", "hoàn thành"]
    news_keywords = ["tin tức", "bài viết", "xu hướng", "mới nhất", "blog", "tin mới"]
    style_keywords = ["phong cách", "kiến trúc", "nội thất", "style", "japandi", "modern", "luxury", "tân cổ điển"]

    if any(k in q for k in company_keywords):
        return "company"
    if any(k in q for k in project_keywords):
        return "project"
    if any(k in q for k in news_keywords):
        return "news"
    if any(k in q for k in style_keywords):
        return "style"
    return "general"


def _source_boost(intent: str, doc: RetrievedDocument) -> float:
    source_type = doc.metadata.get("type", "")
    chunk_type = doc.metadata.get("chunk_type", "")
    boost = 0.0

    # intent boost
    if intent == "company":
        if source_type == "company_info":
            boost += 0.05
        if chunk_type == "contact_info":
            boost += 0.05

    elif intent == "project":
        if source_type == "project":
            boost += 0.05
        elif source_type == "project_category":
            boost += 0.02
        if chunk_type in {"overview", "context", "specs", "full_content"}:
            boost += 0.02

    elif intent == "news":
        if source_type == "news":
            boost += 0.05
        elif source_type == "news_category":
            boost += 0.02
        if chunk_type in {"overview", "full_content", "meta"}:
            boost += 0.02

    elif intent == "style":
        if source_type in {"architecture_type", "interior_style"}:
            boost += 0.05
        if chunk_type in {"definition", "description"}:
            boost += 0.02

    # chunk-type prior
    if chunk_type == "overview":
        boost += 0.02
    elif chunk_type == "definition":
        boost += 0.02
    elif chunk_type == "contact_info":
        boost += 0.03
    elif chunk_type in {"context", "specs", "full_content", "description"}:
        boost += 0.01
    elif chunk_type == "seo":
        boost -= 0.005
    elif chunk_type in {"media", "video", "icon"}:
        boost -= 0.02

    # source prior
    if source_type == "hero_slide":
        boost -= 0.01

    return boost


def _exact_name_boost(query: str, doc: RetrievedDocument) -> float:
    q = query.lower()
    candidate_fields = [
        doc.metadata.get("project_name", ""),
        doc.metadata.get("news_item_title", ""),
        doc.metadata.get("company_name", ""),
        doc.metadata.get("architecture_type_name", ""),
        doc.metadata.get("interior_name", ""),
        doc.metadata.get("project_category_name", ""),
        doc.metadata.get("news_category_name", ""),
        doc.metadata.get("hero_slide_title", ""),
    ]

    for field in candidate_fields:
        field = str(field).strip().lower()
        if field and field in q:
            return 0.03

    return 0.0


def _get_source_limits(intent: str) -> dict:
    if intent == "company":
        return {
            "company_info": 3,
            "project": 2,
            "news": 2,
            "architecture_type": 1,
            "interior_style": 1,
            "project_category": 1,
            "news_category": 1,
            "hero_slide": 1,
        }
    if intent == "project":
        return {
            "project": 5,
            "project_category": 2,
            "company_info": 2,
            "news": 2,
            "architecture_type": 1,
            "interior_style": 1,
            "news_category": 1,
            "hero_slide": 1,
        }
    if intent == "news":
        return {
            "news": 5,
            "news_category": 2,
            "project": 2,
            "company_info": 1,
            "architecture_type": 1,
            "interior_style": 1,
            "project_category": 1,
            "hero_slide": 1,
        }
    if intent == "style":
        return {
            "architecture_type": 3,
            "interior_style": 3,
            "project": 3,
            "company_info": 1,
            "news": 1,
            "project_category": 1,
            "news_category": 1,
            "hero_slide": 1,
        }
    return {
        "project": 4,
        "news": 3,
        "company_info": 2,
        "architecture_type": 2,
        "interior_style": 2,
        "project_category": 1,
        "news_category": 1,
        "hero_slide": 1,
    }


def hybrid_retrieve(query: str, bm25: BM25, top_k: int | None = None) -> List[RetrievedDocument]:
    if not query or not query.strip():
        logger.warning("Empty query received for hybrid retrieval.")
        return []

    top_k = top_k or TOP_K
    intent = _infer_intent(query)

    try:
        client: QdrantClient = get_qdrant_client()
        dense_vectors = embed_texts([query])
        if not dense_vectors:
            logger.error("Failed to embed query.")
            return []

        query_vector = dense_vectors[0]
        dense_limit = max(top_k * 6, 30)

        # 1) Dense candidates từ Qdrant
        dense_response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            using="dense",
            limit=dense_limit,
            with_payload=True,
            score_threshold=SCORE_THRESHOLD,
        )

        dense_points: list[ScoredPoint] = dense_response.points
        dense_docs_by_id = {}
        dense_rank_by_id = {}

        for rank, point in enumerate(dense_points, start=1):
            payload = point.payload or {}
            text = payload.get("text", "")
            if not text:
                continue

            doc = RetrievedDocument(
                id=str(point.id),
                score=float(point.score),
                text=text,
                metadata={
                    **{k: v for k, v in payload.items() if k != "text"},
                    "dense_score": float(point.score),
                },
            )
            dense_docs_by_id[doc.id] = doc
            dense_rank_by_id[doc.id] = rank

        # 2) Sparse/BM25 candidates trên toàn corpus
        corpus_docs = get_corpus_documents()
        if not corpus_docs:
            logger.warning("Corpus documents not available in startup cache.")
            return list(dense_docs_by_id.values())[:top_k]

        bm25_scored = []
        for doc in corpus_docs:
            score = bm25.score(query, doc.text)
            if score > 0:
                bm25_scored.append((doc, score))

        bm25_scored.sort(key=lambda x: x[1], reverse=True)
        sparse_limit = max(top_k * 6, 30)
        sparse_candidates = bm25_scored[:sparse_limit]

        sparse_rank_by_id = {}
        sparse_docs_by_id = {}
        for rank, (doc, score) in enumerate(sparse_candidates, start=1):
            copied_doc = RetrievedDocument(
                id=doc.id,
                score=float(score),
                text=doc.text,
                metadata={
                    **doc.metadata,
                    "bm25_score": float(score),
                },
            )
            sparse_docs_by_id[copied_doc.id] = copied_doc
            sparse_rank_by_id[copied_doc.id] = rank

        # 3) Merge dense + sparse
        merged_ids = set(dense_docs_by_id.keys()) | set(sparse_docs_by_id.keys())
        merged_docs: list[RetrievedDocument] = []

        for doc_id in merged_ids:
            base_doc = dense_docs_by_id.get(doc_id) or sparse_docs_by_id.get(doc_id)
            if base_doc is None:
                continue

            dense_rank = dense_rank_by_id.get(doc_id)
            sparse_rank = sparse_rank_by_id.get(doc_id)

            dense_rrf = DENSE_WEIGHT * (1.0 / (RRF_K + dense_rank)) if dense_rank is not None else 0.0
            sparse_rrf = BM25_WEIGHT * (1.0 / (RRF_K + sparse_rank)) if sparse_rank is not None else 0.0

            boost = _source_boost(intent, base_doc) + _exact_name_boost(query, base_doc)
            final_score = dense_rrf + sparse_rrf + boost

            metadata = dict(base_doc.metadata)
            metadata["intent"] = intent
            metadata["dense_rank"] = dense_rank
            metadata["sparse_rank"] = sparse_rank
            metadata["dense_rrf"] = dense_rrf
            metadata["sparse_rrf"] = sparse_rrf
            metadata["hybrid_boost"] = boost

            merged_docs.append(
                RetrievedDocument(
                    id=base_doc.id,
                    score=float(final_score),
                    text=base_doc.text,
                    metadata=metadata,
                )
            )

        merged_docs.sort(key=lambda d: d.score, reverse=True)

        # 4) Source balancing
        source_limits = _get_source_limits(intent)
        source_counts = {}
        final_docs = []

        for doc in merged_docs:
            source_type = doc.metadata.get("type", "unknown")
            current_count = source_counts.get(source_type, 0)
            max_allowed = source_limits.get(source_type, 1)

            if current_count >= max_allowed:
                continue

            final_docs.append(doc)
            source_counts[source_type] = current_count + 1

            if len(final_docs) >= top_k:
                break

        logger.info(
            f"Hybrid retrieved {len(final_docs)} docs | "
            f"intent={intent} | dense_candidates={len(dense_docs_by_id)} | sparse_candidates={len(sparse_candidates)}"
        )

        return final_docs

    except ResponseHandlingException as e:
        logger.error(f"Qdrant connection error: {e}")
        raise ConnectionError("Cannot connect to vector database")
    except Exception as e:
        logger.error(f"Error during retrieval: {e}", exc_info=True)
        return []