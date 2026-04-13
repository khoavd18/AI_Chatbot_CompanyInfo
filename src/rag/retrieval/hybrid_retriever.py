import logging
import re
import time
import unicodedata
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import ScoredPoint

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


def _strip_accents(text: str) -> str:
    normalized = "".join(
        character
        for character in unicodedata.normalize("NFKD", text or "")
        if not unicodedata.combining(character)
    )
    return normalized.replace("đ", "d").replace("Đ", "D")


def _normalize(text: str) -> str:
    plain = _strip_accents(text).lower()
    return re.sub(r"\s+", " ", plain).strip()


def _contains_marker(text: str, marker: str) -> bool:
    normalized_text = _normalize(text)
    normalized_marker = _normalize(marker)
    if not normalized_text or not normalized_marker:
        return False

    if " " in normalized_marker:
        return f" {normalized_marker} " in f" {normalized_text} "

    return normalized_marker in set(normalized_text.split())


def _contains_any(text: str, markers: list[str]) -> bool:
    return any(_contains_marker(text, marker) for marker in markers)


def _infer_intent(query: str) -> str:
    company_keywords = [
        "cong ty",
        "hotline",
        "email",
        "dia chi",
        "website",
        "gio lam viec",
        "lien he",
        "the manh",
        "linh vuc",
        "dich vu",
        "chuyen ve",
        "nang luc",
        "nhan vien",
        "nhan su",
        "nhan luc",
        "ky su",
        "kien truc su",
    ]
    project_keywords = ["du an", "cong trinh", "dia diem", "chu dau tu", "dien tich", "hoan thanh"]
    news_keywords = ["tin tuc", "bai viet", "xu huong", "moi nhat", "blog", "tin moi"]
    category_keywords = ["danh muc", "chuyen muc", "loai hinh"]
    style_keywords = ["phong cach", "style", "japandi", "modern", "luxury", "tan co dien", "hien dai", "tropical"]
    media_keywords = ["hinh anh", "anh", "thumbnail", "video", "clip", "minh hoa"]

    if _contains_any(query, media_keywords):
        return "media"
    if _contains_any(query, category_keywords) and _contains_any(query, project_keywords):
        return "project_category"
    if _contains_any(query, category_keywords) and _contains_any(query, news_keywords):
        return "news_category"
    if _contains_any(query, project_keywords):
        return "project"
    if _contains_any(query, style_keywords):
        return "style"
    if _contains_any(query, news_keywords):
        return "news"
    if _contains_any(query, company_keywords):
        return "company"
    return "general"


def _source_boost(intent: str, doc: RetrievedDocument) -> float:
    source_type = doc.metadata.get("type", "")
    chunk_type = doc.metadata.get("chunk_type", "")
    boost = 0.0

    if intent == "company":
        if source_type == "company_info":
            boost += 0.05
        if chunk_type == "contact_info":
            boost += 0.05
        if chunk_type == "stats":
            boost += 0.06
    elif intent == "project_category":
        if source_type == "project_category":
            boost += 0.08
        elif source_type == "project":
            boost += 0.02
        if chunk_type in {"definition", "description"}:
            boost += 0.03
    elif intent == "project":
        if source_type == "project":
            boost += 0.05
        elif source_type == "project_category":
            boost += 0.02
        if chunk_type in {"overview", "context", "specs", "full_content"}:
            boost += 0.02
    elif intent == "news_category":
        if source_type == "news_category":
            boost += 0.08
        elif source_type == "news":
            boost += 0.02
        if chunk_type in {"definition", "description"}:
            boost += 0.03
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
    elif intent == "media":
        if chunk_type in {"media", "video"}:
            boost += 0.08
        if source_type in {"project", "news", "hero_slide", "interior_style", "architecture_type", "company_info"}:
            boost += 0.02

    if chunk_type == "overview":
        boost += 0.02
    elif chunk_type == "definition":
        boost += 0.02
    elif chunk_type == "contact_info":
        boost += 0.03
    elif chunk_type == "stats":
        boost += 0.03
    elif chunk_type in {"context", "specs", "full_content", "description"}:
        boost += 0.01
    elif chunk_type == "seo":
        boost -= 0.005
    elif chunk_type in {"media", "video", "icon"}:
        boost += 0.01 if intent == "media" else -0.02

    if source_type == "hero_slide":
        boost -= 0.01 if intent != "media" else 0.0

    return boost


def _exact_name_boost(query: str, doc: RetrievedDocument) -> float:
    q = _normalize(query)
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
        normalized_field = _normalize(str(field).strip())
        if normalized_field and normalized_field in q:
            return 0.03

    return 0.0


def _get_source_limits(intent: str) -> dict[str, int]:
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
    if intent == "project_category":
        return {
            "project_category": 4,
            "project": 2,
            "company_info": 1,
            "news": 1,
            "architecture_type": 1,
            "interior_style": 1,
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
    if intent == "news_category":
        return {
            "news_category": 4,
            "news": 2,
            "project": 1,
            "company_info": 1,
            "architecture_type": 1,
            "interior_style": 1,
            "project_category": 1,
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
    if intent == "media":
        return {
            "project": 3,
            "news": 2,
            "hero_slide": 2,
            "interior_style": 2,
            "architecture_type": 2,
            "company_info": 1,
            "project_category": 1,
            "news_category": 1,
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
        overall_start = time.perf_counter()
        client: QdrantClient = get_qdrant_client()

        embed_start = time.perf_counter()
        dense_vectors = embed_texts([query])
        embed_ms = (time.perf_counter() - embed_start) * 1000
        if not dense_vectors:
            logger.error("Failed to embed query.")
            return []

        query_vector = dense_vectors[0]
        dense_limit = max(top_k * 6, 30)

        qdrant_start = time.perf_counter()
        dense_response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            using="dense",
            limit=dense_limit,
            with_payload=True,
            score_threshold=SCORE_THRESHOLD,
        )
        qdrant_ms = (time.perf_counter() - qdrant_start) * 1000

        dense_points: list[ScoredPoint] = dense_response.points
        dense_docs_by_id: dict[str, RetrievedDocument] = {}
        dense_rank_by_id: dict[str, int] = {}

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
                    **{key: value for key, value in payload.items() if key != "text"},
                    "dense_score": float(point.score),
                },
            )
            dense_docs_by_id[doc.id] = doc
            dense_rank_by_id[doc.id] = rank

        corpus_docs = get_corpus_documents()
        if not corpus_docs:
            logger.warning("Corpus documents not available in startup cache.")
            return list(dense_docs_by_id.values())[:top_k]

        bm25_start = time.perf_counter()
        bm25_scored: list[tuple[RetrievedDocument, float]] = []
        for doc in corpus_docs:
            score = bm25.score(query, doc.text)
            if score > 0:
                bm25_scored.append((doc, score))
        bm25_ms = (time.perf_counter() - bm25_start) * 1000

        merge_start = time.perf_counter()
        bm25_scored.sort(key=lambda item: item[1], reverse=True)
        sparse_limit = max(top_k * 6, 30)
        sparse_candidates = bm25_scored[:sparse_limit]

        sparse_rank_by_id: dict[str, int] = {}
        sparse_docs_by_id: dict[str, RetrievedDocument] = {}
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

        merged_docs.sort(key=lambda doc: doc.score, reverse=True)

        source_limits = _get_source_limits(intent)
        source_counts: dict[str, int] = {}
        final_docs: list[RetrievedDocument] = []

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

        merge_ms = (time.perf_counter() - merge_start) * 1000
        total_ms = (time.perf_counter() - overall_start) * 1000

        logger.info(
            "Hybrid retrieved %s docs | intent=%s | dense_candidates=%s | sparse_candidates=%s | latency_ms total=%.2f embed=%.2f qdrant=%.2f bm25=%.2f merge=%.2f",
            len(final_docs),
            intent,
            len(dense_docs_by_id),
            len(sparse_candidates),
            total_ms,
            embed_ms,
            qdrant_ms,
            bm25_ms,
            merge_ms,
        )
        return final_docs

    except ResponseHandlingException as error:
        logger.error("Qdrant connection error: %s", error)
        raise ConnectionError("Cannot connect to vector database")
    except Exception as error:
        logger.error("Error during retrieval: %s", error, exc_info=True)
        return []
