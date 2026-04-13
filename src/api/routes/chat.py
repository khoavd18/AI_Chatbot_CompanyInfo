from __future__ import annotations

import logging
import os
import re
import time
import unicodedata
import uuid

from fastapi import APIRouter, HTTPException, Request

from src.core.schema import ChatLatency, ChatRequest, ChatResponse, DebugItem, ScoreBreakdown, SourceItem
from src.core.setting_loader import load_settings
from src.core.startup import get_bm25, get_reranker
from src.llm.generator import generate_answer
from src.llm.source_answer import compose_grounded_answer
from src.rag.retrieval.context_builder import ContextBuilder
from src.rag.retrieval.hybrid_retriever import hybrid_retrieve

settings = load_settings()
logger = logging.getLogger("chat")
router = APIRouter()

MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "500"))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
MAX_RETRIEVAL_TOP_K = int(os.getenv("MAX_RETRIEVAL_TOP_K", "20"))
DEMO_RATE_LIMIT_ENABLED = os.getenv("DEMO_RATE_LIMIT_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
RERANKING_TOP_K = int(settings.get("reranking", {}).get("top_k", 5))
DEFAULT_RETRIEVAL_TOP_K = int(settings.get("retrieval", {}).get("top_k", 10))
LLM_FALLBACK_ENABLED = bool(settings.get("llm", {}).get("enable_fallback", False))
LLM_FALLBACK_MIN_RELEVANT_DOCS = 2

MEDIA_IMAGE_FIELDS = [
    "project_thumbnail_url",
    "news_thumbnail_url",
    "company_thumbnail_url",
    "architecture_type_image_url",
    "interior_image_url",
    "hero_slide_image_url",
]
MEDIA_VIDEO_FIELDS = [
    "hero_slide_video_url",
]
MEDIA_MARKERS = ["hinh anh", "anh", "image", "thumbnail", "video", "clip", "media"]
DETAIL_MARKERS = ["chi tiet", "thong tin", "gioi thieu", "noi ro hon", "them thong tin"]
PROJECT_MARKERS = ["du an", "cong trinh"]
NEWS_MARKERS = ["tin tuc", "bai viet", "news"]
STYLE_MARKERS = ["phong cach", "style", "japandi", "modern", "luxury", "tan co dien"]
COMPANY_FACTUAL_MARKERS = [
    "cong ty",
    "hotline",
    "email",
    "dia chi",
    "website",
    "gio lam viec",
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
AMBIGUOUS_FOLLOWUP_MARKERS = [
    " do",
    " nay",
    " kia",
    " o tren",
    " vua roi",
    " vua noi",
    " tiep theo",
    " hinh anh dau",
    " video dau",
    " link dau",
    " chi tiet hon",
    " them thong tin",
    " du an do",
    " cong trinh do",
    " bai viet do",
    " noi ro hon",
]
GREETING_MARKERS = {
    "xin chao",
    "xin chao ban",
    "chao",
    "chao ban",
    "hello",
    "hello ban",
    "hi",
    "hi ban",
    "hey",
    "alo",
    "alo alo",
}
GREETING_FILLER_TOKENS = {
    "xin",
    "chao",
    "hello",
    "hi",
    "hey",
    "alo",
    "ban",
    "nhe",
    "nha",
    "oi",
    "a",
    "ah",
    "ad",
    "admin",
    "team",
    "em",
    "anh",
    "chi",
    "ca",
}
GREETING_INFO_MARKERS = [
    "cong ty",
    "du an",
    "cong trinh",
    "tin tuc",
    "bai viet",
    "dia chi",
    "email",
    "hotline",
    "website",
    "gio lam viec",
    "hinh anh",
    "video",
    "phong cach",
    "style",
    "thong tin",
    "gioi thieu",
    "chi tiet",
    "bao nhieu",
    "tong so",
    "danh muc",
    "loai hinh",
]

_demo_rate_limit_storage: dict[str, list[float]] = {}


if DEMO_RATE_LIMIT_ENABLED:
    logger.warning("Demo in-memory rate limit is enabled. Do not use this mode in production.")


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


def _normalized_tokens(text: str) -> list[str]:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", _normalize(text))
    return [token for token in cleaned.split() if token]


def _is_greeting_only(question: str) -> bool:
    tokens = _normalized_tokens(question)
    if not tokens:
        return False

    normalized_question = " ".join(tokens)
    if _contains_any(normalized_question, GREETING_INFO_MARKERS):
        return False

    if normalized_question in GREETING_MARKERS:
        return True

    greeting_tokens = {"xin", "chao", "hello", "hi", "hey", "alo"}
    return (
        len(tokens) <= 5
        and any(token in greeting_tokens for token in tokens)
        and all(token in GREETING_FILLER_TOKENS for token in tokens)
    )


def _safe_log_text(text: str) -> str:
    return text.encode("unicode_escape", errors="ignore").decode("ascii")


def _clip_text(text: str, limit: int = 220) -> str:
    value = (text or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "..."


def _maybe_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_non_empty(values) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _source_title(metadata: dict) -> str:
    return (
        metadata.get("project_name")
        or metadata.get("news_item_title")
        or metadata.get("company_name")
        or metadata.get("architecture_type_name")
        or metadata.get("interior_name")
        or metadata.get("project_category_name")
        or metadata.get("news_category_name")
        or metadata.get("hero_slide_title")
        or "Không rõ"
    )


def _extract_image_url(metadata: dict) -> str | None:
    return _first_non_empty(metadata.get(field) for field in MEDIA_IMAGE_FIELDS)


def _extract_video_url(metadata: dict) -> str | None:
    return _first_non_empty(metadata.get(field) for field in MEDIA_VIDEO_FIELDS)


def _score_breakdown(document) -> ScoreBreakdown:
    metadata = document.metadata or {}
    return ScoreBreakdown(
        hybrid=_maybe_float(document.score),
        dense=_maybe_float(metadata.get("dense_score")),
        bm25=_maybe_float(metadata.get("bm25_score")),
        rerank=_maybe_float(metadata.get("rerank_score")),
    )


def _serialize_source(index: int, document) -> SourceItem:
    metadata = document.metadata or {}
    return SourceItem(
        index=index,
        title=_source_title(metadata),
        source_type=str(metadata.get("type", "unknown")),
        chunk_type=str(metadata.get("chunk_type", "unknown")),
        doc_id=str(document.id),
        snippet=_clip_text(document.text),
        image_url=_extract_image_url(metadata),
        video_url=_extract_video_url(metadata),
        scores=_score_breakdown(document),
    )


def _build_debug_items(stage: str, documents: list) -> list[DebugItem]:
    return [
        DebugItem(
            stage=stage,
            rank=index,
            title=_source_title(document.metadata or {}),
            source_type=str((document.metadata or {}).get("type", "unknown")),
            chunk_type=str((document.metadata or {}).get("chunk_type", "unknown")),
            doc_id=str(document.id),
            snippet=_clip_text(document.text, limit=180),
            scores=_score_breakdown(document),
        )
        for index, document in enumerate(documents, start=1)
    ]


def _check_demo_rate_limit(client_ip: str) -> bool:
    if not DEMO_RATE_LIMIT_ENABLED:
        return True

    current_time = time.time()
    minute_ago = current_time - 60
    timestamps = _demo_rate_limit_storage.setdefault(client_ip, [])
    _demo_rate_limit_storage[client_ip] = [timestamp for timestamp in timestamps if timestamp > minute_ago]

    if len(_demo_rate_limit_storage[client_ip]) >= RATE_LIMIT_PER_MINUTE:
        return False

    _demo_rate_limit_storage[client_ip].append(current_time)
    return True


def _resolve_top_k(requested_top_k: int | None) -> int:
    top_k = requested_top_k or DEFAULT_RETRIEVAL_TOP_K
    return max(1, min(top_k, MAX_RETRIEVAL_TOP_K))


def _empty_response(answer: str, session_id: str, request_id: str, latency: ChatLatency) -> ChatResponse:
    return ChatResponse(
        answer=answer,
        sources=[],
        show_media_preview=False,
        session_id=session_id,
        request_id=request_id,
        latency=latency,
        debug=[],
    )


def _preferred_source_types(question: str) -> list[str]:
    preferred: list[str] = []

    if _contains_any(question, ["danh muc du an", "loai hinh du an"]):
        preferred.extend(["project_category", "project"])
    if _contains_any(question, PROJECT_MARKERS):
        preferred.extend(["project", "project_category"])
    if _contains_any(question, ["danh muc tin tuc", "danh muc bai viet", "chuyen muc tin tuc"]):
        preferred.extend(["news_category", "news"])
    if _contains_any(question, NEWS_MARKERS):
        preferred.extend(["news", "news_category"])
    if _contains_any(question, COMPANY_FACTUAL_MARKERS):
        preferred.append("company_info")
    if _contains_any(question, STYLE_MARKERS):
        preferred.extend(["interior_style", "architecture_type", "project"])
    if _contains_any(question, MEDIA_MARKERS):
        preferred.extend(["project", "news", "hero_slide", "company_info", "interior_style", "architecture_type"])

    deduped = []
    seen = set()
    for source_type in preferred:
        if source_type in seen:
            continue
        seen.add(source_type)
        deduped.append(source_type)
    return deduped


def _document_has_media(document) -> bool:
    metadata = document.metadata or {}
    return bool(_extract_image_url(metadata) or _extract_video_url(metadata))


def _should_show_media_preview(question: str, documents: list) -> bool:
    return _contains_any(question, MEDIA_MARKERS) and any(_document_has_media(document) for document in documents)


def _filter_documents_for_answer(question: str, documents: list) -> list:
    if not documents:
        return []

    preferred_types = _preferred_source_types(question)

    filtered_documents = []
    if _contains_any(question, MEDIA_MARKERS):
        filtered_documents = [document for document in documents if _document_has_media(document)]

    if not filtered_documents and preferred_types:
        filtered_documents = [
            document
            for document in documents
            if str((document.metadata or {}).get("type", "")) in preferred_types
        ]

    if not filtered_documents:
        return list(documents)

    if len(filtered_documents) >= 2 or len(documents) <= 1:
        return list(filtered_documents)

    seen_ids = {str(document.id) for document in filtered_documents}
    blended_documents = list(filtered_documents)
    for document in documents:
        document_id = str(document.id)
        if document_id in seen_ids:
            continue
        if len(blended_documents) >= min(2, len(documents), RERANKING_TOP_K):
            break
        blended_documents.append(document)
        seen_ids.add(document_id)

    return blended_documents


def _fallback_intent(question: str) -> str:
    if _contains_any(question, MEDIA_MARKERS):
        return "media"
    if _contains_any(question, PROJECT_MARKERS):
        return "project"
    if _contains_any(question, NEWS_MARKERS):
        return "news"
    if _contains_any(question, STYLE_MARKERS):
        return "style"
    if _contains_any(question, COMPANY_FACTUAL_MARKERS):
        return "company"
    return "general"


def _fallback_allowed_source_types(intent: str) -> set[str]:
    mapping = {
        "project": {"project", "project_category"},
        "news": {"news", "news_category"},
        "style": {"interior_style", "architecture_type", "project"},
    }
    return mapping.get(intent, set())


def _relevant_fallback_documents(intent: str, documents: list) -> list:
    allowed_source_types = _fallback_allowed_source_types(intent)
    if not allowed_source_types:
        return []

    return [
        document
        for document in documents
        if str((document.metadata or {}).get("type", "")) in allowed_source_types
    ]


def _llm_fallback_policy(question: str, documents: list) -> tuple[bool, str, str]:
    intent = _fallback_intent(question)
    if intent not in {"project", "news", "style"}:
        return False, intent, "intent_blocked"

    relevant_documents = _relevant_fallback_documents(intent, documents)
    if len(relevant_documents) < LLM_FALLBACK_MIN_RELEVANT_DOCS:
        return False, intent, "insufficient_evidence"

    return True, intent, "allowed"


def _grounded_only_message(intent: str, reason: str) -> str:
    if intent == "company":
        return (
            "Câu hỏi này thuộc nhóm thông tin công ty/fact nên mình sẽ không dùng LLM fallback. "
            "Bạn có thể hỏi rõ hơn theo địa chỉ, hotline, email, giờ làm việc hoặc tên dự án cụ thể."
        )

    if reason == "insufficient_evidence":
        if intent == "project":
            return (
                "Mình đã tìm thấy một ít nguồn nhưng chưa đủ evidence liên quan để dùng LLM fallback an toàn. "
                "Bạn có thể hỏi rõ hơn theo tên dự án, địa điểm hoặc loại công trình."
            )
        if intent == "news":
            return (
                "Mình đã tìm thấy một ít nguồn nhưng chưa đủ evidence liên quan để dùng LLM fallback an toàn. "
                "Bạn có thể nêu rõ tên bài viết, chủ đề hoặc danh mục tin tức."
            )
        if intent == "style":
            return (
                "Mình đã tìm thấy một ít nguồn nhưng chưa đủ evidence liên quan để dùng LLM fallback an toàn. "
                "Bạn có thể nêu rõ tên phong cách hoặc loại hình thiết kế bạn muốn xem."
            )

    return (
        "Mình chưa có câu trả lời grounded đủ chắc từ dữ liệu hiện tại, nên sẽ không dùng LLM fallback. "
        "Bạn có thể hỏi rõ hơn theo tên dự án, địa điểm, loại công trình, thông tin công ty hoặc yêu cầu hình ảnh."
    )


def _is_ambiguous_followup(question: str) -> bool:
    return _contains_any(question, AMBIGUOUS_FOLLOWUP_MARKERS)


def _latest_history_source(history: list, preferred_types: list[str] | None = None):
    preferred_types = preferred_types or []

    for allowed_types in (preferred_types, []):
        for item in reversed(history):
            if getattr(item, "role", None) != "assistant":
                continue
            for source in getattr(item, "sources", []) or []:
                if not getattr(source, "title", None):
                    continue
                if allowed_types and getattr(source, "source_type", None) not in allowed_types:
                    continue
                return source
    return None


def _unique_history_sources(sources: list) -> list:
    unique_sources = []
    seen = set()

    for source in sources:
        title = getattr(source, "title", None)
        source_type = getattr(source, "source_type", None)
        if not title:
            continue
        key = (_normalize(title), source_type or "")
        if key in seen:
            continue
        seen.add(key)
        unique_sources.append(source)

    return unique_sources


def _latest_relevant_history_sources(history: list, preferred_types: list[str] | None = None) -> list:
    preferred_types = preferred_types or []

    for item in reversed(history):
        if getattr(item, "role", None) != "assistant":
            continue

        raw_sources = list(getattr(item, "sources", []) or [])
        if not raw_sources:
            continue

        filtered_sources = [
            source
            for source in raw_sources
            if getattr(source, "title", None)
            and (not preferred_types or getattr(source, "source_type", None) in preferred_types)
        ]

        if filtered_sources:
            return _unique_history_sources(filtered_sources)
        if not preferred_types:
            return _unique_history_sources(raw_sources)

    return []


def _clarification_label(source_type: str | None) -> str:
    mapping = {
        "project": "dự án",
        "project_category": "dự án",
        "news": "bài viết",
        "news_category": "bài viết",
        "company_info": "mục thông tin công ty",
        "interior_style": "phong cách nội thất",
        "architecture_type": "phong cách kiến trúc",
        "hero_slide": "nội dung",
    }
    return mapping.get(source_type or "", "mục")


def _maybe_followup_clarification(question: str, history: list) -> str | None:
    if not history or not _is_ambiguous_followup(question):
        return None

    preferred_types = _preferred_source_types(question)
    sources = _latest_relevant_history_sources(history, preferred_types)
    if len(sources) <= 1:
        return None

    label = _clarification_label(getattr(sources[0], "source_type", None))
    lines = [f"Mình thấy ở lượt trước có nhiều {label}. Bạn muốn xem {label} nào?"]
    lines.extend(f"- {source.title}" for source in sources[:4] if getattr(source, "title", None))
    return "\n".join(lines)


def _source_type_label(source_type: str | None) -> str:
    mapping = {
        "project": "dự án",
        "news": "bài viết",
        "company_info": "công ty",
        "architecture_type": "phong cách kiến trúc",
        "interior_style": "phong cách nội thất",
        "hero_slide": "hero slide",
        "project_category": "danh mục dự án",
        "news_category": "danh mục tin tức",
    }
    return mapping.get(source_type or "", "nội dung")


def _rewrite_followup_query(question: str, history: list) -> str:
    if not history or not _is_ambiguous_followup(question):
        return question.strip()

    preferred_types = _preferred_source_types(question)
    latest_source = _latest_history_source(history, preferred_types)
    if latest_source is None:
        return question.strip()

    normalized_question = _normalize(question)
    normalized_title = _normalize(latest_source.title)
    if normalized_title and normalized_title in normalized_question:
        return question.strip()

    source_label = _source_type_label(getattr(latest_source, "source_type", None))

    if _contains_any(question, MEDIA_MARKERS):
        media_prefix = "video" if _contains_any(question, ["video", "clip"]) else "hình ảnh"
        return f"{media_prefix} {source_label} {latest_source.title}".strip()

    if _contains_any(question, DETAIL_MARKERS):
        return f"thông tin chi tiết về {source_label} {latest_source.title}".strip()

    return f"{question.strip()} {latest_source.title}".strip()


def _history_to_prompt_text(history: list, limit: int = 6) -> str:
    if not history:
        return ""

    lines = []
    for item in history[-limit:]:
        role = "Người dùng" if getattr(item, "role", None) == "user" else "Trợ lý"
        content = _clip_text(getattr(item, "content", ""), limit=220)
        source_titles = [source.title for source in (getattr(item, "sources", []) or []) if getattr(source, "title", None)]
        if source_titles:
            lines.append(f"{role}: {content} | Nguồn gần nhất: {', '.join(source_titles[:3])}")
        else:
            lines.append(f"{role}: {content}")

    return "\n".join(lines)


def _run_chat_turn(
    *,
    question: str,
    session_id: str,
    request_id: str,
    debug: bool,
    top_k: int | None,
    history: list,
) -> ChatResponse:
    normalized_question = question.strip()
    if not normalized_question:
        raise HTTPException(status_code=400, detail="Vui lòng nhập câu hỏi.")

    if len(normalized_question) > MAX_QUERY_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Câu hỏi quá dài. Vui lòng giới hạn dưới {MAX_QUERY_LENGTH} ký tự.",
        )

    if _is_greeting_only(normalized_question):
        total_ms = 0.0
        logger.info(
            "request_id=%s session_id=%s answer_strategy=greeting history_items=%s latency_ms total=%.2f",
            request_id,
            session_id,
            len(history),
            total_ms,
        )
        return _empty_response(
            answer=(
                "Xin chao! Minh co the ho tro thong tin ve cong ty, du an, tin tuc, "
                "phong cach thiet ke va hinh anh. Ban muon tim noi dung nao?"
            ),
            session_id=session_id,
            request_id=request_id,
            latency=ChatLatency(total_ms=total_ms),
        )

    clarification = _maybe_followup_clarification(normalized_question, history)
    if clarification:
        total_ms = 0.0
        logger.info(
            "request_id=%s session_id=%s answer_strategy=clarify history_items=%s latency_ms total=%.2f",
            request_id,
            session_id,
            len(history),
            total_ms,
        )
        return ChatResponse(
            answer=clarification,
            sources=[],
            show_media_preview=False,
            session_id=session_id,
            request_id=request_id,
            latency=ChatLatency(
                retrieval_ms=0.0,
                rerank_ms=0.0,
                answer_ms=0.0,
                total_ms=total_ms,
            ),
            debug=[],
        )

    rewritten_question = _rewrite_followup_query(normalized_question, history)
    retrieve_top_k = _resolve_top_k(top_k)
    overall_start = time.perf_counter()
    retrieval_ms = 0.0
    rerank_ms = 0.0
    answer_ms = 0.0

    logger.info(
        "request_id=%s session_id=%s question=%s rewritten_question=%s top_k=%s history_items=%s",
        request_id,
        session_id,
        _safe_log_text(normalized_question),
        _safe_log_text(rewritten_question),
        retrieve_top_k,
        len(history),
    )

    bm25 = get_bm25()
    reranker = get_reranker()

    if bm25 is None:
        logger.error("request_id=%s session_id=%s bm25_not_initialized", request_id, session_id)
        raise HTTPException(
            status_code=503,
            detail="Hệ thống chưa sẵn sàng. Vui lòng build index rồi thử lại.",
        )

    retrieval_start = time.perf_counter()
    retrieved_documents = hybrid_retrieve(rewritten_question, bm25, top_k=retrieve_top_k)
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

    if not retrieved_documents:
        latency = ChatLatency(
            retrieval_ms=retrieval_ms,
            rerank_ms=0.0,
            answer_ms=0.0,
            total_ms=(time.perf_counter() - overall_start) * 1000,
        )
        logger.warning(
            "request_id=%s session_id=%s no_documents latency_ms=%.2f",
            request_id,
            session_id,
            latency.total_ms,
        )
        return _empty_response(
            answer="Mình không tìm thấy thông tin phù hợp trong dữ liệu hiện có.",
            session_id=session_id,
            request_id=request_id,
            latency=latency,
        )

    debug_items = _build_debug_items("retrieved", retrieved_documents) if debug else []

    final_documents = list(retrieved_documents)
    rerank_start = time.perf_counter()
    if reranker is not None:
        final_documents = reranker.rerank(rewritten_question, list(retrieved_documents), top_k=RERANKING_TOP_K)
    else:
        final_documents = final_documents[:RERANKING_TOP_K]
    rerank_ms = (time.perf_counter() - rerank_start) * 1000

    if debug:
        debug_items.extend(_build_debug_items("final", final_documents))

    answer_documents = _filter_documents_for_answer(rewritten_question, final_documents)
    if debug:
        debug_items.extend(_build_debug_items("answer_context", answer_documents))

    answer_start = time.perf_counter()
    answer = compose_grounded_answer(rewritten_question, answer_documents)
    answer_strategy = "grounded"

    if not answer:
        fallback_allowed, fallback_intent, fallback_reason = _llm_fallback_policy(
            rewritten_question,
            answer_documents,
        )
        relevant_fallback_docs = _relevant_fallback_documents(fallback_intent, answer_documents)
        logger.info(
            "request_id=%s session_id=%s fallback_policy enabled=%s intent=%s reason=%s relevant_docs=%s answer_docs=%s",
            request_id,
            session_id,
            fallback_allowed,
            fallback_intent,
            fallback_reason,
            len(relevant_fallback_docs),
            len(answer_documents),
        )
        if not LLM_FALLBACK_ENABLED:
            answer = _grounded_only_message(fallback_intent, fallback_reason)
            answer_strategy = "grounded_only"
        elif not fallback_allowed:
            answer = _grounded_only_message(fallback_intent, fallback_reason)
            answer_strategy = f"fallback_blocked:{fallback_intent}:{fallback_reason}"
        else:
            context_builder = ContextBuilder(
                max_documents=RERANKING_TOP_K,
                max_context_length=3200,
            )
            context = context_builder.build(answer_documents)
            conversation_history = _history_to_prompt_text(history)

            if not context.strip():
                answer = "Mình không tìm thấy đủ dữ liệu để trả lời câu hỏi này."
            else:
                answer = generate_answer(context, normalized_question, conversation_history=conversation_history)
                answer_strategy = f"llm:{fallback_intent}"

    answer_ms = (time.perf_counter() - answer_start) * 1000
    total_ms = (time.perf_counter() - overall_start) * 1000

    sources = [
        _serialize_source(index, document)
        for index, document in enumerate(answer_documents, start=1)
    ]
    show_media_preview = _should_show_media_preview(rewritten_question, answer_documents)

    logger.info(
        "request_id=%s session_id=%s answer_strategy=%s retrieved=%s final=%s answer_docs=%s latency_ms total=%.2f retrieval=%.2f rerank=%.2f answer=%.2f",
        request_id,
        session_id,
        answer_strategy,
        len(retrieved_documents),
        len(final_documents),
        len(answer_documents),
        total_ms,
        retrieval_ms,
        rerank_ms,
        answer_ms,
    )

    return ChatResponse(
        answer=answer,
        sources=sources,
        show_media_preview=show_media_preview,
        session_id=session_id,
        request_id=request_id,
        latency=ChatLatency(
            retrieval_ms=retrieval_ms,
            rerank_ms=rerank_ms,
            answer_ms=answer_ms,
            total_ms=total_ms,
        ),
        debug=debug_items,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, req: Request):
    client_ip = req.client.host if req.client else "unknown"
    if not _check_demo_rate_limit(client_ip):
        logger.warning("request_id=%s client_ip=%s demo_rate_limit_exceeded", getattr(req.state, "request_id", "n/a"), client_ip)
        raise HTTPException(
            status_code=429,
            detail=(
                "Tốc độ request quá nhanh. Rate limit in-memory này chỉ dành cho demo local. "
                f"Giới hạn hiện tại là {RATE_LIMIT_PER_MINUTE} requests/phút."
            ),
        )

    request_id = getattr(req.state, "request_id", None) or str(uuid.uuid4())
    session_id = request.session_id or request_id

    try:
        return _run_chat_turn(
            question=request.query,
            session_id=session_id,
            request_id=request_id,
            debug=request.debug,
            top_k=request.top_k,
            history=request.history,
        )
    except HTTPException:
        raise
    except Exception as error:
        logger.error("request_id=%s session_id=%s error=%s", request_id, session_id, error, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi của bạn. Vui lòng thử lại sau.",
        )


def chat(question: str) -> str:
    response = _run_chat_turn(
        question=question,
        session_id=str(uuid.uuid4()),
        request_id=str(uuid.uuid4()),
        debug=False,
        top_k=None,
        history=[],
    )
    return response.answer
