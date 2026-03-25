import logging
import os
import time
import uuid
from typing import Optional
import re

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.core.setting_loader import load_settings
from src.core.startup import get_bm25, get_reranker
from src.llm.generator import generate_answer
from src.rag.retrieval.context_builder import ContextBuilder
from src.rag.retrieval.hybrid_retriever import hybrid_retrieve

settings = load_settings()
logger = logging.getLogger("chat")
router = APIRouter()

MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "500"))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
RERANKING_TOP_K = settings.get("reranking", {}).get("top_k", 5)
DEFAULT_RETRIEVAL_TOP_K = settings.get("retrieval", {}).get("top_k", 10)

sessions = {}
rate_limit_storage = {}



def clean_context_text(text: str) -> str:
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def check_rate_limit(client_ip: str) -> bool:
    current_time = time.time()
    minute_ago = current_time - 60

    if client_ip not in rate_limit_storage:
        rate_limit_storage[client_ip] = []

    rate_limit_storage[client_ip] = [
        ts for ts in rate_limit_storage[client_ip] if ts > minute_ago
    ]

    if len(rate_limit_storage[client_ip]) >= RATE_LIMIT_PER_MINUTE:
        return False

    rate_limit_storage[client_ip].append(current_time)
    return True


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH, description="User's question")
    session_id: Optional[str] = Field(None, description="Session ID for conversation tracking")
    top_k: Optional[int] = Field(None, description="Optional retrieval top_k override")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Bot's answer")
    sources: list = Field(default_factory=list, description="Source documents")
    session_id: str = Field(..., description="Session ID")


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


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, req: Request):
    client_ip = req.client.host if req.client else "unknown"
    if not check_rate_limit(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=429,
            detail=f"Tốc độ request quá nhanh. Vui lòng thử lại sau. (Max {RATE_LIMIT_PER_MINUTE} requests/minute)"
        )

    question = request.query.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Vui lòng nhập câu hỏi.")

    session_id = request.session_id or str(uuid.uuid4())
    retrieve_top_k = request.top_k or DEFAULT_RETRIEVAL_TOP_K

    logger.info(f"Session {session_id}: Received question: {question}")

    try:
        bm25 = get_bm25()
        reranker = get_reranker()

        if bm25 is None:
            logger.error(f"Session {session_id}: BM25 not initialized!")
            raise HTTPException(
                status_code=503,
                detail="Hệ thống chưa sẵn sàng. Vui lòng thử lại sau."
            )

        logger.info(f"Session {session_id}: Running hybrid retrieval...")
        documents = hybrid_retrieve(question, bm25, top_k=retrieve_top_k)

        if not documents:
            logger.warning(f"Session {session_id}: No documents retrieved")
            return ChatResponse(
                answer="Mình không tìm thấy thông tin phù hợp trong dữ liệu hiện có nè.",
                sources=[],
                session_id=session_id
            )

        logger.info(f"Session {session_id}: Retrieved {len(documents)} documents from hybrid search")

        if reranker is not None:
            logger.info(f"Session {session_id}: Reranking documents...")
            documents = reranker.rerank(question, documents, top_k=RERANKING_TOP_K)
            logger.info(f"Session {session_id}: After reranking: {len(documents)} documents")
        else:
            logger.warning(f"Session {session_id}: Reranker not available, using hybrid scores only")
            documents = documents[:RERANKING_TOP_K]

        context_builder = ContextBuilder(
            max_documents=RERANKING_TOP_K,
            max_context_length=3200,
        )
        context = context_builder.build(documents)

        if not context.strip():
            logger.warning(f"Session {session_id}: Context is empty after context building")
            return ChatResponse(
                answer="Mình không tìm thấy đủ dữ liệu để trả lời câu hỏi này nè.",
                sources=[],
                session_id=session_id
            )

        answer = generate_answer(context, question)
        logger.info(f"Session {session_id}: Generated answer successfully")

        sources = [
            {
                "title": _source_title(doc.metadata),
                "type": doc.metadata.get("type"),
                "chunk_type": doc.metadata.get("chunk_type"),
                "text": doc.text[:220] + "..." if len(doc.text) > 220 else doc.text,
                "score": doc.score,
                "rerank_score": doc.metadata.get("rerank_score"),
                "metadata": doc.metadata,
            }
            for doc in documents
        ]

        if session_id not in sessions:
            sessions[session_id] = []
        sessions[session_id].append({
            "question": question,
            "answer": answer,
            "sources": sources,
        })

        return ChatResponse(
            answer=answer,
            sources=sources,
            session_id=session_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session {session_id}: Error in chat: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi của bạn. Vui lòng thử lại sau."
        )


def chat(question: str) -> str:
    if not question or not question.strip():
        logger.warning("Empty question received")
        return "Vui lòng nhập câu hỏi."

    if len(question) > MAX_QUERY_LENGTH:
        logger.warning(f"Query too long: {len(question)} characters")
        return f"Câu hỏi quá dài. Vui lòng giới hạn dưới {MAX_QUERY_LENGTH} ký tự."

    logger.info(f"Received question: {question}")

    try:
        bm25 = get_bm25()
        reranker = get_reranker()

        if bm25 is None:
            return "Hệ thống chưa sẵn sàng. Vui lòng thử lại sau."

        documents = hybrid_retrieve(question, bm25, top_k=DEFAULT_RETRIEVAL_TOP_K)

        if not documents:
            logger.warning("No documents retrieved for the question")
            return "Tôi không tìm thấy thông tin phù hợp trong dữ liệu hiện có."

        if reranker is not None:
            documents = reranker.rerank(question, documents, top_k=RERANKING_TOP_K)
        else:
            documents = documents[:RERANKING_TOP_K]

        context_builder = ContextBuilder(
            max_documents=RERANKING_TOP_K,
            max_context_length=3200,
        )
        context = context_builder.build(documents)

        if not context.strip():
            return "Mình không tìm thấy đủ dữ liệu để trả lời câu hỏi này nè."

        answer = generate_answer(context, question)
        logger.info("Generated answer successfully")
        return answer

    except Exception as e:
        logger.error(f"Error in chat function: {e}", exc_info=True)
        return "Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi của bạn. Vui lòng thử lại sau."