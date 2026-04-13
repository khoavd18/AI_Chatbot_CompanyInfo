from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field


@dataclass
class RetrievedDocument:
    id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    debug: bool = False
    top_k: int | None = None


class ScoreBreakdown(BaseModel):
    hybrid: float | None = None
    dense: float | None = None
    bm25: float | None = None
    rerank: float | None = None


class HistorySource(BaseModel):
    title: str
    source_type: str | None = None
    image_url: str | None = None
    video_url: str | None = None


class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)
    sources: list[HistorySource] = Field(default_factory=list)


class SourceItem(BaseModel):
    index: int
    title: str
    source_type: str
    chunk_type: str
    doc_id: str
    snippet: str
    image_url: str | None = None
    video_url: str | None = None
    scores: ScoreBreakdown = Field(default_factory=ScoreBreakdown)


class ChatLatency(BaseModel):
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    answer_ms: float = 0.0
    total_ms: float = 0.0


class DebugItem(BaseModel):
    stage: str
    rank: int
    title: str
    source_type: str
    chunk_type: str
    doc_id: str
    snippet: str
    scores: ScoreBreakdown = Field(default_factory=ScoreBreakdown)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    debug: bool = False
    top_k: int | None = None
    session_id: str | None = None
    history: list[ChatHistoryItem] = Field(default_factory=list)


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem] = Field(default_factory=list)
    debug: list[dict[str, Any]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    request_id: str
    sources: list[SourceItem] = Field(default_factory=list)
    show_media_preview: bool = False
    latency: ChatLatency = Field(default_factory=ChatLatency)
    debug: list[DebugItem] = Field(default_factory=list)
