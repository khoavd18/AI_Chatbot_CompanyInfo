from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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


class SourceItem(BaseModel):
    index: int
    title: str
    source: str
    doc_id: str
    rerank_score: float | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem] = Field(default_factory=list)
    debug: list[dict[str, Any]] = Field(default_factory=list)