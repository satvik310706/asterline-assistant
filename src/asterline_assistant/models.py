"""Typed boundaries between the baseline research stages."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class CorpusChunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    category: str
    owner: str
    source_url: str
    published: str
    version: str
    text: str


class RetrievedChunk(BaseModel):
    chunk: CorpusChunk
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float = 0.0


class EvidencePack(BaseModel):
    query: str
    chunks: list[RetrievedChunk]


class AnswerDraft(BaseModel):
    answer: str
    cited_chunk_ids: list[str] = Field(default_factory=list)
    abstained: bool = False
    provider_error: bool = False
    degraded: bool = False
    attempts: int = 0
    error_class: str | None = None
    primary_error: str | None = None
    fallback_error: str | None = None
    trace_metadata: dict[str, str | int | bool | None] = Field(default_factory=dict)


class ResearchResponse(BaseModel):
    answer: str
    evidence: list[RetrievedChunk]
    memory_used: list[str] = Field(default_factory=list)
    memory_written: list[str] = Field(default_factory=list)
    abstained: bool = False
    provider: str
    model: str
    degraded: bool = False
    attempts: int = 0
    error_class: str | None = None
    primary_error: str | None = None
    fallback_error: str | None = None
    provider_error: bool = False


class MemoryRecord(BaseModel):
    id: int | None = None
    user_id: str
    type: Literal["preference", "project", "task", "user_claim"]
    content: str
    status: Literal["active", "superseded", "deleted"] = "active"
    created_at: str = ""
    updated_at: str = ""
    last_confirmed_at: str = ""
    source_session: str = ""
    source_turn: str = ""
    confidence: float = 1.0
    is_authoritative: bool = False

