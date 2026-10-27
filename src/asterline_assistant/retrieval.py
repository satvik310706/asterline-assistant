"""One-entry-per-chunk hybrid retrieval using Chroma and BM25."""

from __future__ import annotations

import re
from pathlib import Path

import chromadb
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi

from .models import CorpusChunk, RetrievedChunk

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-.][A-Za-z0-9]+)*%?")


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


class HybridRetriever:
    def __init__(self, chunks: list[CorpusChunk], chroma_path: Path) -> None:
        self.chunks = chunks
        self.embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.client = chromadb.PersistentClient(path=str(chroma_path))
        self.collection = self.client.get_or_create_collection("corpus_chunks")
        self.bm25 = BM25Okapi([tokenize(f"{c.title} {c.text}") for c in chunks])
        self._index()

    def _index(self) -> None:
        existing = self.collection.count()
        if existing == len(self.chunks):
            return
        if existing:
            self.client.delete_collection("corpus_chunks")
            self.collection = self.client.get_or_create_collection("corpus_chunks")
        documents = [f"{chunk.title} {chunk.text}" for chunk in self.chunks]
        embeddings = [[float(value) for value in vector] for vector in self.embedder.embed(documents)]
        self.collection.add(
            ids=[chunk.chunk_id for chunk in self.chunks],
            documents=documents,
            embeddings=embeddings,
            metadatas=[
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "title": c.title,
                    "category": c.category,
                    "published": c.published,
                    "version": c.version,
                }
                for c in self.chunks
            ],
        )

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        dense = self.collection.query(
            query_embeddings=[[float(value) for value in next(self.embedder.embed([query]))]], n_results=min(k * 3, len(self.chunks))
        )["ids"][0]
        bm25_scores = self.bm25.get_scores(tokenize(query))
        bm25 = sorted(range(len(self.chunks)), key=lambda i: bm25_scores[i], reverse=True)[: k * 3]
        positions: dict[str, RetrievedChunk] = {}
        for rank, chunk_id in enumerate(dense, start=1):
            positions.setdefault(chunk_id, RetrievedChunk(chunk=self._by_id(chunk_id))).dense_rank = rank
        for rank, index in enumerate(bm25, start=1):
            chunk = self.chunks[index]
            item = positions.setdefault(chunk.chunk_id, RetrievedChunk(chunk=chunk))
            item.bm25_rank = rank
        for item in positions.values():
            item.rrf_score = sum(1 / (60 + rank) for rank in (item.dense_rank, item.bm25_rank) if rank)
        return sorted(positions.values(), key=lambda item: item.rrf_score, reverse=True)[:k]

    def _by_id(self, chunk_id: str) -> CorpusChunk:
        return next(chunk for chunk in self.chunks if chunk.chunk_id == chunk_id)

