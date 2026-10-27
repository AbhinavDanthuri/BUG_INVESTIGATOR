"""Hybrid retriever: FAISS (dense) + BM25 (sparse), merged with Reciprocal Rank Fusion.

Dense search understands paraphrase ("crashes on empty input"); BM25 nails exact identifiers and
error strings. Neither alone is enough for code, so we run both and fuse the rankings.
"""
from __future__ import annotations

import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import models as m
from app.retrieval.base import Evidence
from app.retrieval.embedder import get_embedder
from app.retrieval.faiss_store import load_index
from app.retrieval.faiss_store import search as faiss_search
from app.retrieval.fusion import rrf_fuse

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


def tokenize(text: str) -> list[str]:
    """Lowercased identifiers, plus camelCase/snake_case parts so `getUserById` matches `user`."""
    tokens: list[str] = []
    for tok in _TOKEN.findall(text):
        low = tok.lower()
        tokens.append(low)
        parts = [p for p in re.split(r"_|(?<=[a-z0-9])(?=[A-Z])", tok) if p]
        if len(parts) > 1:
            tokens.extend(p.lower() for p in parts)
    return tokens


class _VersionIndex:
    """Everything needed to search one index version, loaded once and reused."""

    def __init__(self, version: m.IndexVersion, chunks: list[m.Chunk]):
        from rank_bm25 import BM25Okapi

        self.version_id = version.id
        self.embedding_model = version.embedding_model
        self.faiss = load_index(version.faiss_path)
        self.chunk_ids = [c.id for c in chunks]
        self.by_id = {c.id: c for c in chunks}
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks]) if chunks else None

    def bm25_rank(self, query: str, k: int) -> list[int]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [self.chunk_ids[i] for i in ranked if scores[i] > 0]




_INDEX_CACHE: dict[int, _VersionIndex] = {}


def load_version_index(db: Session, version: m.IndexVersion) -> _VersionIndex:
    cached = _INDEX_CACHE.get(version.id)
    if cached is None:
        chunks = list(db.scalars(select(m.Chunk).where(m.Chunk.index_version_id == version.id).order_by(m.Chunk.id)))
        cached = _VersionIndex(version, chunks)
        _INDEX_CACHE[version.id] = cached
    return cached


def clear_index_cache() -> None:
    _INDEX_CACHE.clear()


class HybridRetriever:
    """Implements the Retriever protocol the agents depend on."""

    def __init__(
        self,
        db: Session,
        version: m.IndexVersion,
        embedding_model: str | None = None,
        api_key: str = "",
    ):
        self._db = db
        self._vi = load_version_index(db, version)
        # Always the model the index was built with: querying with a different one is meaningless.
        self._embedder = get_embedder(embedding_model or version.embedding_model, api_key)
        self._evidence_ids: dict[int, str] = {}  # chunk_id -> "E3", stable within a run

    def _label(self, chunk_id: int) -> str:
        if chunk_id not in self._evidence_ids:
            self._evidence_ids[chunk_id] = f"E{len(self._evidence_ids) + 1}"
        return self._evidence_ids[chunk_id]

    def search(self, query: str, source_types: list[str] | None = None, k: int = 8) -> list[Evidence]:
        pool = max(k * 4, 20)  # over-fetch, then filter and fuse down to k
        qvec = self._embedder.encode([query])[0]
        dense = [cid for cid, _ in faiss_search(self._vi.faiss, qvec, pool)]
        sparse = self._vi.bm25_rank(query, pool)

        wanted = set(source_types or [])
        def keep(cid: int) -> bool:
            chunk = self._vi.by_id.get(cid)
            return chunk is not None and (not wanted or chunk.source_type.value in wanted)

        fused = rrf_fuse([[c for c in dense if keep(c)], [c for c in sparse if keep(c)]])

        results: list[Evidence] = []
        for chunk_id, score in fused[:k]:
            chunk = self._vi.by_id[chunk_id]
            results.append(
                Evidence(
                    id=self._label(chunk_id),
                    source_type=chunk.source_type.value,
                    ref=chunk.source_ref,
                    text=chunk.text,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    score=round(score, 6),
                )
            )
        return results

    def chunk_id_for(self, evidence_id: str) -> int | None:
        for cid, label in self._evidence_ids.items():
            if label == evidence_id:
                return cid
        return None


class BM25OnlyRetriever(HybridRetriever):
    """Baseline for the eval: sparse search alone, no embeddings."""

    def search(self, query: str, source_types: list[str] | None = None, k: int = 8) -> list[Evidence]:
        wanted = set(source_types or [])
        ids = [
            cid
            for cid in self._vi.bm25_rank(query, k * 4)
            if (c := self._vi.by_id.get(cid)) and (not wanted or c.source_type.value in wanted)
        ]
        out: list[Evidence] = []
        for rank, cid in enumerate(ids[:k], start=1):
            c = self._vi.by_id[cid]
            out.append(
                Evidence(
                    id=self._label(cid),
                    source_type=c.source_type.value,
                    ref=c.source_ref,
                    text=c.text,
                    start_line=c.start_line,
                    end_line=c.end_line,
                    score=1.0 / rank,
                )
            )
        return out