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
from app.retrieval.ranking import rerank

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
        exclude_issue: int | None = None,
    ):
        self._db = db
        self._vi = load_version_index(db, version)
        # Always the model the index was built with: querying with a different one is meaningless.
        self._embedder = get_embedder(embedding_model or version.embedding_model, api_key)
        self._evidence_ids: dict[int, str] = {}  # chunk_id -> "E3", stable within a run
        # The index holds this repository's issue history, which includes the issue under
        # investigation and anything that answered it. Retrieving those is not investigation, it is
        # looking up the answer - and in the eval it is leakage, since every case is a closed issue
        # sitting in its own index. A search on flask#6143 returned "issue #6143" as its top hit.
        self._excluded_ref = f"#{exclude_issue}" if exclude_issue is not None else None
        self._mentions = re.compile(rf"#{exclude_issue}\b") if exclude_issue is not None else None

    def _is_about_the_issue_under_investigation(self, chunk) -> bool:
        if self._excluded_ref is None:
            return False
        if chunk.source_type not in (m.SourceType.issue, m.SourceType.pr):
            return False  # code and docs are the point; only the conversation is off limits
        return chunk.source_ref == self._excluded_ref or bool(self._mentions.search(chunk.text))

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
            if chunk is None or self._is_about_the_issue_under_investigation(chunk):
                return False
            return not wanted or chunk.source_type.value in wanted

        fused = rrf_fuse([[c for c in dense if keep(c)], [c for c in sparse if keep(c)]])
        # Fusion ranks by resemblance alone, which lets packaging files and one chatty test file
        # take most of the budget. Demote supporting files and cap per-file duplicates.
        fused = rerank(fused, lambda cid: self._vi.by_id[cid].source_ref)

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

    @staticmethod
    def _paths_match(indexed: str, wanted: str) -> bool:
        """Whether two paths name the same file, comparing from the right.

        Either may carry a prefix the other lacks. Issues are written by hand and frequently add
        the repository name: pallets/click#3832 points at "click/src/click/core.py" while the index
        holds "src/click/core.py". Matching only one direction missed every such reference.
        Comparison is by path component, so "core.py" never matches "my_core.py".
        """
        if indexed == wanted:
            return True
        a, b = indexed.split("/"), wanted.split("/")
        shared = min(len(a), len(b))
        return a[-shared:] == b[-shared:]

    def fetch_file(
        self, path: str, line: int | None = None, span: int = 40, limit: int = 2
    ) -> list[Evidence]:
        """Return indexed chunks for `path`, narrowed to those covering `line` when given.

        Paths are matched on a suffix, so "core.py" finds "src/click/core.py" - a reporter rarely
        writes the full path from the repository root.

        Without a line number this returns nothing. Handing back the top of a file is worse than
        useless: a Python module starts with imports, which say nothing about any bug, and those
        chunks then occupy evidence slots that real search results would have filled. An earlier
        version did exactly that and turned a correct verdict into "unclear".
        """
        wanted = path.strip().lstrip("./")
        if not wanted or line is None:
            return []

        matches = [
            c
            for c in self._vi.by_id.values()
            if c.source_type is m.SourceType.code or c.source_type is m.SourceType.doc
            if self._paths_match(c.source_ref, wanted)
        ]
        covering = [
            c
            for c in matches
            if c.start_line is not None
            and c.start_line - span <= line <= (c.end_line or c.start_line) + span
        ]
        if not covering:
            return []  # a stale line number: say nothing rather than something irrelevant

        # Closest chunk first, so a tight limit still lands on the named line.
        covering.sort(key=lambda c: abs((c.start_line or 0) - line))
        return [
            Evidence(
                id=self._label(c.id),
                source_type=c.source_type.value,
                ref=c.source_ref,
                text=c.text,
                start_line=c.start_line,
                end_line=c.end_line,
                score=1.0,  # an exact lookup, not a ranked guess
            )
            for c in covering[:limit]
        ]

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
            if (c := self._vi.by_id.get(cid))
            and not self._is_about_the_issue_under_investigation(c)
            and (not wanted or c.source_type.value in wanted)
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
