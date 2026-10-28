"""Build a searchable index version for a repo: clone -> chunk -> embed -> FAISS + MySQL."""
from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.db import models as m
from app.github.client import GitHubClient
from app.ingest import repo_source
from app.ingest.chunking import RawChunk, chunk_issue
from app.retrieval.embedder import get_embedder
from app.retrieval.faiss_store import build_index, index_path, save_index

log = logging.getLogger(__name__)


def repo_dir(data_dir: str, owner: str, name: str) -> Path:
    return Path(data_dir) / "repos" / owner / name


def ready_version(db: Session, repo_id: int, commit_sha: str | None = None) -> m.IndexVersion | None:
    q = select(m.IndexVersion).where(
        m.IndexVersion.repo_id == repo_id, m.IndexVersion.status == m.IndexStatus.ready
    )
    if commit_sha:
        q = q.where(m.IndexVersion.commit_sha == commit_sha)
    return db.scalars(q.order_by(m.IndexVersion.id.desc())).first()


def collect_chunks(
    repo: m.Repository, checkout: Path, gh: GitHubClient | None, issue_limit: int, commit_limit: int
) -> list[RawChunk]:
    chunks = repo_source.iter_file_chunks(checkout)
    if commit_limit > 0:
        chunks += repo_source.iter_commit_chunks(checkout, limit=commit_limit)
    if gh is not None and issue_limit > 0:
        # Issue history sharpens retrieval but is not essential: a rate limit, an auth problem or
        # an unexpected API shape must not throw away a perfectly good code index.
        try:
            for issue in gh.list_issues(repo.owner, repo.name, limit=issue_limit):
                chunks += chunk_issue(issue)
        except Exception as exc:  # noqa: BLE001
            log.warning("skipping issue history for %s/%s: %s", repo.owner, repo.name, exc)
    return chunks


def build_index_version(
    db: Session,
    repo: m.Repository,
    settings: Settings,
    gh: GitHubClient | None = None,
    issue_limit: int = 300,
    commit_limit: int = 300,
    force: bool = False,
) -> m.IndexVersion:
    """Index the repo at its current HEAD. Returns the existing version if HEAD is unchanged."""
    checkout = repo_source.clone_or_update(repo.clone_url, repo_dir(settings.data_dir, repo.owner, repo.name))
    sha = repo_source.head_sha(checkout)

    if not force:
        existing = ready_version(db, repo.id, sha)
        if existing is not None:
            log.info("index for %s/%s @ %s already ready", repo.owner, repo.name, sha[:8])
            return existing

    embedder = get_embedder(settings.embedding_model, settings.google_api_key)
    version = m.IndexVersion(
        repo_id=repo.id,
        commit_sha=sha,
        embedding_model=settings.embedding_model,
        embedding_dim=embedder.dim,
        status=m.IndexStatus.building,
    )
    db.add(version)
    db.commit()

    try:
        raw = collect_chunks(repo, checkout, gh, issue_limit, commit_limit)
        if not raw:
            raise RuntimeError("no indexable content found in repository")

        rows = [
            m.Chunk(
                repo_id=repo.id,
                index_version_id=version.id,
                source_type=m.SourceType(c.source_type),
                source_ref=c.source_ref[:512],
                start_line=c.start_line,
                end_line=c.end_line,
                text=c.text,
                content_hash=c.content_hash,
                meta=c.meta or None,
            )
            for c in raw
        ]
        db.add_all(rows)
        db.flush()  # assigns chunk ids, which double as FAISS vector ids

        vectors = embedder.encode([c.embed_text() for c in raw])
        index = build_index(vectors, [r.id for r in rows], embedder.dim)
        path = index_path(settings.data_dir, repo.id, version.id)
        save_index(index, path)

        version.faiss_path = str(path)
        version.chunk_count = len(rows)
        version.status = m.IndexStatus.ready
        version.finished_at = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        db.commit()
        log.info("indexed %s/%s @ %s: %d chunks", repo.owner, repo.name, sha[:8], len(rows))
        return version
    except Exception as exc:  # noqa: BLE001 - record the failure, then re-raise
        db.rollback()
        version.status = m.IndexStatus.failed
        version.error = f"{type(exc).__name__}: {exc}"[:2000]
        version.finished_at = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        db.commit()
        raise
