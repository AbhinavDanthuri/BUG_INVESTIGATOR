from __future__ import annotations

import logging
import re

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import runner
from app.config import Settings, get_settings
from app.db import models as m
from app.db.session import get_session, session_scope
from app.github.client import GitHubClient, GitHubError, parse_issue_url
from app.ingest.indexer import build_index_version, ready_version
from app.llm import get_llm
from app.schemas import (
    AgentStepOut,
    EvidenceOut,
    IndexVersionOut,
    InvestigationCreate,
    InvestigationDetail,
    InvestigationOut,
    IssueOut,
    RepoCreate,
    RepoOut,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

_REPO_URL = re.compile(r"^(?:https?://github\.com/)?([\w.-]+)/([\w.-]+?)(?:\.git)?/?$")


def parse_repo_url(value: str) -> tuple[str, str]:
    m_ = _REPO_URL.match(value.strip())
    if not m_:
        raise ValueError(f"not a GitHub repo: {value!r}")
    return m_.group(1), m_.group(2)


# ---------- background jobs ----------
def _index_job(repo_id: int, settings: Settings) -> None:
    with session_scope() as db:
        repo = db.get(m.Repository, repo_id)
        if repo is None:
            return
        try:
            build_index_version(db, repo, settings, GitHubClient(settings.github_token))
        except Exception:  # noqa: BLE001 - already recorded on the index version row
            log.exception("indexing failed for repo %s", repo_id)


def _investigate_job(inv_id: int, settings: Settings) -> None:
    with session_scope() as db:
        runner.run(db, inv_id, get_llm(settings), settings)


# ---------- repos ----------
def _ensure_repo(db: Session, gh: GitHubClient, owner: str, name: str) -> m.Repository:
    repo = db.scalar(select(m.Repository).where(m.Repository.owner == owner, m.Repository.name == name))
    if repo is not None:
        return repo
    info = gh.get_repo(owner, name)
    repo = m.Repository(
        owner=owner,
        name=name,
        default_branch=info.get("default_branch", "main"),
        clone_url=info["clone_url"],
    )
    db.add(repo)
    db.flush()
    return repo


def _repo_out(db: Session, repo: m.Repository) -> RepoOut:
    latest = db.scalars(
        select(m.IndexVersion)
        .where(m.IndexVersion.repo_id == repo.id)
        .order_by(m.IndexVersion.id.desc())
        .limit(1)
    ).first()
    return RepoOut(
        id=repo.id,
        owner=repo.owner,
        name=repo.name,
        default_branch=repo.default_branch,
        latest_index=IndexVersionOut.model_validate(latest) if latest else None,
    )


@router.post("/repos", response_model=RepoOut, status_code=202)
def register_repo(
    body: RepoCreate,
    background: BackgroundTasks,
    db: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
):
    """Register a repo and start indexing it in the background."""
    try:
        owner, name = parse_repo_url(body.repo_url)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    try:
        repo = _ensure_repo(db, GitHubClient(settings.github_token), owner, name)
    except GitHubError as e:
        raise HTTPException(502, str(e)) from e
    db.commit()
    background.add_task(_index_job, repo.id, settings)
    return _repo_out(db, repo)


@router.get("/repos", response_model=list[RepoOut])
def list_repos(db: Session = Depends(get_session)):
    repos = db.scalars(select(m.Repository).order_by(m.Repository.id.desc())).all()
    return [_repo_out(db, r) for r in repos]


@router.post("/repos/{repo_id}/reindex", response_model=RepoOut, status_code=202)
def reindex_repo(
    repo_id: int,
    background: BackgroundTasks,
    db: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
):
    repo = db.get(m.Repository, repo_id)
    if repo is None:
        raise HTTPException(404, "repository not found")
    background.add_task(_index_job, repo.id, settings)
    return _repo_out(db, repo)


# ---------- investigations ----------
@router.post("/investigations", response_model=InvestigationOut, status_code=202)
def create_investigation(
    body: InvestigationCreate,
    background: BackgroundTasks,
    db: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
):
    """Queue an investigation. Indexes the repo first if it has never been indexed."""
    try:
        ref = parse_issue_url(body.issue_url)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e

    gh = GitHubClient(settings.github_token)
    try:
        repo = _ensure_repo(db, gh, ref.owner, ref.repo)
        issue_json = gh.get_issue(ref)
    except GitHubError as e:
        raise HTTPException(502, str(e)) from e

    issue = db.scalar(select(m.Issue).where(m.Issue.repo_id == repo.id, m.Issue.number == ref.number))
    if issue is None:
        issue = m.Issue(repo_id=repo.id, number=ref.number, html_url=issue_json["html_url"], state="", title="")
        db.add(issue)
    issue.title = issue_json["title"]
    issue.body = issue_json.get("body")
    issue.state = issue_json["state"]
    issue.labels = [lbl["name"] for lbl in issue_json.get("labels", [])]
    issue.author = (issue_json.get("user") or {}).get("login")
    db.flush()

    inv = m.Investigation(
        repo_id=repo.id,
        issue_id=issue.id,
        max_iterations=settings.max_iterations,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        prompt_version=settings.prompt_version,
    )
    db.add(inv)
    db.commit()

    if ready_version(db, repo.id) is None:
        background.add_task(_index_job, repo.id, settings)
    background.add_task(_investigate_job, inv.id, settings)
    return inv


@router.get("/investigations", response_model=list[InvestigationOut])
def list_investigations(db: Session = Depends(get_session)):
    return db.scalars(select(m.Investigation).order_by(m.Investigation.id.desc()).limit(50)).all()


@router.get("/investigations/{inv_id}", response_model=InvestigationDetail)
def get_investigation(inv_id: int, db: Session = Depends(get_session)):
    inv = db.get(m.Investigation, inv_id)
    if inv is None:
        raise HTTPException(404, "investigation not found")
    repo = db.get(m.Repository, inv.repo_id)
    issue = db.get(m.Issue, inv.issue_id)

    rows = db.execute(
        select(m.Evidence, m.Chunk)
        .join(m.Chunk, m.Chunk.id == m.Evidence.chunk_id)
        .where(m.Evidence.investigation_id == inv.id)
        .order_by(m.Evidence.rank)
    ).all()

    detail = InvestigationDetail.model_validate(inv)
    detail.repo_full_name = f"{repo.owner}/{repo.name}" if repo else None
    detail.issue = IssueOut.model_validate(issue) if issue else None
    detail.steps = [AgentStepOut.model_validate(s) for s in inv.steps]
    detail.evidence = [
        EvidenceOut(
            source_type=c.source_type.value,
            ref=c.source_ref,
            start_line=c.start_line,
            end_line=c.end_line,
            score=e.score,
            cited=e.cited_in_report,
            text=c.text[:2000],
        )
        for e, c in rows
    ]
    detail.report = inv.report
    return detail


@router.get("/investigations/{inv_id}/report.md", response_class=PlainTextResponse)
def get_report_markdown(inv_id: int, db: Session = Depends(get_session)):
    inv = db.get(m.Investigation, inv_id)
    if inv is None or inv.report is None:
        raise HTTPException(404, "no report for that investigation")
    return PlainTextResponse(inv.report.report_md, media_type="text/markdown")
