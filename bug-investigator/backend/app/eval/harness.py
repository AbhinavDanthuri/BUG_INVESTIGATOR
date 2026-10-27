"""Eval harness: replay closed issues whose real fix is known, and score file localization.

Two systems are scored on the same cases:
- `agent`         : the full LangGraph investigation; prediction = report.affected_files
- `bm25_baseline` : BM25 over the issue text alone; prediction = files of the top code chunks

The comparison is the point. A number with no baseline says nothing about whether the agent helps.
"""
from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import runner
from app.config import Settings
from app.db import models as m
from app.eval.metrics import file_recall, hit_at_k, summarize
from app.github.client import GitHubClient, GitHubError, IssueRef
from app.ingest.indexer import ready_version
from app.llm import LLM
from app.retrieval.hybrid import BM25OnlyRetriever

log = logging.getLogger(__name__)


def build_eval_cases(
    db: Session, repo: m.Repository, gh: GitHubClient, limit: int = 20, max_gold_files: int = 5
) -> list[m.EvalCase]:
    """Find closed issues linked to exactly one small fix PR, and record the files it changed."""
    created: list[m.EvalCase] = []
    for issue in gh.list_issues(repo.owner, repo.name, limit=limit * 10, state="closed"):
        if len(created) >= limit:
            break
        if issue["is_pr"]:
            continue
        if db.scalar(
            select(m.EvalCase).where(
                m.EvalCase.repo_id == repo.id, m.EvalCase.issue_number == issue["number"]
            )
        ):
            continue
        try:
            prs = gh.issue_timeline_closers(repo.owner, repo.name, issue["number"])
        except GitHubError as exc:
            log.warning("timeline unavailable for #%s: %s", issue["number"], exc)
            continue
        if len(prs) != 1:  # ambiguous ground truth
            continue
        try:
            files = gh.pr_files(repo.owner, repo.name, prs[0])
        except GitHubError:
            continue
        files = [f for f in files if not f.startswith("test") and "/test" not in f]
        if not files or len(files) > max_gold_files:
            continue
        case = m.EvalCase(
            repo_id=repo.id, issue_number=issue["number"], gold_pr_number=prs[0], gold_files=files
        )
        db.add(case)
        created.append(case)
    db.commit()
    return created


def _ensure_issue(db: Session, repo: m.Repository, gh: GitHubClient, number: int) -> m.Issue:
    issue = db.scalar(select(m.Issue).where(m.Issue.repo_id == repo.id, m.Issue.number == number))
    if issue is not None:
        return issue
    raw = gh.get_issue(IssueRef(repo.owner, repo.name, number))
    issue = m.Issue(
        repo_id=repo.id,
        number=number,
        title=raw["title"],
        body=raw.get("body"),
        state=raw["state"],
        labels=[lbl["name"] for lbl in raw.get("labels", [])],
        html_url=raw["html_url"],
    )
    db.add(issue)
    db.commit()
    return issue


def run_baseline(db: Session, repo: m.Repository, issue: m.Issue, k: int = 10) -> list[str]:
    """BM25 over the issue text; prediction is the distinct code files of the top hits, in order."""
    version = ready_version(db, repo.id)
    if version is None:
        raise RuntimeError("no ready index")
    retriever = BM25OnlyRetriever(db, version)
    files: list[str] = []
    for ev in retriever.search(f"{issue.title}\n{issue.body or ''}", ["code"], k=k):
        if ev.ref not in files:
            files.append(ev.ref)
    return files


def score(predicted: list[str], gold: list[str]) -> dict:
    return {
        "hit_at_1": hit_at_k(predicted, gold, 1),
        "hit_at_3": hit_at_k(predicted, gold, 3),
        "file_recall": file_recall(predicted, gold),
    }


def run_eval(
    db: Session,
    repo: m.Repository,
    settings: Settings,
    llm: LLM,
    gh: GitHubClient,
    systems: tuple[str, ...] = ("agent", "bm25_baseline"),
    limit: int | None = None,
) -> dict:
    """Score every eval case for this repo. Returns {system: summary}."""
    cases = list(
        db.scalars(select(m.EvalCase).where(m.EvalCase.repo_id == repo.id).order_by(m.EvalCase.id))
    )
    if limit:
        cases = cases[:limit]
    results: dict[str, list[dict]] = {s: [] for s in systems}

    for case in cases:
        issue = _ensure_issue(db, repo, gh, case.issue_number)

        if "bm25_baseline" in systems:
            try:
                row = score(run_baseline(db, repo, issue), case.gold_files)
                db.add(m.EvalResult(eval_case_id=case.id, system="bm25_baseline", **row))
                results["bm25_baseline"].append(row)
            except Exception:  # noqa: BLE001
                log.exception("baseline failed for issue #%s", case.issue_number)

        if "agent" in systems:
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
            runner.run(db, inv.id, llm, settings)
            db.refresh(inv)
            if inv.status == m.RunStatus.completed and inv.report is not None:
                row = score(inv.report.affected_files or [], case.gold_files)
                db.add(m.EvalResult(eval_case_id=case.id, investigation_id=inv.id, system="agent", **row))
                results["agent"].append(row)
            else:
                log.warning("agent run failed for issue #%s: %s", case.issue_number, inv.error)
        db.commit()

    return {system: summarize(rows) for system, rows in results.items()}


def format_summary(summary: dict) -> str:
    lines = [f"{'system':<16}{'n':>4}{'hit@1':>9}{'hit@3':>9}{'recall':>9}", "-" * 47]
    for system, s in summary.items():
        lines.append(f"{system:<16}{s['n']:>4}{s['hit@1']:>9.2f}{s['hit@3']:>9.2f}{s['recall']:>9.2f}")
    return "\n".join(lines)
