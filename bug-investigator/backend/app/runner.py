"""Execute an investigation end to end and persist every step.

Separated from the API so it can be driven by a background task, a CLI, or the eval harness.
"""
from __future__ import annotations

import datetime as dt
import logging

from sqlalchemy.orm import Session

from app.agents.graph import build_graph
from app.config import Settings
from app.db import models as m
from app.ingest.indexer import ready_version
from app.llm import LLM
from app.report import render_markdown
from app.retrieval.hybrid import HybridRetriever

log = logging.getLogger(__name__)


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def issue_payload(issue: m.Issue) -> dict:
    return {
        "number": issue.number,
        "title": issue.title,
        "body": issue.body or "",
        "labels": issue.labels or [],
        "url": issue.html_url,
    }


def _persist_trace(db: Session, inv: m.Investigation, trace: list[dict]) -> dict[int, int]:
    """Write agent_steps; returns {trace index -> step id} so evidence can point at a step."""
    step_ids: dict[int, int] = {}
    for seq, entry in enumerate(trace, start=1):
        step = m.AgentStep(
            investigation_id=inv.id,
            iteration=entry.get("iteration", 0),
            seq=seq,
            agent=m.AgentName(entry["agent"]),
            output=entry.get("output"),
            latency_ms=entry.get("latency_ms"),
        )
        db.add(step)
        db.flush()
        step_ids[seq - 1] = step.id
    return step_ids


def _persist_evidence(db: Session, inv: m.Investigation, retriever: HybridRetriever, state: dict) -> None:
    cited = set(state.get("analysis", {}).get("evidence_ids") or [])
    for rank, ev in enumerate(state.get("evidence") or [], start=1):
        chunk_id = retriever.chunk_id_for(ev["id"])
        if chunk_id is None:
            continue
        db.add(
            m.Evidence(
                investigation_id=inv.id,
                chunk_id=chunk_id,
                rank=rank,
                score=float(ev.get("score") or 0.0),
                cited_in_report=ev["id"] in cited,
            )
        )


def _persist_reviews(db: Session, inv: m.Investigation, trace: list[dict]) -> None:
    for entry in trace:
        if entry["agent"] != "critic":
            continue
        out = entry.get("output") or {}
        db.add(
            m.CriticReview(
                investigation_id=inv.id,
                iteration=entry.get("iteration", 0),
                verdict=m.Verdict(out.get("verdict", "revise")),
                scores=out.get("scores"),
                feedback=out.get("feedback"),
            )
        )


def _persist_report(db: Session, inv: m.Investigation, repo: m.Repository, issue: m.Issue, report: dict) -> m.Report:
    md = render_markdown(issue_payload(issue), report, f"{repo.owner}/{repo.name}")
    row = m.Report(
        investigation_id=inv.id,
        is_defect=report.get("is_defect", "unclear"),
        verdict_reason=report.get("verdict_reason"),
        root_cause=report.get("root_cause", ""),
        summary=(report.get("root_cause") or "")[:500],
        confidence=float(report.get("confidence") or 0.0),
        affected_files=report.get("affected_files") or [],
        low_confidence=bool(report.get("low_confidence")),
        report_md=md,
    )
    db.add(row)
    db.flush()

    fix = report.get("fix") or {}
    if fix.get("diff"):
        db.add(
            m.FixSuggestion(
                report_id=row.id,
                file_path=(fix.get("file") or "")[:512],
                diff=fix["diff"],
                explanation=fix.get("explanation"),
            )
        )
    for t in report.get("tests") or []:
        db.add(
            m.TestSuggestion(
                report_id=row.id,
                name=(t.get("name") or "test")[:200],
                code=t.get("code") or "",
                rationale=t.get("rationale"),
            )
        )
    return row


def run(db: Session, inv_id: int, llm: LLM, settings: Settings) -> m.Investigation:
    """Run one investigation. Failures are recorded on the row rather than raised to the caller."""
    inv = db.get(m.Investigation, inv_id)
    if inv is None:
        raise ValueError(f"investigation {inv_id} not found")
    repo = db.get(m.Repository, inv.repo_id)
    issue = db.get(m.Issue, inv.issue_id)

    inv.status = m.RunStatus.running
    inv.started_at = _now()
    db.commit()

    try:
        version = ready_version(db, repo.id)
        if version is None:
            raise RuntimeError(
                f"no ready index for {repo.owner}/{repo.name}; index the repository first"
            )
        inv.index_version_id = version.id

        retriever = HybridRetriever(
            db, version, api_key=settings.google_api_key, exclude_issue=issue.number
        )
        graph = build_graph(llm, retriever, settings.retrieval_top_k)
        state = graph.invoke(
            {
                "issue": issue_payload(issue),
                "max_iterations": inv.max_iterations,
                "iteration": 0,
                "trace": [],
            }
        )

        trace = state.get("trace") or []
        _persist_trace(db, inv, trace)
        _persist_evidence(db, inv, retriever, state)
        _persist_reviews(db, inv, trace)
        report = state.get("report") or {}
        _persist_report(db, inv, repo, issue, report)

        inv.iterations = state.get("iteration", 0)
        inv.confidence = float(report.get("confidence") or 0.0)
        inv.status = m.RunStatus.completed
        inv.finished_at = _now()
        db.commit()
        log.info("investigation %s completed in %s passes", inv.id, inv.iterations)
    except Exception as exc:  # noqa: BLE001 - surface the failure on the row
        db.rollback()
        inv = db.get(m.Investigation, inv_id)
        inv.status = m.RunStatus.failed
        inv.error = f"{type(exc).__name__}: {exc}"[:2000]
        inv.finished_at = _now()
        db.commit()
        log.exception("investigation %s failed", inv_id)
    return inv
