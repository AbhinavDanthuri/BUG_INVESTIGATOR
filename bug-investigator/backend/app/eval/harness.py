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
from app.eval.metrics import file_recall, hit_at_k, summarize, verdict_is_correct
from app.github.client import GitHubClient, GitHubError, IssueRef
from app.ingest.indexer import ready_version
from app.llm import LLM
from app.retrieval.hybrid import BM25OnlyRetriever

log = logging.getLogger(__name__)

# A "not planned" close is only a usable negative when the maintainers judged the behaviour. These
# labels mean it was closed for some other reason (pallets/click#3892 describes a plausible crash,
# but was closed as "rejected AI" because of where the report came from).
_PROCESS_LABELS = ("rejected", "duplicate", "invalid", "spam", "stale")

_DOC_SUFFIXES = (".md", ".rst", ".txt")


def _is_code_file(path: str) -> bool:
    """Gold files for localization must be code: not tests, and not docs.

    A docs-only fix (pallets/click#3887 was answered by an FAQ entry) is the maintainers deciding
    to document behaviour, not repairing a defect. Scoring it as a missed bug penalises the agent
    for being right.
    """
    low = path.lower()
    if low.startswith(("test", "docs/", "doc/")) or "/test" in low:
        return False
    return not low.endswith(_DOC_SUFFIXES)


def build_eval_cases(
    db: Session,
    repo: m.Repository,
    gh: GitHubClient,
    limit: int = 20,
    max_gold_files: int = 5,
    wont_fix_limit: int | None = None,
) -> list[m.EvalCase]:
    """Build both kinds of case from closed issues.

    Positive (`fixed`): closed by exactly one small PR, whose changed files are the gold answer.
    Negative (`wont_fix`): closed as not planned with no fix PR - the maintainers decided the
    behaviour was intended, out of scope, or otherwise not a defect.

    The negatives matter more than they look. An eval built only from fixed issues cannot see the
    failure where the tool invents a defect and patches working code, because such issues are
    excluded by construction.
    """
    if wont_fix_limit is None:
        wont_fix_limit = max(1, limit // 2)
    created: list[m.EvalCase] = []
    n_fixed = n_wont_fix = 0

    for issue in gh.list_issues(repo.owner, repo.name, limit=(limit + wont_fix_limit) * 10, state="closed"):
        if n_fixed >= limit and n_wont_fix >= wont_fix_limit:
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

        # ---- negative case: closed as not planned, with no fix ----
        if not prs:
            if n_wont_fix >= wont_fix_limit:
                continue
            if any(p in lbl.lower() for lbl in issue.get("labels", []) for p in _PROCESS_LABELS):
                continue
            try:
                if gh.issue_state_reason(repo.owner, repo.name, issue["number"]) != "not_planned":
                    continue  # closed for some other reason; too ambiguous to score
            except GitHubError:
                continue
            case = m.EvalCase(
                repo_id=repo.id,
                issue_number=issue["number"],
                kind=m.EvalCaseKind.wont_fix,
                gold_pr_number=None,
                gold_files=[],
            )
            db.add(case)
            created.append(case)
            n_wont_fix += 1
            continue

        # ---- positive case: exactly one small fix PR ----
        if len(prs) != 1 or n_fixed >= limit:  # several PRs = ambiguous ground truth
            continue
        try:
            files = gh.pr_files(repo.owner, repo.name, prs[0])
        except GitHubError:
            continue
        files = [f for f in files if _is_code_file(f)]
        if not files or len(files) > max_gold_files:
            continue
        case = m.EvalCase(
            repo_id=repo.id,
            issue_number=issue["number"],
            kind=m.EvalCaseKind.fixed,
            gold_pr_number=prs[0],
            gold_files=files,
        )
        db.add(case)
        created.append(case)
        n_fixed += 1

    db.commit()
    log.info("built %d fixed and %d wont_fix cases", n_fixed, n_wont_fix)
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
    retriever = BM25OnlyRetriever(db, version, exclude_issue=issue.number)
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

        kind = case.kind.value

        if "bm25_baseline" in systems:
            # The baseline retrieves files; it has no notion of whether something is a defect, so
            # it is only scored on positive cases and carries no verdict.
            try:
                row = score(run_baseline(db, repo, issue), case.gold_files)
                db.add(m.EvalResult(eval_case_id=case.id, system="bm25_baseline", **row))
                results["bm25_baseline"].append({**row, "kind": kind, "verdict_correct": None})
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
                predicted = inv.report.is_defect.value
                correct = verdict_is_correct(predicted, kind)
                db.add(
                    m.EvalResult(
                        eval_case_id=case.id,
                        investigation_id=inv.id,
                        system="agent",
                        defect_verdict=predicted,
                        verdict_correct=correct,
                        **row,
                    )
                )
                results["agent"].append(
                    {**row, "kind": kind, "verdict_correct": correct, "defect_verdict": predicted}
                )
            else:
                log.warning("agent run failed for issue #%s: %s", case.issue_number, inv.error)
        db.commit()

    return {system: summarize(rows) for system, rows in results.items()}


def format_summary(summary: dict) -> str:
    """`verdict` is the share of cases where the tool correctly judged defect vs not-a-defect.

    Localization columns cover the `fixed` cases only; `loc` says how many those were.
    """
    header = f"{'system':<16}{'n':>4}{'loc':>5}{'hit@1':>8}{'hit@3':>8}{'recall':>8}{'verdict':>9}"
    lines = [header, "-" * len(header)]
    for system, s in summary.items():
        verdict = "    n/a" if s.get("verdict") is None else f"{s['verdict']:>9.2f}"
        lines.append(
            f"{system:<16}{s['n']:>4}{s['localized']:>5}"
            f"{s['hit@1']:>8.2f}{s['hit@3']:>8.2f}{s['recall']:>8.2f}{verdict}"
        )
    return "\n".join(lines)
