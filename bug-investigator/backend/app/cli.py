"""Command line entry point.

    python -m app.cli index   owner/repo
    python -m app.cli run     https://github.com/owner/repo/issues/123
    python -m app.cli eval    owner/repo --build-cases 15
    python -m app.cli initdb
"""
from __future__ import annotations

import argparse
import logging
import sys

from sqlalchemy import select

from app import runner
from app.config import get_settings
from app.db import models as m
from app.db.session import get_engine, session_scope
from app.eval.harness import build_eval_cases, format_summary, run_eval
from app.github.client import GitHubClient, parse_issue_url
from app.ingest.indexer import build_index_version
from app.llm import get_llm


def _repo(db, gh, owner: str, name: str) -> m.Repository:
    repo = db.scalar(select(m.Repository).where(m.Repository.owner == owner, m.Repository.name == name))
    if repo is None:
        info = gh.get_repo(owner, name)
        repo = m.Repository(
            owner=owner, name=name, default_branch=info.get("default_branch", "main"), clone_url=info["clone_url"]
        )
        db.add(repo)
        db.commit()
    return repo


def cmd_initdb(_args) -> int:
    m.Base.metadata.create_all(get_engine())
    print("schema created")
    return 0


def cmd_index(args) -> int:
    settings = get_settings()
    owner, name = args.repo.split("/", 1)
    with session_scope() as db:
        gh = GitHubClient(settings.github_token)
        repo = _repo(db, gh, owner, name)
        version = build_index_version(
            db,
            repo,
            settings,
            gh,
            issue_limit=args.issue_limit,
            commit_limit=args.commit_limit,
            force=args.force,
        )
        print(f"index version {version.id}: {version.status.value}, {version.chunk_count} chunks @ {version.commit_sha[:8]}")
    return 0


def cmd_run(args) -> int:
    settings = get_settings()
    ref = parse_issue_url(args.issue_url)
    with session_scope() as db:
        gh = GitHubClient(settings.github_token)
        repo = _repo(db, gh, ref.owner, ref.repo)
        raw = gh.get_issue(ref)
        issue = db.scalar(select(m.Issue).where(m.Issue.repo_id == repo.id, m.Issue.number == ref.number))
        if issue is None:
            issue = m.Issue(repo_id=repo.id, number=ref.number, html_url=raw["html_url"], state="", title="")
            db.add(issue)
        issue.title, issue.body, issue.state = raw["title"], raw.get("body"), raw["state"]
        issue.labels = [lbl["name"] for lbl in raw.get("labels", [])]
        db.commit()

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

        runner.run(db, inv.id, get_llm(settings), settings)
        db.refresh(inv)
        if inv.status is m.RunStatus.failed:
            print(f"FAILED: {inv.error}", file=sys.stderr)
            return 1
        print(inv.report.report_md if inv.report else "(no report)")
    return 0


def cmd_eval(args) -> int:
    settings = get_settings()
    owner, name = args.repo.split("/", 1)
    with session_scope() as db:
        gh = GitHubClient(settings.github_token)
        repo = _repo(db, gh, owner, name)
        if args.build_cases:
            created = build_eval_cases(db, repo, gh, limit=args.build_cases)
            print(f"built {len(created)} new eval cases")
        systems = ("bm25_baseline",) if args.baseline_only else ("agent", "bm25_baseline")
        summary = run_eval(db, repo, settings, get_llm(settings), gh, systems=systems, limit=args.limit)
        print()
        print(format_summary(summary))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="bug-investigator")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("initdb").set_defaults(func=cmd_initdb)

    p_index = sub.add_parser("index", help="clone and index a repository")
    p_index.add_argument("repo", help="owner/name")
    p_index.add_argument("--force", action="store_true", help="reindex even if HEAD is unchanged")
    p_index.add_argument(
        "--issue-limit", type=int, default=300, help="past issues/PRs to index (0 to skip; lower = faster)"
    )
    p_index.add_argument(
        "--commit-limit", type=int, default=300, help="recent commits to index (0 to skip)"
    )
    p_index.set_defaults(func=cmd_index)

    p_run = sub.add_parser("run", help="investigate one issue")
    p_run.add_argument("issue_url")
    p_run.set_defaults(func=cmd_run)

    p_eval = sub.add_parser("eval", help="score the agent against the BM25 baseline")
    p_eval.add_argument("repo", help="owner/name")
    p_eval.add_argument("--build-cases", type=int, default=0, help="discover N new eval cases first")
    p_eval.add_argument("--limit", type=int, default=None, help="only score the first N cases")
    p_eval.add_argument("--baseline-only", action="store_true", help="skip the LLM; score BM25 only")
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())