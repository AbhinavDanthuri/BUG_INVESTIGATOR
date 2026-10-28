from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

_ISSUE_URL = re.compile(r"^https?://github\.com/([\w.-]+)/([\w.-]+)/issues/(\d+)/?(?:[#?].*)?$")


@dataclass(frozen=True)
class IssueRef:
    owner: str
    repo: str
    number: int


def parse_issue_url(url: str) -> IssueRef:
    m = _ISSUE_URL.match(url.strip())
    if not m:
        raise ValueError(f"not a GitHub issue URL: {url!r}")
    return IssueRef(owner=m.group(1), repo=m.group(2), number=int(m.group(3)))


class GitHubError(RuntimeError):
    pass


class GitHubClient:
    """Thin read-only GitHub REST client. Ingestion helpers (list issues/PRs/commits) arrive in M1."""

    def __init__(self, token: str = "", base_url: str = "https://api.github.com"):
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._http = httpx.Client(base_url=base_url, headers=headers, timeout=30)

    def _get(self, path: str, **params) -> dict | list:
        r = self._http.get(path, params=params or None)
        if r.status_code == 403 and r.headers.get("X-RateLimit-Remaining") == "0":
            raise GitHubError(f"GitHub rate limit hit; resets at epoch {r.headers.get('X-RateLimit-Reset')}")
        if r.status_code == 404:
            raise GitHubError(f"not found: {path}")
        r.raise_for_status()
        return r.json()

    def get_repo(self, owner: str, repo: str) -> dict:
        return self._get(f"/repos/{owner}/{repo}")  # type: ignore[return-value]

    def get_issue(self, ref: IssueRef) -> dict:
        data = self._get(f"/repos/{ref.owner}/{ref.repo}/issues/{ref.number}")
        if "pull_request" in data:  # type: ignore[operator]
            raise GitHubError("that URL is a pull request, not an issue")
        return data  # type: ignore[return-value]

    # ---------- ingestion helpers ----------
    def list_issues(self, owner: str, repo: str, limit: int = 300, state: str = "all") -> list[dict]:
        """Issues and PRs (GitHub returns both from this endpoint), newest first, normalized."""
        out: list[dict] = []
        page = 1
        while len(out) < limit:
            batch = self._get(
                f"/repos/{owner}/{repo}/issues",
                state=state,
                per_page=min(100, limit - len(out)),
                page=page,
                sort="created",
                direction="desc",
            )
            if not batch:
                break
            for raw in batch:  # type: ignore[union-attr]
                out.append(
                    {
                        "number": raw["number"],
                        "title": raw["title"],
                        "body": raw.get("body") or "",
                        "state": raw["state"],
                        "labels": [lbl["name"] for lbl in raw.get("labels", [])],
                        "author": (raw.get("user") or {}).get("login"),
                        "html_url": raw["html_url"],
                        "is_pr": "pull_request" in raw,
                        "created_at": raw.get("created_at"),
                        "closed_at": raw.get("closed_at"),
                    }
                )
            page += 1
            if len(batch) < 100:  # type: ignore[arg-type]
                break
        return out[:limit]

    def pr_files(self, owner: str, repo: str, number: int, limit: int = 100) -> list[str]:
        files = self._get(f"/repos/{owner}/{repo}/pulls/{number}/files", per_page=limit)
        return [f["filename"] for f in files]  # type: ignore[union-attr]

    def get_pull(self, owner: str, repo: str, number: int) -> dict:
        return self._get(f"/repos/{owner}/{repo}/pulls/{number}")  # type: ignore[return-value]

    def issue_state_reason(self, owner: str, repo: str, number: int) -> str | None:
        """'completed' or 'not_planned'. A not_planned close is how maintainers say "won't fix"."""
        data = self._get(f"/repos/{owner}/{repo}/issues/{number}")
        return data.get("state_reason")  # type: ignore[union-attr]

    def issue_timeline_closers(self, owner: str, repo: str, number: int) -> list[int]:
        """PR numbers that closed this issue (used to build eval ground truth)."""
        events = self._get(
            f"/repos/{owner}/{repo}/issues/{number}/timeline", per_page=100
        )
        prs: list[int] = []
        for ev in events:  # type: ignore[union-attr]
            if ev.get("event") in ("cross-referenced", "closed", "connected"):
                src = ev.get("source") or {}
                issue = src.get("issue") or {}
                if "pull_request" in issue:
                    prs.append(issue["number"])
                elif ev.get("commit_id") is None and issue.get("number"):
                    continue
        return sorted(set(prs))
