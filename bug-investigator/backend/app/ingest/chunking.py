"""Turn raw sources into retrievable chunks.

Code and docs are split by a sliding line window; issues, PRs and commits are one chunk each
(split if long). Every chunk carries its path/ref so the LLM can cite a real location.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

# Files we never index: binaries, generated output, vendored deps, lockfiles.
SKIP_DIRS = {
    ".git", ".github", "node_modules", "venv", ".venv", "__pycache__", "dist", "build",
    "target", "vendor", ".idea", ".vscode", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "site-packages", "migrations", ".next", "coverage", "htmlcov",
}
SKIP_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".pdf", ".zip", ".tar", ".gz", ".bz2",
    ".whl", ".so", ".dll", ".dylib", ".exe", ".bin", ".jar", ".class", ".pyc", ".pyo",
    ".woff", ".woff2", ".ttf", ".eot", ".mp4", ".mp3", ".wav", ".parquet", ".db", ".sqlite",
}
SKIP_NAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "Pipfile.lock",
    "go.sum", "Cargo.lock", "composer.lock",
}
DOC_SUFFIXES = {".md", ".rst", ".txt", ".adoc"}

MAX_FILE_BYTES = 400_000  # skip anything bigger; usually generated or data


@dataclass
class RawChunk:
    source_type: str  # code | doc | issue | pr | commit
    source_ref: str  # path, "#123", or commit sha
    text: str
    start_line: int | None = None
    end_line: int | None = None
    meta: dict = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        key = f"{self.source_type}|{self.source_ref}|{self.start_line}|{self.text}"
        return hashlib.sha256(key.encode("utf-8", "replace")).hexdigest()

    def embed_text(self) -> str:
        """What actually gets embedded: the location prefix helps the model match on paths."""
        loc = f":{self.start_line}-{self.end_line}" if self.start_line else ""
        return f"{self.source_type} {self.source_ref}{loc}\n{self.text}"


def should_index(rel_path: str, size: int) -> bool:
    parts = rel_path.split("/")
    if any(p in SKIP_DIRS for p in parts[:-1]):
        return False
    name = parts[-1]
    if name in SKIP_NAMES or name.startswith("."):
        return False
    if any(name.endswith(s) for s in SKIP_SUFFIXES):
        return False
    return 0 < size <= MAX_FILE_BYTES


def source_type_for(rel_path: str) -> str:
    return "doc" if any(rel_path.endswith(s) for s in DOC_SUFFIXES) else "code"


def chunk_text_file(rel_path: str, text: str, window: int = 60, overlap: int = 10) -> list[RawChunk]:
    """Sliding line window. Overlap keeps a function from being cut in half at every boundary."""
    lines = text.splitlines()
    if not lines:
        return []
    stype = source_type_for(rel_path)
    step = max(1, window - overlap)
    chunks: list[RawChunk] = []
    for start in range(0, len(lines), step):
        block = lines[start : start + window]
        if not any(line.strip() for line in block):
            continue
        chunks.append(
            RawChunk(
                source_type=stype,
                source_ref=rel_path,
                text="\n".join(block),
                start_line=start + 1,
                end_line=start + len(block),
                meta={"ext": rel_path.rsplit(".", 1)[-1] if "." in rel_path else ""},
            )
        )
        if start + window >= len(lines):
            break
    return chunks


def _split_long(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    return [text[i : i + limit] for i in range(0, len(text), limit)]


def chunk_issue(issue: dict, limit: int = 4000) -> list[RawChunk]:
    stype = "pr" if issue.get("is_pr") else "issue"
    head = f"{issue['title']}\n\n{issue.get('body') or ''}".strip()
    labels = issue.get("labels") or []
    meta = {"state": issue.get("state"), "labels": labels}
    if issue.get("changed_files"):
        meta["changed_files"] = issue["changed_files"]
        head += "\n\nFiles changed: " + ", ".join(issue["changed_files"][:40])
    return [
        RawChunk(stype, f"#{issue['number']}", part, meta=meta)
        for part in _split_long(head, limit)
    ]


def chunk_commit(commit: dict, limit: int = 4000) -> list[RawChunk]:
    text = commit["message"]
    if commit.get("files"):
        text += "\n\nFiles: " + ", ".join(commit["files"][:40])
    return [
        RawChunk("commit", commit["sha"], part, meta={"files": commit.get("files", [])[:40]})
        for part in _split_long(text, limit)
    ]
