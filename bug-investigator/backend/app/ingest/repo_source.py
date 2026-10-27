"""Clone a repo and read its files and commit history via git."""
from __future__ import annotations

import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

from app.ingest.chunking import RawChunk, chunk_commit, chunk_text_file, should_index


class GitError(RuntimeError):
    pass


def _git(*args: str, cwd: Path | None = None, timeout: int = 600) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {proc.stderr.strip()[:500]}")
    return proc.stdout


def _is_git_repo(dest: Path) -> bool:
    if not (dest / ".git").exists():
        return False
    try:
        _git("rev-parse", "--git-dir", cwd=dest)
        return True
    except (GitError, subprocess.TimeoutExpired):
        return False


def clone_or_update(clone_url: str, dest: Path, depth: int = 500) -> Path:
    """Shallow-clone (or fetch into) `dest`, serialized per destination.

    A clone killed partway leaves a directory git refuses to clone into again
    ("cannot copy ... File exists"), so anything that is not a working repo is removed first.
    The lock keeps two concurrent index jobs for the same repo from racing on that directory.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with _clone_lock(dest):
        if dest.exists() and not _is_git_repo(dest):
            shutil.rmtree(dest, ignore_errors=True)

        if _is_git_repo(dest):
            _git("fetch", "--depth", str(depth), "origin", cwd=dest)
            target = "origin/HEAD" if _has_origin_head(dest) else "FETCH_HEAD"
            _git("reset", "--hard", target, cwd=dest)
        else:
            _git("clone", "--depth", str(depth), clone_url, str(dest))
    return dest


@contextmanager
def _clone_lock(dest: Path, timeout: float = 900.0, stale_after: float = 1800.0):
    """Lock on a <dest>.lock directory so concurrent index jobs queue instead of corrupting a clone.

    `mkdir` is atomic on every platform, which `fcntl` is not (it does not exist on Windows).
    A job killed mid-clone leaves the directory behind, so a lock older than `stale_after` is
    reclaimed rather than blocking the repo forever.
    """
    lock_dir = dest.with_name(dest.name + ".lock")
    lock_dir.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    acquired = False

    while True:
        try:
            lock_dir.mkdir()
            acquired = True
            break
        except FileExistsError:
            try:
                if time.time() - lock_dir.stat().st_mtime > stale_after:
                    shutil.rmtree(lock_dir, ignore_errors=True)
                    continue
            except OSError:
                pass  # vanished between the check and the stat; just retry
            if time.monotonic() > deadline:
                raise GitError(f"timed out waiting for another index job on {dest}") from None
            time.sleep(0.5)

    try:
        yield
    finally:
        if acquired:
            shutil.rmtree(lock_dir, ignore_errors=True)


def _has_origin_head(repo: Path) -> bool:
    """Whether origin/HEAD is set; some shallow clones have no symbolic default branch ref."""
    try:
        _git("rev-parse", "origin/HEAD", cwd=repo)
        return True
    except GitError:
        return False


def head_sha(repo: Path) -> str:
    return _git("rev-parse", "HEAD", cwd=repo).strip()


def tracked_files(repo: Path) -> list[str]:
    out = _git("ls-files", "-z", cwd=repo)
    return [p for p in out.split("\0") if p]


def iter_file_chunks(repo: Path) -> list[RawChunk]:
    """Chunk every indexable tracked file."""
    chunks: list[RawChunk] = []
    for rel in tracked_files(repo):
        path = repo / rel
        try:
            if not path.is_file() or path.is_symlink():
                continue
            size = path.stat().st_size
            if not should_index(rel, size):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "\0" in text[:4096]:  # binary that slipped past the suffix filter
            continue
        chunks.extend(chunk_text_file(rel, text))
    return chunks


def iter_commit_chunks(repo: Path, limit: int = 300) -> list[RawChunk]:
    """Recent commits: message + changed file list."""
    sep = "\x1e"
    out = _git(
        "log", f"-{limit}", "--name-only", f"--pretty=format:{sep}%H%n%s%n%b%n--FILES--",
        cwd=repo,
    )
    chunks: list[RawChunk] = []
    for record in out.split(sep):
        record = record.strip("\n")
        if not record:
            continue
        head, _, files_blob = record.partition("--FILES--")
        lines = head.strip("\n").split("\n")
        sha, message = lines[0], "\n".join(lines[1:]).strip()
        files = [f for f in files_blob.strip().split("\n") if f]
        if not sha:
            continue
        chunks.extend(chunk_commit({"sha": sha, "message": message or "(no message)", "files": files}))
    return chunks