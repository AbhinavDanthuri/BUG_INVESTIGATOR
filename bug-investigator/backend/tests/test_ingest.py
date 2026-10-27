"""Ingestion: chunking rules, git walking, and index building against a real repo."""
from app.db import models as m
from app.ingest import repo_source
from app.ingest.chunking import chunk_commit, chunk_issue, chunk_text_file, should_index, source_type_for


def test_should_index_filters_noise():
    assert should_index("src/app.py", 100)
    assert not should_index("node_modules/left-pad/index.js", 100)
    assert not should_index("assets/logo.png", 100)
    assert not should_index("package-lock.json", 100)
    assert not should_index(".env", 100)
    assert not should_index("src/huge.py", 10_000_000)
    assert not should_index("src/empty.py", 0)


def test_source_type_for():
    assert source_type_for("docs/guide.md") == "doc"
    assert source_type_for("README.rst") == "doc"
    assert source_type_for("src/app.py") == "code"


def test_chunk_text_file_windows_overlap_and_tracks_lines():
    text = "\n".join(f"line{i}" for i in range(1, 141))
    chunks = chunk_text_file("src/big.py", text, window=60, overlap=10)
    assert len(chunks) == 3
    assert (chunks[0].start_line, chunks[0].end_line) == (1, 60)
    assert (chunks[1].start_line, chunks[1].end_line) == (51, 110)  # 10-line overlap
    assert chunks[-1].end_line == 140
    assert all(c.source_type == "code" and c.source_ref == "src/big.py" for c in chunks)


def test_chunk_text_file_skips_blank_blocks_and_empty_files():
    assert chunk_text_file("a.py", "") == []
    assert chunk_text_file("a.py", "\n\n\n\n") == []


def test_embed_text_includes_location():
    chunk = chunk_text_file("src/stats.py", "def average(v):\n    pass")[0]
    assert "src/stats.py:1-2" in chunk.embed_text()


def test_content_hash_is_stable_and_distinct():
    a = chunk_text_file("a.py", "x = 1")[0]
    b = chunk_text_file("a.py", "x = 1")[0]
    c = chunk_text_file("a.py", "x = 2")[0]
    assert a.content_hash == b.content_hash != c.content_hash


def test_chunk_issue_and_commit():
    ic = chunk_issue({"number": 42, "title": "Crash", "body": "boom", "labels": ["bug"], "state": "open"})[0]
    assert ic.source_type == "issue" and ic.source_ref == "#42" and "Crash" in ic.text

    pc = chunk_issue({"number": 9, "title": "Fix", "body": "", "is_pr": True, "changed_files": ["src/a.py"]})[0]
    assert pc.source_type == "pr" and "src/a.py" in pc.text

    cc = chunk_commit({"sha": "abc123", "message": "Fix the thing", "files": ["src/a.py"]})[0]
    assert cc.source_type == "commit" and cc.source_ref == "abc123" and "src/a.py" in cc.text


def test_long_issue_is_split():
    parts = chunk_issue({"number": 1, "title": "t", "body": "x" * 9000}, limit=4000)
    assert len(parts) == 3


# ---------- against the real git fixture ----------
def test_tracked_files_and_file_chunks(sample_repo):
    files = repo_source.tracked_files(sample_repo)
    assert "src/stats.py" in files and "README.md" in files

    chunks = repo_source.iter_file_chunks(sample_repo)
    refs = {c.source_ref for c in chunks}
    assert {"src/stats.py", "src/report.py", "docs/usage.md", "README.md"} <= refs
    assert {c.source_type for c in chunks} == {"code", "doc"}


def test_commit_chunks_carry_message_and_files(sample_repo):
    commits = repo_source.iter_commit_chunks(sample_repo, limit=10)
    assert len(commits) == 1
    assert "Add stats helpers" in commits[0].text
    assert "src/stats.py" in commits[0].text
    assert len(commits[0].source_ref) == 40  # full sha


def test_build_index_version_persists_chunks_and_faiss(db, settings, indexed_repo):
    repo, version = indexed_repo
    assert version.status is m.IndexStatus.ready
    assert version.chunk_count > 0
    assert version.embedding_dim == 256  # hashing embedder

    from pathlib import Path

    assert Path(version.faiss_path).exists()

    chunks = db.query(m.Chunk).filter_by(index_version_id=version.id).all()
    assert len(chunks) == version.chunk_count
    assert {c.source_type for c in chunks} >= {m.SourceType.code, m.SourceType.doc, m.SourceType.commit}


def test_reindexing_same_head_reuses_version(db, settings, indexed_repo):
    from app.ingest.indexer import build_index_version

    repo, version = indexed_repo
    again = build_index_version(db, repo, settings, gh=None, issue_limit=0)
    assert again.id == version.id  # HEAD unchanged -> no duplicate work


def test_force_reindex_creates_new_version(db, settings, indexed_repo):
    from app.ingest.indexer import build_index_version

    repo, version = indexed_repo
    forced = build_index_version(db, repo, settings, gh=None, issue_limit=0, force=True)
    assert forced.id != version.id and forced.status is m.IndexStatus.ready


# ---------- clone robustness ----------
def test_clone_recovers_from_a_half_written_directory(tmp_path, sample_repo):
    """A clone killed partway leaves a dir git refuses to reuse; we remove and re-clone."""
    dest = tmp_path / "checkout"
    (dest / ".git").mkdir(parents=True)
    (dest / ".git" / "description").write_text("leftover from an interrupted clone")

    repo_source.clone_or_update(str(sample_repo), dest)

    assert (dest / "src" / "stats.py").exists()
    assert len(repo_source.head_sha(dest)) == 40


def test_clone_into_existing_non_repo_directory(tmp_path, sample_repo):
    dest = tmp_path / "checkout"
    dest.mkdir()
    (dest / "stray.txt").write_text("not a repo")
    repo_source.clone_or_update(str(sample_repo), dest)
    assert (dest / "src" / "stats.py").exists()


def test_second_clone_fetches_instead_of_failing(tmp_path, sample_repo):
    dest = tmp_path / "checkout"
    repo_source.clone_or_update(str(sample_repo), dest)
    first = repo_source.head_sha(dest)
    repo_source.clone_or_update(str(sample_repo), dest)  # must not raise
    assert repo_source.head_sha(dest) == first


def test_concurrent_clones_do_not_corrupt_the_checkout(tmp_path, sample_repo):
    """Two index jobs racing on one repo is the case that broke in the smoke test."""
    import threading

    dest = tmp_path / "checkout"
    errors: list[Exception] = []

    def worker():
        try:
            repo_source.clone_or_update(str(sample_repo), dest)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert (dest / "src" / "stats.py").exists()
    assert len(repo_source.head_sha(dest)) == 40


def test_index_still_builds_when_issue_history_fails(db, settings, sample_repo):
    """Issue history is a bonus. A GitHub failure must not throw away the code index."""
    from app.ingest.indexer import build_index_version

    class BrokenGH:
        def list_issues(self, *a, **k):
            raise RuntimeError("GitHub is having a day")

    repo = m.Repository(owner="acme", name="sample2", clone_url=str(sample_repo), default_branch="main")
    db.add(repo)
    db.commit()

    version = build_index_version(db, repo, settings, gh=BrokenGH(), issue_limit=50)

    assert version.status is m.IndexStatus.ready
    assert version.chunk_count > 0
    types = {c.source_type for c in db.query(m.Chunk).filter_by(index_version_id=version.id)}
    assert m.SourceType.code in types
    assert m.SourceType.issue not in types  # history was skipped, cleanly
