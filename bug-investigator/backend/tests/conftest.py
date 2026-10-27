import subprocess
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.db import models as m
from app.retrieval.hybrid import clear_index_cache


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        database_url="sqlite://",
        embedding_model="hashing",  # offline, no model download
        data_dir=str(tmp_path / "data"),
        github_token="",
        max_iterations=3,
        retrieval_top_k=5,
    )


@pytest.fixture
def db(tmp_path) -> Session:
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    m.Base.metadata.create_all(engine)
    clear_index_cache()
    with sessionmaker(bind=engine, expire_on_commit=False)() as session:
        yield session
    clear_index_cache()


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True)


@pytest.fixture
def sample_repo(tmp_path) -> Path:
    """A real git repo with a real bug: average() divides by len() without guarding empty input."""
    src = tmp_path / "sample_repo"
    (src / "src").mkdir(parents=True)
    (src / "docs").mkdir()

    (src / "src" / "stats.py").write_text(
        '"""Summary statistics helpers."""\n\n\n'
        "def average(values):\n"
        '    """Arithmetic mean of a list of numbers."""\n'
        "    return sum(values) / len(values)\n\n\n"
        "def median(values):\n"
        "    ordered = sorted(values)\n"
        "    mid = len(ordered) // 2\n"
        "    return ordered[mid]\n"
    )
    (src / "src" / "report.py").write_text(
        "from src.stats import average\n\n\n"
        "def build_report(rows):\n"
        "    return {'mean_score': average([r['score'] for r in rows])}\n"
    )
    (src / "src" / "network.py").write_text(
        "import socket\n\n\n"
        "def connect(host, port, timeout=30):\n"
        "    sock = socket.create_connection((host, port), timeout)\n"
        "    return sock\n"
    )
    (src / "docs" / "usage.md").write_text(
        "# Usage\n\nCall `average(values)` to get the mean of a list of numbers.\n"
    )
    (src / "README.md").write_text("# sample\n\nA tiny statistics library.\n")

    _git("init", "-b", "main", cwd=src)
    _git("config", "user.email", "t@example.com", cwd=src)
    _git("config", "user.name", "Test", cwd=src)
    _git("add", ".", cwd=src)
    _git("commit", "-m", "Add stats helpers and reporting", cwd=src)
    return src


@pytest.fixture
def indexed_repo(db, settings, sample_repo):
    """Register + fully index the sample repo. Returns (repo_row, index_version_row)."""
    from app.ingest.indexer import build_index_version

    repo = m.Repository(owner="acme", name="sample", clone_url=str(sample_repo), default_branch="main")
    db.add(repo)
    db.commit()
    version = build_index_version(db, repo, settings, gh=None, issue_limit=0, commit_limit=50)
    return repo, version
