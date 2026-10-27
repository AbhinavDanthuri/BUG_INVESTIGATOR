"""API tests. GitHub and the LLM are stubbed; the DB and retrieval are real."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api import routes
from app.db import models as m
from app.db.session import get_session
from app.main import app
from tests.test_runner import ScriptedLLM

REPO_JSON = {"default_branch": "main", "clone_url": "https://github.com/acme/sample.git"}
ISSUE_JSON = {
    "number": 12,
    "title": "ZeroDivisionError on empty list",
    "body": "average([]) crashes",
    "state": "open",
    "labels": [{"name": "bug"}],
    "user": {"login": "reporter"},
    "html_url": "https://github.com/acme/sample/issues/12",
}


class StubGitHub:
    def __init__(self, *a, **k):
        pass

    def get_repo(self, owner, name):
        return REPO_JSON

    def get_issue(self, ref):
        return dict(ISSUE_JSON, number=ref.number)


@pytest.fixture
def client(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'api.db'}")
    m.Base.metadata.create_all(engine)
    Local = sessionmaker(bind=engine, expire_on_commit=False)

    monkeypatch.setattr(routes, "GitHubClient", StubGitHub)
    # never touch the network or an LLM in the background jobs
    monkeypatch.setattr(routes, "_index_job", lambda *a, **k: None)
    monkeypatch.setattr(routes, "_investigate_job", lambda *a, **k: None)

    def override():
        with Local() as s:
            yield s

    app.dependency_overrides[get_session] = override
    with TestClient(app) as c:
        c.session_factory = Local
        yield c
    app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


@pytest.mark.parametrize(
    "value,owner,name",
    [
        ("https://github.com/psf/requests", "psf", "requests"),
        ("psf/requests", "psf", "requests"),
        ("https://github.com/psf/requests.git", "psf", "requests"),
        ("https://github.com/psf/requests/", "psf", "requests"),
    ],
)
def test_parse_repo_url(value, owner, name):
    assert routes.parse_repo_url(value) == (owner, name)


def test_parse_repo_url_rejects_junk():
    with pytest.raises(ValueError):
        routes.parse_repo_url("not a repo at all")


def test_register_repo_then_list(client):
    r = client.post("/api/repos", json={"repo_url": "acme/sample"})
    assert r.status_code == 202
    body = r.json()
    assert (body["owner"], body["name"]) == ("acme", "sample")
    assert body["latest_index"] is None

    listed = client.get("/api/repos").json()
    assert [(x["owner"], x["name"]) for x in listed] == [("acme", "sample")]


def test_register_repo_is_idempotent(client):
    first = client.post("/api/repos", json={"repo_url": "acme/sample"}).json()
    second = client.post("/api/repos", json={"repo_url": "acme/sample"}).json()
    assert first["id"] == second["id"]
    assert len(client.get("/api/repos").json()) == 1


def test_register_repo_rejects_bad_url(client):
    assert client.post("/api/repos", json={"repo_url": "¯\\_(ツ)_/¯"}).status_code == 422


def test_reindex_unknown_repo_is_404(client):
    assert client.post("/api/repos/999/reindex").status_code == 404


def test_create_investigation_queues_and_caches_the_issue(client):
    r = client.post(
        "/api/investigations", json={"issue_url": "https://github.com/acme/sample/issues/12"}
    )
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "queued"

    with client.session_factory() as db:
        issue = db.query(m.Issue).one()
        assert issue.number == 12
        assert issue.labels == ["bug"]
        assert issue.author == "reporter"


def test_create_investigation_rejects_a_pull_request_url(client):
    assert (
        client.post("/api/investigations", json={"issue_url": "https://github.com/acme/sample/pull/3"}).status_code
        == 422
    )


def test_investigation_detail_includes_steps_evidence_and_report(client, tmp_path):
    """Run a real investigation through the runner, then read it back over HTTP."""
    import subprocess

    from app import runner
    from app.config import Settings
    from app.ingest.indexer import build_index_version

    # a real repo to index
    src = tmp_path / "repo"
    (src / "src").mkdir(parents=True)
    (src / "src" / "stats.py").write_text("def average(values):\n    return sum(values) / len(values)\n")
    for args in (["init", "-b", "main"], ["config", "user.email", "t@e.com"], ["config", "user.name", "T"],
                 ["add", "."], ["commit", "-m", "init"]):
        subprocess.run(["git", *args], cwd=src, check=True, capture_output=True)

    settings = Settings(
        database_url="sqlite://", embedding_model="hashing", data_dir=str(tmp_path / "data"), github_token=""
    )

    client.post("/api/investigations", json={"issue_url": "https://github.com/acme/sample/issues/12"})
    with client.session_factory() as db:
        repo = db.query(m.Repository).one()
        repo.clone_url = str(src)
        db.commit()
        build_index_version(db, repo, settings, gh=None, issue_limit=0)
        inv = db.query(m.Investigation).one()
        runner.run(db, inv.id, ScriptedLLM(), settings)
        inv_id = inv.id

    detail = client.get(f"/api/investigations/{inv_id}").json()
    assert detail["status"] == "completed"
    assert detail["repo_full_name"] == "acme/sample"
    assert detail["issue"]["number"] == 12
    assert [s["agent"] for s in detail["steps"]] == ["planner", "investigator", "fix", "critic"]
    assert detail["evidence"] and any(e["cited"] for e in detail["evidence"])
    assert detail["report"]["root_cause"]
    assert detail["report"]["fixes"][0]["file_path"] == "src/stats.py"
    assert detail["report"]["tests"][0]["name"] == "test_average_empty"

    md = client.get(f"/api/investigations/{inv_id}/report.md")
    assert md.status_code == 200
    assert md.text.startswith("# Investigation:")


def test_detail_and_report_404_for_unknown_id(client):
    assert client.get("/api/investigations/999").status_code == 404
    assert client.get("/api/investigations/999/report.md").status_code == 404


def test_report_404_when_run_has_no_report_yet(client):
    client.post("/api/investigations", json={"issue_url": "https://github.com/acme/sample/issues/12"})
    with client.session_factory() as db:
        inv_id = db.query(m.Investigation).one().id
    assert client.get(f"/api/investigations/{inv_id}/report.md").status_code == 404
