from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, ConfigDict


# ---------- repos ----------
class RepoCreate(BaseModel):
    repo_url: str  # https://github.com/owner/name or "owner/name"


class IndexVersionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    commit_sha: str
    status: str
    chunk_count: int
    embedding_model: str
    error: str | None = None
    created_at: dt.datetime
    finished_at: dt.datetime | None = None


class RepoOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    owner: str
    name: str
    default_branch: str
    latest_index: IndexVersionOut | None = None


# ---------- investigations ----------
class InvestigationCreate(BaseModel):
    issue_url: str


class IssueOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    number: int
    title: str
    state: str
    html_url: str
    labels: list[str] | None = None


class InvestigationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    repo_id: int
    issue_id: int
    status: str
    iterations: int
    max_iterations: int
    confidence: float | None = None
    llm_model: str | None = None
    error: str | None = None
    created_at: dt.datetime
    started_at: dt.datetime | None = None
    finished_at: dt.datetime | None = None


class AgentStepOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    seq: int
    iteration: int
    agent: str
    output: dict | None = None
    latency_ms: int | None = None


class EvidenceOut(BaseModel):
    source_type: str
    ref: str
    start_line: int | None = None
    end_line: int | None = None
    score: float
    cited: bool
    text: str


class FixOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    file_path: str
    diff: str
    explanation: str | None = None


class TestOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    code: str
    rationale: str | None = None


class ReportOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    root_cause: str
    confidence: float
    affected_files: list[str] | None = None
    low_confidence: bool
    report_md: str
    fixes: list[FixOut] = []
    tests: list[TestOut] = []


class InvestigationDetail(InvestigationOut):
    issue: IssueOut | None = None
    repo_full_name: str | None = None
    steps: list[AgentStepOut] = []
    evidence: list[EvidenceOut] = []
    report: ReportOut | None = None
