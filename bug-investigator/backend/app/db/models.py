"""SQLAlchemy models. `backend/sql/schema.sql` is generated from these (scripts/export_schema.py)."""
from __future__ import annotations

import datetime as dt
import enum

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects import mysql
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# BIGINT in MySQL, plain INTEGER in SQLite (SQLite only autoincrements INTEGER PKs; used in tests).
BigPK = BigInteger().with_variant(Integer, "sqlite")
LongText = Text().with_variant(mysql.LONGTEXT(), "mysql")


def _enum(e: type[enum.Enum]) -> Enum:
    return Enum(e, values_callable=lambda x: [m.value for m in x], name=e.__name__.lower())


class Base(DeclarativeBase):
    pass


# ---------- enums ----------
class IndexStatus(str, enum.Enum):
    building = "building"
    ready = "ready"
    failed = "failed"


class SourceType(str, enum.Enum):
    code = "code"
    doc = "doc"
    issue = "issue"
    pr = "pr"
    commit = "commit"


class RunStatus(str, enum.Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class AgentName(str, enum.Enum):
    planner = "planner"
    investigator = "investigator"
    fix = "fix"
    critic = "critic"


class Verdict(str, enum.Enum):
    accept = "accept"
    revise = "revise"


class DefectVerdict(str, enum.Enum):
    """Whether the report describes a real defect. 'no' is a valid, useful answer."""

    yes = "yes"
    no = "no"
    unclear = "unclear"


# ---------- knowledge base ----------
class Repository(Base):
    __tablename__ = "repositories"
    __table_args__ = (UniqueConstraint("owner", "name", name="uq_repo_owner_name"),)

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    owner: Mapped[str] = mapped_column(String(100))
    name: Mapped[str] = mapped_column(String(100))
    default_branch: Mapped[str] = mapped_column(String(100), default="main")
    clone_url: Mapped[str] = mapped_column(String(300))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())

    index_versions: Mapped[list[IndexVersion]] = relationship(back_populates="repo")


class IndexVersion(Base):
    """One searchable snapshot of a repo at a specific commit."""

    __tablename__ = "index_versions"
    __table_args__ = (Index("ix_indexver_repo_sha", "repo_id", "commit_sha"),)

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    repo_id: Mapped[int] = mapped_column(ForeignKey("repositories.id", ondelete="CASCADE"))
    commit_sha: Mapped[str] = mapped_column(String(40))
    embedding_model: Mapped[str] = mapped_column(String(200))
    embedding_dim: Mapped[int] = mapped_column(Integer)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    faiss_path: Mapped[str | None] = mapped_column(String(500))
    status: Mapped[IndexStatus] = mapped_column(_enum(IndexStatus), default=IndexStatus.building)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime)

    repo: Mapped[Repository] = relationship(back_populates="index_versions")


class Chunk(Base):
    """A retrievable unit of text. `id` doubles as the FAISS vector id."""

    __tablename__ = "chunks"
    __table_args__ = (
        Index("ix_chunk_version_type", "index_version_id", "source_type"),
        Index("ix_chunk_repo_ref", "repo_id", "source_ref"),
    )

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    repo_id: Mapped[int] = mapped_column(ForeignKey("repositories.id", ondelete="CASCADE"))
    index_version_id: Mapped[int] = mapped_column(ForeignKey("index_versions.id", ondelete="CASCADE"))
    source_type: Mapped[SourceType] = mapped_column(_enum(SourceType))
    source_ref: Mapped[str] = mapped_column(String(512))  # file path, "#123", or commit sha
    start_line: Mapped[int | None] = mapped_column(Integer)
    end_line: Mapped[int | None] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(LongText)
    content_hash: Mapped[str] = mapped_column(String(64))
    meta: Mapped[dict | None] = mapped_column(JSON)


class Issue(Base):
    __tablename__ = "issues"
    __table_args__ = (UniqueConstraint("repo_id", "number", name="uq_issue_repo_number"),)

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    repo_id: Mapped[int] = mapped_column(ForeignKey("repositories.id", ondelete="CASCADE"))
    number: Mapped[int] = mapped_column(Integer)
    title: Mapped[str] = mapped_column(String(500))
    body: Mapped[str | None] = mapped_column(LongText)
    state: Mapped[str] = mapped_column(String(20))
    labels: Mapped[list | None] = mapped_column(JSON)
    author: Mapped[str | None] = mapped_column(String(100))
    html_url: Mapped[str] = mapped_column(String(300))
    created_at_gh: Mapped[dt.datetime | None] = mapped_column(DateTime)
    closed_at_gh: Mapped[dt.datetime | None] = mapped_column(DateTime)
    fetched_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())


# ---------- investigations ----------
class Investigation(Base):
    __tablename__ = "investigations"
    __table_args__ = (Index("ix_inv_status_created", "status", "created_at"),)

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    repo_id: Mapped[int] = mapped_column(ForeignKey("repositories.id"))
    issue_id: Mapped[int] = mapped_column(ForeignKey("issues.id"))
    index_version_id: Mapped[int | None] = mapped_column(ForeignKey("index_versions.id"))
    status: Mapped[RunStatus] = mapped_column(_enum(RunStatus), default=RunStatus.queued)
    iterations: Mapped[int] = mapped_column(Integer, default=0)
    max_iterations: Mapped[int] = mapped_column(Integer, default=3)
    llm_provider: Mapped[str | None] = mapped_column(String(50))
    llm_model: Mapped[str | None] = mapped_column(String(100))
    prompt_version: Mapped[str | None] = mapped_column(String(20))
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    confidence: Mapped[float | None] = mapped_column(Float)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime)

    steps: Mapped[list[AgentStep]] = relationship(back_populates="investigation", order_by="AgentStep.seq")
    report: Mapped[Report | None] = relationship(back_populates="investigation", uselist=False)


class AgentStep(Base):
    """Append-only log: one row per node execution."""

    __tablename__ = "agent_steps"
    __table_args__ = (Index("ix_step_inv_seq", "investigation_id", "seq"),)

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    investigation_id: Mapped[int] = mapped_column(ForeignKey("investigations.id", ondelete="CASCADE"))
    iteration: Mapped[int] = mapped_column(Integer)
    seq: Mapped[int] = mapped_column(Integer)
    agent: Mapped[AgentName] = mapped_column(_enum(AgentName))
    input: Mapped[dict | None] = mapped_column(JSON)
    output: Mapped[dict | None] = mapped_column(JSON)
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())

    investigation: Mapped[Investigation] = relationship(back_populates="steps")


class Evidence(Base):
    """A chunk retrieved during a run, with its rank/score and whether the final report cites it."""

    __tablename__ = "evidence"
    __table_args__ = (Index("ix_evidence_inv", "investigation_id"),)

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    investigation_id: Mapped[int] = mapped_column(ForeignKey("investigations.id", ondelete="CASCADE"))
    step_id: Mapped[int | None] = mapped_column(ForeignKey("agent_steps.id", ondelete="SET NULL"))
    chunk_id: Mapped[int] = mapped_column(ForeignKey("chunks.id"))
    rank: Mapped[int] = mapped_column(Integer)
    score: Mapped[float] = mapped_column(Float)
    cited_in_report: Mapped[bool] = mapped_column(Boolean, default=False)


class CriticReview(Base):
    __tablename__ = "critic_reviews"
    __table_args__ = (Index("ix_review_inv", "investigation_id", "iteration"),)

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    investigation_id: Mapped[int] = mapped_column(ForeignKey("investigations.id", ondelete="CASCADE"))
    iteration: Mapped[int] = mapped_column(Integer)
    verdict: Mapped[Verdict] = mapped_column(_enum(Verdict))
    scores: Mapped[dict | None] = mapped_column(JSON)
    feedback: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())


# ---------- outputs ----------
class Report(Base):
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    investigation_id: Mapped[int] = mapped_column(
        ForeignKey("investigations.id", ondelete="CASCADE"), unique=True
    )
    is_defect: Mapped[DefectVerdict] = mapped_column(
        _enum(DefectVerdict), default=DefectVerdict.unclear
    )
    verdict_reason: Mapped[str | None] = mapped_column(Text)
    root_cause: Mapped[str] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)
    affected_files: Mapped[list | None] = mapped_column(JSON)
    low_confidence: Mapped[bool] = mapped_column(Boolean, default=False)
    report_md: Mapped[str] = mapped_column(LongText)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())

    investigation: Mapped[Investigation] = relationship(back_populates="report")
    fixes: Mapped[list[FixSuggestion]] = relationship(cascade="all, delete-orphan")
    tests: Mapped[list[TestSuggestion]] = relationship(cascade="all, delete-orphan")


class FixSuggestion(Base):
    __tablename__ = "fix_suggestions"

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    report_id: Mapped[int] = mapped_column(ForeignKey("reports.id", ondelete="CASCADE"))
    file_path: Mapped[str] = mapped_column(String(512))
    diff: Mapped[str] = mapped_column(LongText)
    explanation: Mapped[str | None] = mapped_column(Text)


class TestSuggestion(Base):
    __tablename__ = "test_suggestions"
    __test__ = False  # not a pytest class

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    report_id: Mapped[int] = mapped_column(ForeignKey("reports.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(200))
    code: Mapped[str] = mapped_column(LongText)
    rationale: Mapped[str | None] = mapped_column(Text)


# ---------- evaluation ----------
class EvalCaseKind(str, enum.Enum):
    """Positive cases have a known fix; negative cases were closed without one."""

    fixed = "fixed"
    wont_fix = "wont_fix"


class EvalCase(Base):
    """A closed issue with a known outcome: either a fix PR, or a decision not to fix."""

    __tablename__ = "eval_cases"
    __table_args__ = (UniqueConstraint("repo_id", "issue_number", name="uq_eval_repo_issue"),)

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    repo_id: Mapped[int] = mapped_column(ForeignKey("repositories.id", ondelete="CASCADE"))
    issue_number: Mapped[int] = mapped_column(Integer)
    kind: Mapped[EvalCaseKind] = mapped_column(_enum(EvalCaseKind), default=EvalCaseKind.fixed)
    gold_pr_number: Mapped[int | None] = mapped_column(Integer)
    gold_files: Mapped[list] = mapped_column(JSON)  # empty for wont_fix cases
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())


class EvalResult(Base):
    __tablename__ = "eval_results"

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    eval_case_id: Mapped[int] = mapped_column(ForeignKey("eval_cases.id", ondelete="CASCADE"))
    investigation_id: Mapped[int | None] = mapped_column(ForeignKey("investigations.id", ondelete="SET NULL"))
    system: Mapped[str] = mapped_column(String(30))  # "agent" or "bm25_baseline"
    hit_at_1: Mapped[bool] = mapped_column(Boolean)
    hit_at_3: Mapped[bool] = mapped_column(Boolean)
    file_recall: Mapped[float] = mapped_column(Float)
    # Did it call the defect right? Null for systems with no verdict, such as the BM25 baseline.
    defect_verdict: Mapped[str | None] = mapped_column(String(10))
    verdict_correct: Mapped[bool | None] = mapped_column(Boolean)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())
