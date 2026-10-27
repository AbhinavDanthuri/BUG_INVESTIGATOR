from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class InvestigationState(TypedDict, total=False):
    # input
    issue: dict  # {number, title, body, labels, url}
    max_iterations: int

    # loop control
    iteration: int  # completed critic passes
    critic_feedback: str

    # agent outputs
    plan: dict  # {hypotheses: [...], queries: [{q, sources}]}
    evidence: list[dict]  # merged across iterations, keyed by id
    analysis: dict  # {root_cause, confidence, affected_files, reasoning, evidence_ids}
    fix: dict  # {file, diff, explanation}
    tests: list[dict]  # [{name, code, rationale}]
    review: dict  # {verdict, scores, feedback}

    # result
    low_confidence: bool
    report: dict

    # append-only log, persisted to agent_steps
    trace: Annotated[list[dict], operator.add]
