"""Graph nodes. Built via `make_nodes(llm, retriever)` so tests can inject fakes."""
from __future__ import annotations

import json
import time
from collections.abc import Callable

from app.agents import prompts
from app.agents.locations import extract_locations
from app.agents.state import InvestigationState
from app.llm import LLM
from app.retrieval.base import Retriever

MAX_EVIDENCE = 12  # cap what we send to the LLM (keeps prompts small for free-tier token limits)
MAX_LOOKUP_EVIDENCE = 6  # half the budget; search keeps the rest
# Raised from 4 once lookups became precise (a line is required, two chunks each). While they
# could return the top of a file, a smaller cap was the only thing keeping them from crowding
# out search entirely.


def _trace(agent: str, state: InvestigationState, started: float, output: dict) -> dict:
    return {
        "agent": agent,
        "iteration": state.get("iteration", 0),
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "output": output,
    }


def make_nodes(llm: LLM, retriever: Retriever, top_k: int = 8) -> dict[str, Callable]:
    def planner(state: InvestigationState) -> dict:
        t0 = time.perf_counter()
        previous = [q["q"] for q in state.get("plan", {}).get("queries", [])]
        user = prompts.issue_block(state["issue"])
        if state.get("critic_feedback"):
            user += f"\n\n<feedback>{state['critic_feedback']}</feedback>"
            user += f"\nPrevious queries (do not repeat): {json.dumps(previous)}"
        plan = llm.invoke_json(prompts.PLANNER, user)
        plan.setdefault("hypotheses", [])
        plan.setdefault("queries", [])
        plan.setdefault("lookups", [])

        # The issue text usually names the place. Look those up whether or not the model asked.
        issue = state["issue"]
        named = extract_locations(f"{issue.get('title', '')}\n{issue.get('body', '')}")
        existing = {
            (item.get("path"), item.get("line"))
            for item in plan["lookups"]
            if isinstance(item, dict)
        }
        for loc in named:
            if (loc.path, loc.line) not in existing:
                plan["lookups"].append({"path": loc.path, "line": loc.line, "from": "issue"})

        return {"plan": plan, "trace": [_trace("planner", state, t0, plan)]}

    def investigator(state: InvestigationState) -> dict:
        t0 = time.perf_counter()
        merged = {e["id"]: e for e in state.get("evidence", [])}

        # Direct lookups first: a named location beats anything similarity search would rank.
        # They are capped, though - lookups that return little of value must not crowd out search.
        looked_up = 0
        for loc in state["plan"].get("lookups", []):
            if looked_up >= MAX_LOOKUP_EVIDENCE:
                break
            if not isinstance(loc, dict) or not loc.get("path"):
                continue
            try:
                hits = retriever.fetch_file(str(loc["path"]), loc.get("line"))
            except Exception:  # noqa: BLE001 - a bad path must not sink the run
                continue
            for ev in hits:
                if looked_up >= MAX_LOOKUP_EVIDENCE:
                    break
                if ev.id not in merged:
                    merged[ev.id] = ev.to_dict()
                    looked_up += 1

        for q in state["plan"]["queries"]:
            for ev in retriever.search(q["q"], q.get("sources"), top_k):
                merged.setdefault(ev.id, ev.to_dict())
        evidence = list(merged.values())[:MAX_EVIDENCE]

        user = prompts.issue_block(state["issue"]) + "\n\n" + prompts.evidence_block(evidence)
        analysis = llm.invoke_json(prompts.INVESTIGATOR, user)

        # Deterministic guardrail: drop citations to evidence we never retrieved.
        valid = {e["id"] for e in evidence}
        analysis["evidence_ids"] = [i for i in analysis.get("evidence_ids", []) if i in valid]
        analysis.setdefault("affected_files", [])
        analysis.setdefault("confidence", 0.0)
        analysis.setdefault("verdict_reason", "")
        # An unrecognised verdict must not be read as "yes": defaulting to a defect is the bias
        # this whole change exists to remove.
        if analysis.get("is_defect") not in ("yes", "no", "unclear"):
            analysis["is_defect"] = "unclear"

        out = {**analysis, "evidence_retrieved": len(evidence), "looked_up": looked_up}
        return {"evidence": evidence, "analysis": analysis, "trace": [_trace("investigator", state, t0, out)]}

    def fix(state: InvestigationState) -> dict:
        """Only reached when the investigator called it a defect; see route_after_investigator."""
        t0 = time.perf_counter()
        user = (
            prompts.issue_block(state["issue"])
            + "\n\nRoot cause analysis:\n"
            + json.dumps(state["analysis"], indent=2)
            + "\n\n"
            + prompts.evidence_block(state["evidence"])
        )
        result = llm.invoke_json(prompts.FIX, user)
        fix_ = result.get("fix") or {}
        tests = result.get("tests") or []
        return {"fix": fix_, "tests": tests, "trace": [_trace("fix", state, t0, result)]}

    def critic(state: InvestigationState) -> dict:
        t0 = time.perf_counter()
        # The fix agent is skipped for non-defects, so these may be absent.
        fix_ = state.get("fix") or {}
        tests = state.get("tests") or []
        fix_block = (
            json.dumps(fix_, indent=2)
            if fix_
            else "(none proposed - the investigator did not judge this a defect)"
        )
        user = (
            prompts.issue_block(state["issue"])
            + "\n\nAnalysis:\n" + json.dumps(state["analysis"], indent=2)
            + "\n\nFix:\n" + fix_block
            + "\n\nTests:\n" + json.dumps(tests, indent=2)
            + "\n\n" + prompts.evidence_block(state["evidence"], max_chars=600)
        )
        review = llm.invoke_json(prompts.CRITIC, user)
        review.setdefault("scores", {})
        review.setdefault("feedback", "")

        # Deterministic guardrail: an answer with no valid citation can never be accepted.
        if not state["analysis"].get("evidence_ids"):
            review["verdict"] = "revise"
            review["feedback"] = (
                "No claim is backed by retrieved evidence. Search for the code that implements "
                "the failing behaviour and cite it. " + review["feedback"]
            ).strip()

        # Deterministic guardrail: never ship a patch for something judged not to be a defect.
        # The graph should have skipped the fix agent, so this means the state is inconsistent.
        if state["analysis"].get("is_defect") != "yes" and (state.get("fix") or {}).get("diff"):
            review["verdict"] = "revise"
            review["feedback"] = (
                "A patch was produced for behaviour not judged to be a defect. Either justify the "
                "defect verdict from the evidence, or drop the fix. " + review["feedback"]
            ).strip()
        if review.get("verdict") not in ("accept", "revise"):
            review["verdict"] = "revise"

        return {
            "review": review,
            "critic_feedback": review["feedback"],
            "iteration": state.get("iteration", 0) + 1,
            "trace": [_trace("critic", state, t0, review)],
        }

    def finalize(state: InvestigationState) -> dict:
        accepted = state["review"]["verdict"] == "accept"
        a = state["analysis"]
        report = {
            "is_defect": a.get("is_defect", "unclear"),
            "verdict_reason": a.get("verdict_reason", ""),
            "root_cause": a.get("root_cause", ""),
            "reasoning": a.get("reasoning", ""),
            "confidence": float(a.get("confidence", 0.0)),
            "affected_files": a.get("affected_files", []),
            "cited_evidence": [e for e in state["evidence"] if e["id"] in a.get("evidence_ids", [])],
            "fix": state.get("fix", {}),
            "tests": state.get("tests", []),
            "iterations": state.get("iteration", 0),
            "low_confidence": not accepted,
        }
        return {"report": report, "low_confidence": not accepted}

    return {"planner": planner, "investigator": investigator, "fix": fix, "critic": critic, "finalize": finalize}
