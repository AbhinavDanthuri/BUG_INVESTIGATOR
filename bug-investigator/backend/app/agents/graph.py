from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.agents.nodes import make_nodes
from app.agents.state import InvestigationState
from app.llm import LLM
from app.retrieval.base import Retriever


def route_after_critic(state: InvestigationState) -> str:
    """accept -> finalize; revise with iterations left -> planner; otherwise finalize (low confidence)."""
    if state["review"]["verdict"] == "accept":
        return "finalize"
    if state.get("iteration", 0) < state.get("max_iterations", 3):
        return "planner"
    return "finalize"


def build_graph(llm: LLM, retriever: Retriever, top_k: int = 8):
    nodes = make_nodes(llm, retriever, top_k)
    g = StateGraph(InvestigationState)
    for name, fn in nodes.items():
        g.add_node(name, fn)
    g.add_edge(START, "planner")
    g.add_edge("planner", "investigator")
    g.add_edge("investigator", "fix")
    g.add_edge("fix", "critic")
    g.add_conditional_edges("critic", route_after_critic, {"planner": "planner", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()


def run_investigation(llm: LLM, retriever: Retriever, issue: dict, max_iterations: int = 3, top_k: int = 8) -> dict:
    graph = build_graph(llm, retriever, top_k)
    return graph.invoke({"issue": issue, "max_iterations": max_iterations, "iteration": 0, "trace": []})
