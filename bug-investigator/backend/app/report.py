"""Render a finished investigation as Markdown."""
from __future__ import annotations


def render_markdown(issue: dict, report: dict, repo_full_name: str) -> str:
    conf = report.get("confidence", 0.0)
    lines = [
        f"# Investigation: {repo_full_name}#{issue.get('number')} — {issue.get('title', '')}",
        "",
        f"**Confidence:** {conf:.0%}  ·  **Critic passes:** {report.get('iterations', 0)}",
    ]
    if report.get("low_confidence"):
        lines.append("")
        lines.append("> The critic never accepted this analysis. Treat it as a lead, not an answer.")

    lines += ["", "## Root cause", "", report.get("root_cause", "_none_").strip()]

    if report.get("reasoning"):
        lines += ["", "## Reasoning", "", report["reasoning"].strip()]

    if report.get("affected_files"):
        lines += ["", "## Affected files", ""]
        lines += [f"- `{f}`" for f in report["affected_files"]]

    cited = report.get("cited_evidence") or []
    if cited:
        lines += ["", "## Evidence", ""]
        for ev in cited:
            loc = f":{ev['start_line']}-{ev['end_line']}" if ev.get("start_line") else ""
            lines.append(f"**[{ev['id']}]** `{ev['source_type']}` `{ev['ref']}{loc}`")
            lines += ["", "```", ev["text"][:1500].rstrip(), "```", ""]

    fix = report.get("fix") or {}
    if fix.get("diff"):
        lines += ["", "## Suggested fix", "", f"`{fix.get('file', '')}`", "", "```diff", fix["diff"].rstrip(), "```"]
        if fix.get("explanation"):
            lines += ["", fix["explanation"].strip()]

    tests = report.get("tests") or []
    if tests:
        lines += ["", "## Suggested tests", ""]
        for t in tests:
            lines += [f"### {t.get('name', 'test')}", "", "```python", (t.get("code") or "").rstrip(), "```"]
            if t.get("rationale"):
                lines += ["", t["rationale"].strip(), ""]

    return "\n".join(lines).rstrip() + "\n"
