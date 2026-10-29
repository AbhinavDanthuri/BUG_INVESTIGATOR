"""Render a finished investigation as Markdown."""
from __future__ import annotations

VERDICT_LABEL = {
    "yes": "Looks like a defect",
    "no": "Does not look like a defect",
    "unclear": "Cannot tell from the evidence",
}


def render_markdown(issue: dict, report: dict, repo_full_name: str) -> str:
    conf = report.get("confidence", 0.0)
    verdict = report.get("is_defect", "unclear")
    lines = [
        f"# Investigation: {repo_full_name}#{issue.get('number')} — {issue.get('title', '')}",
        "",
        f"**Verdict:** {VERDICT_LABEL.get(verdict, verdict)}  ·  "
        f"**Confidence:** {conf:.0%}  ·  **Critic passes:** {report.get('iterations', 0)}",
    ]
    if report.get("low_confidence"):
        lines.append("")
        lines.append(
            f"> **The critic never accepted this analysis** after {report.get('iterations', 0)} "
            f"passes, so treat it as a lead rather than an answer. The {conf:.0%} above is the "
            f"investigator's own estimate, which the critic did not agree with."
        )

    if verdict != "yes":
        lines += ["", "## Why no fix is proposed", "", report.get("verdict_reason", "").strip() or
                  "The evidence does not establish that this is a defect."]
        if verdict == "no":
            lines.append("")
            lines.append("> Patching intended behaviour is worse than leaving it alone. "
                         "The mechanism is explained below so a maintainer can decide.")

    heading = "## Root cause" if verdict == "yes" else "## What is happening"
    lines += ["", heading, "", report.get("root_cause", "_none_").strip()]

    if report.get("reasoning"):
        lines += ["", "## Reasoning", "", report["reasoning"].strip()]

    if report.get("affected_files"):
        lines += ["", "## Relevant files", ""]
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
        if fix.get("diff_valid") is False:
            lines += ["", "> The patch above could not be verified as a valid diff. Read it as a sketch of the change, not something to apply."]
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
