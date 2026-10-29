"""Make the fix agent's patch a real unified diff, or say plainly that it is not.

The prompt asks for a unified diff with an example. The model (gpt-oss on Groq) answers in the
`*** Begin Patch` format anyway, with bare `@@` markers and no line numbers. Asking again does not
change that, so this is handled in code: the patch is converted by finding where its context lines
sit in the evidence the model was shown, which gives real line numbers. If that cannot be done the
diff is flagged rather than passed off as valid.
"""
from __future__ import annotations

import re

_HUNK = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def _strip_fences(text: str) -> str:
    lines = text.strip("\n").split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)


def _paths_match(a: str, b: str) -> bool:
    pa, pb = a.strip("/").split("/"), b.strip("/").split("/")
    n = min(len(pa), len(pb))
    return pa[-n:] == pb[-n:]


def _body_ok(lines: list[str]) -> bool:
    changes = [ln for ln in lines if ln[:1] in ("+", "-")]
    if not changes:
        return False
    removed = [ln[1:] for ln in lines if ln.startswith("-")]
    added = [ln[1:] for ln in lines if ln.startswith("+")]
    return removed != added  # a hunk that removes and re-adds identical lines changes nothing


def _render(path: str, hunks: list[tuple[int, list[str]]]) -> str:
    out = [f"--- a/{path}", f"+++ b/{path}"]
    for start, body in hunks:
        old = sum(1 for ln in body if ln[:1] in (" ", "-"))
        new = sum(1 for ln in body if ln[:1] in (" ", "+"))
        out.append(f"@@ -{start},{old} +{start},{new} @@")
        out.extend(body)
    return "\n".join(out) + "\n"


def _locate(old_lines: list[str], path: str, evidence: list[dict]) -> int | None:
    """Line number in `path` where `old_lines` begin, found in the evidence the model saw."""
    if not old_lines:
        return None
    for strip in (str.rstrip, str.strip):  # exact first, then ignoring indentation
        needle = [strip(x) for x in old_lines]
        for ev in evidence:
            if ev.get("source_type") != "code" or not ev.get("start_line"):
                continue
            if not _paths_match(str(ev.get("ref", "")), path):
                continue
            hay = [strip(x) for x in str(ev.get("text", "")).split("\n")]
            for i in range(len(hay) - len(needle) + 1):
                if hay[i : i + len(needle)] == needle:
                    return int(ev["start_line"]) + i
    return None


def normalize_diff(diff: str, file: str, evidence: list[dict]) -> tuple[str, bool]:
    """Return (diff, is_valid). On failure the original text comes back with is_valid False."""
    if not diff or not diff.strip():
        return diff, False
    text = _strip_fences(diff)
    lines = text.split("\n")
    path = file

    for ln in lines:  # apply_patch names the file itself
        m = re.match(r"\*\*\* (?:Update|Add) File:\s*(.+)", ln)
        if m:
            path = m.group(1).strip()

    # Sections starting at an @@ marker. Anything before the first marker is header noise.
    hunks_raw: list[tuple[str, list[str]]] = []
    for ln in lines:
        if ln.startswith("@@"):
            hunks_raw.append((ln, []))
        elif hunks_raw and not ln.startswith(("*** ", "--- ", "+++ ")):
            hunks_raw[-1][1].append(ln)
    if not hunks_raw or not path:
        return diff, False

    hunks: list[tuple[int, list[str]]] = []
    for header, body in hunks_raw:
        while body and body[-1] == "":
            body.pop()
        body = [ln if ln[:1] in (" ", "+", "-", "\\") else " " + ln for ln in body]
        if not _body_ok(body):
            return diff, False
        m = _HUNK.match(header)
        if m:
            start = int(m.group(1))
        else:  # bare "@@": work out where it belongs from the code the model was shown
            old = [ln[1:] for ln in body if ln[:1] in (" ", "-")]
            found = _locate(old, path, evidence)
            if found is None:
                return diff, False
            start = found
        hunks.append((start, body))
    return _render(path, hunks), True
