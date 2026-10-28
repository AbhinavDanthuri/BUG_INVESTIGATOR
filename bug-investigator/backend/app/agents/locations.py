"""Pull file locations out of issue text.

A bug report usually says where it hurts: "core.py line 1391", "src/click/types.py:1010",
"in `utils.py`". Similarity search is the wrong tool for an address that is already written down,
so these are looked up directly instead of searched for.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# A GitHub permalink is how people usually point at code in an issue:
#   https://github.com/owner/repo/blob/<sha>/src/click/core.py#L1391
#   https://github.com/owner/repo/blob/main/src/click/core.py#L1391-L1400
# Missing these was why a run looked up three files and fetched only their import blocks.
_GITHUB_BLOB = re.compile(
    r"github\.com/[\w.\-]+/[\w.\-]+/blob/[^/\s]+/([\w./\-]+\.[A-Za-z]{1,6})#L(\d+)",
    re.IGNORECASE,
)
# path#L123 on its own, the same anchor without the full URL
_PATH_HASH_LINE = re.compile(r"\b([\w./\-]+\.[A-Za-z]{1,6})#L(\d+)\b")
# path:line  e.g. src/click/core.py:1391  or  core.py:1391:5
_PATH_COLON_LINE = re.compile(r"\b([\w./\-]+\.[A-Za-z]{1,6}):(\d+)(?::\d+)?\b")
# path "line" N  e.g. core.py line 1391 / core.py, line 1391 / core.py (line 1391)
_PATH_WORD_LINE = re.compile(r"\b([\w./\-]+\.[A-Za-z]{1,6})\b[\s,(]*\bline\s+(\d+)\b", re.IGNORECASE)
# a bare filename, usually in backticks
_BARE_PATH = re.compile(r"`([\w./\-]+\.[A-Za-z]{1,6})`")

# Extensions worth looking up. Anything else is usually a log, an image or a sample data file.
_SOURCE_SUFFIXES = {
    "py", "pyi", "js", "jsx", "ts", "tsx", "java", "go", "rs", "rb", "c", "h", "cc", "cpp", "hpp",
    "cs", "kt", "swift", "php", "scala", "md", "rst", "txt", "toml", "cfg", "ini", "yaml", "yml",
}


@dataclass(frozen=True)
class Location:
    path: str
    line: int | None = None


def _is_source(path: str) -> bool:
    suffix = path.rsplit(".", 1)[-1].lower()
    return suffix in _SOURCE_SUFFIXES


def extract_locations(text: str, limit: int = 6) -> list[Location]:
    """Locations named in `text`, most specific first, de-duplicated by path.

    A path with a line number wins over the same path without one: the line is the useful part.
    """
    if not text:
        return []

    with_lines: list[Location] = []

    # Permalinks first, then remove them from the text: the simpler patterns would otherwise
    # match inside the URL and yield a "path" of github.com/owner/repo/blob/sha/src/file.py.
    for path, line in _GITHUB_BLOB.findall(text):
        if _is_source(path):
            with_lines.append(Location(path.lstrip("./"), int(line)))
    remainder = _GITHUB_BLOB.sub(" ", text)

    for pattern in (_PATH_HASH_LINE, _PATH_COLON_LINE, _PATH_WORD_LINE):
        for path, line in pattern.findall(remainder):
            if _is_source(path):
                with_lines.append(Location(path.lstrip("./"), int(line)))

    bare = [
        Location(path.lstrip("./"))
        for path in _BARE_PATH.findall(remainder)
        if _is_source(path)
    ]

    seen: set[str] = set()
    out: list[Location] = []
    for loc in with_lines + bare:  # line-qualified first
        if loc.path in seen:
            continue
        seen.add(loc.path)
        out.append(loc)
        if len(out) >= limit:
            break
    return out
