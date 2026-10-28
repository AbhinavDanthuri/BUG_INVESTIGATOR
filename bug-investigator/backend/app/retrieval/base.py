from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol


@dataclass
class Evidence:
    id: str  # stable id the LLM cites, e.g. "E12" (chunk id)
    source_type: str  # code | doc | issue | pr | commit
    ref: str  # path, "#123", or commit sha
    text: str
    start_line: int | None = None
    end_line: int | None = None
    score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def label(self) -> str:
        loc = f":{self.start_line}-{self.end_line}" if self.start_line else ""
        return f"[{self.id}] {self.source_type} {self.ref}{loc}"


class Retriever(Protocol):
    def search(self, query: str, source_types: list[str] | None = None, k: int = 8) -> list[Evidence]: ...

    def fetch_file(self, path: str, line: int | None = None, span: int = 40) -> list[Evidence]:
        """Look a location up directly, rather than searching for it.

        Bug reports usually name the place ("core.py line 1391"). Similarity search is the wrong
        instrument for that: it ranks by resemblance when the address is already known. Without
        this, an agent that knows exactly where to look still has to guess its way there.
        """
        ...
