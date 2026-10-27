"""FAISS index files, one per index_version. Vector ids are chunks.id, so no mapping table."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def index_path(data_dir: str, repo_id: int, version_id: int) -> Path:
    return Path(data_dir) / "indexes" / str(repo_id) / f"{version_id}.faiss"


def build_index(vectors: np.ndarray, ids: list[int], dim: int):
    import faiss

    index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
    if len(ids):
        index.add_with_ids(vectors, np.asarray(ids, dtype="int64"))
    return index


def save_index(index, path: Path) -> None:
    import faiss

    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: Path):
    import faiss

    if not Path(path).exists():
        raise FileNotFoundError(f"FAISS index missing: {path}")
    return faiss.read_index(str(path))


def search(index, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Returns (chunk_id, score) best first. FAISS pads with -1 when fewer than k hits exist."""
    if index.ntotal == 0:
        return []
    scores, ids = index.search(query_vec.reshape(1, -1), min(k, index.ntotal))
    return [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=True) if i != -1]
