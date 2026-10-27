"""Retrieval: tokenization, fusion, and hybrid search against a real FAISS index."""
import pytest

from app.retrieval.hybrid import BM25OnlyRetriever, HybridRetriever, tokenize


def test_tokenize_splits_camel_and_snake_case():
    assert set(tokenize("getUserById")) >= {"getuserbyid", "get", "user", "by", "id"}
    assert set(tokenize("max_retry_count")) >= {"max_retry_count", "max", "retry", "count"}


def test_tokenize_keeps_numbers_and_drops_punctuation():
    assert tokenize("err(404)") == ["err", "404"]


@pytest.fixture
def retriever(db, indexed_repo):
    _, version = indexed_repo
    return HybridRetriever(db, version)


def test_search_finds_the_buggy_file_by_identifier(retriever):
    hits = retriever.search("average divides by len", ["code"], k=5)
    assert hits, "expected at least one hit"
    assert "src/stats.py" in {h.ref for h in hits}


def test_search_respects_source_type_filter(retriever):
    assert {h.source_type for h in retriever.search("average", ["doc"], k=5)} <= {"doc"}
    assert {h.source_type for h in retriever.search("stats", ["commit"], k=5)} <= {"commit"}


def test_search_without_filter_can_return_several_types(retriever):
    types = {h.source_type for h in retriever.search("average mean of values", k=10)}
    assert len(types) >= 2


def test_evidence_ids_are_stable_across_calls(retriever):
    first = {h.ref: h.id for h in retriever.search("average", ["code"], k=5)}
    second = {h.ref: h.id for h in retriever.search("average", ["code"], k=5)}
    assert first == second  # same chunk keeps the same E-label within a run


def test_evidence_ids_are_unique_and_sequential(retriever):
    ids = [h.id for h in retriever.search("average median report", k=8)]
    assert len(ids) == len(set(ids))
    assert ids[0] == "E1"


def test_chunk_id_for_round_trips(retriever):
    hit = retriever.search("average", ["code"], k=1)[0]
    assert retriever.chunk_id_for(hit.id) is not None
    assert retriever.chunk_id_for("E999") is None


def test_hits_carry_line_numbers_for_code(retriever):
    hit = next(h for h in retriever.search("average", ["code"], k=5) if h.ref == "src/stats.py")
    assert hit.start_line == 1 and hit.end_line >= 1


def test_results_are_ordered_by_score(retriever):
    scores = [h.score for h in retriever.search("average mean", k=6)]
    assert scores == sorted(scores, reverse=True)


def test_k_caps_result_count(retriever):
    assert len(retriever.search("average", k=2)) <= 2


def test_bm25_baseline_finds_exact_identifier(db, indexed_repo):
    _, version = indexed_repo
    hits = BM25OnlyRetriever(db, version).search("average values", ["code"], k=5)
    assert "src/stats.py" in {h.ref for h in hits}


def test_unrelated_query_does_not_rank_the_bug_file_first(retriever):
    hits = retriever.search("socket connection timeout host port", ["code"], k=3)
    assert hits[0].ref == "src/network.py"
