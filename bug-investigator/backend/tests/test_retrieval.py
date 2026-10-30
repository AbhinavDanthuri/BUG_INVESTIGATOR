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


# ---------- direct lookup ----------
def test_fetch_file_returns_the_chunk_covering_a_named_line(retriever):
    hits = retriever.fetch_file("src/stats.py", line=5)
    assert hits
    assert all(h.ref == "src/stats.py" for h in hits)
    assert all(h.score == 1.0 for h in hits)  # an exact lookup, not a ranked guess
    assert hits[0].start_line <= 5 + 40


def test_fetch_file_matches_a_bare_filename(retriever):
    """Reporters write 'stats.py', not the path from the repository root."""
    assert [h.ref for h in retriever.fetch_file("stats.py", line=3)] == ["src/stats.py"]


def test_fetch_file_without_a_line_returns_nothing(retriever):
    """The top of a file is imports. Returning it wastes evidence slots search would have used -
    the regression that turned a correct verdict into 'unclear'."""
    assert retriever.fetch_file("src/stats.py") == []
    assert retriever.fetch_file("src/stats.py", line=None) == []


def test_fetch_file_returns_nothing_for_a_stale_line(retriever):
    """A line number from an old issue may no longer exist; say nothing rather than something
    irrelevant."""
    assert retriever.fetch_file("src/stats.py", line=99999) == []


def test_fetch_file_returns_nothing_for_an_unknown_path(retriever):
    assert retriever.fetch_file("src/does_not_exist.py", line=1) == []
    assert retriever.fetch_file("", line=1) == []


def test_fetch_file_respects_its_limit(retriever):
    assert len(retriever.fetch_file("src/stats.py", line=5, limit=1)) <= 1


def test_fetch_file_prefers_the_chunk_closest_to_the_line(retriever):
    """With a tight limit, the chunk actually containing the line must win."""
    hits = retriever.fetch_file("src/stats.py", line=1, limit=1)
    assert hits and hits[0].start_line == 1


def test_fetch_file_ids_are_consistent_with_search(retriever):
    """The same chunk must keep one E-label however it was found, or citations break."""
    searched = {h.ref: h.id for h in retriever.search("average", ["code"], k=5)}
    fetched = retriever.fetch_file("src/stats.py", line=1)
    if "src/stats.py" in searched and fetched:
        assert any(h.id == searched["src/stats.py"] for h in fetched)


def test_fetch_file_ignores_non_code_sources(retriever):
    """Commit and issue chunks have no path; a path lookup must not match them."""
    assert all(h.source_type in ("code", "doc") for h in retriever.fetch_file("stats.py", line=1))


# ---------- path matching ----------
@pytest.mark.parametrize(
    "indexed,wanted,expected",
    [
        ("src/click/core.py", "src/click/core.py", True),      # identical
        ("src/click/core.py", "core.py", True),                # bare filename
        ("src/click/core.py", "click/core.py", True),          # partial path
        # pallets/click#3832 writes "click/src/click/core.py": a repo-name prefix the index lacks.
        ("src/click/core.py", "click/src/click/core.py", True),
        ("src/click/core.py", "src/click/types.py", False),    # different file
        ("src/click/core.py", "my_core.py", False),            # suffix of the name, not a component
        ("src/click/core.py", "other/core.py", False),         # same name, different directory
    ],
)
def test_paths_match(indexed, wanted, expected):
    from app.retrieval.hybrid import HybridRetriever

    assert HybridRetriever._paths_match(indexed, wanted) is expected


def test_fetch_file_handles_a_repo_prefixed_path(retriever):
    """The exact shape that silently returned nothing on a real issue."""
    assert [h.ref for h in retriever.fetch_file("sample/src/stats.py", line=3)] == ["src/stats.py"]


# ---------- not retrieving the answer to the question being asked ----------
def _issue_chunk(db, version, number, text, kind="issue"):
    import hashlib

    from app.db import models as m

    c = m.Chunk(
        repo_id=version.repo_id,
        index_version_id=version.id,
        source_type=m.SourceType.issue if kind == "issue" else m.SourceType.pr,
        source_ref=f"#{number}",
        text=text,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
    )
    db.add(c)
    db.commit()
    return c


def test_the_issue_under_investigation_is_not_retrieved(db, indexed_repo):
    """A search for flask#6143 returned "issue #6143" as its top hit.

    Every eval case is a closed issue, and the index holds the repository's issue history, so each
    case sits in its own index. Reading back the issue you were handed is not investigation; in the
    eval it is leakage.
    """
    from app.retrieval.hybrid import BM25OnlyRetriever, HybridRetriever, clear_index_cache

    repo, version = indexed_repo
    _issue_chunk(db, version, 6143, "add_url_rule leaves a route registered after an endpoint conflict")
    clear_index_cache()

    query = "add_url_rule leaves a route registered after an endpoint conflict"
    for cls in (HybridRetriever, BM25OnlyRetriever):
        plain = cls(db, version)
        guarded = cls(db, version, exclude_issue=6143)
        assert any(e.ref == "#6143" for e in plain.search(query, k=12)), f"{cls.__name__}: setup"
        assert not any(e.ref == "#6143" for e in guarded.search(query, k=12))
    clear_index_cache()


def test_a_pull_request_that_answers_the_issue_is_excluded_too(db, indexed_repo):
    """pr #6144 came back alongside issue #6143. "Fixes #6143" hands over the verdict."""
    from app.retrieval.hybrid import HybridRetriever, clear_index_cache

    repo, version = indexed_repo
    _issue_chunk(db, version, 6144, "Fixes #6143 by removing the rule when the endpoint clashes", kind="pr")
    clear_index_cache()

    guarded = HybridRetriever(db, version, exclude_issue=6143)
    assert not any(e.ref == "#6144" for e in guarded.search("endpoint clashes rule removed", k=12))
    clear_index_cache()


def test_a_coincidental_number_is_not_mistaken_for_a_reference(db, indexed_repo):
    """#61431 is not #6143."""
    from app.retrieval.hybrid import HybridRetriever, clear_index_cache

    repo, version = indexed_repo
    _issue_chunk(db, version, 7000, "see #61431 for the endpoint clash discussion")
    clear_index_cache()

    guarded = HybridRetriever(db, version, exclude_issue=6143)
    assert any(e.ref == "#7000" for e in guarded.search("endpoint clash discussion", k=12))
    clear_index_cache()


def test_code_is_never_excluded_however_the_numbers_fall(db, indexed_repo):
    """Only the conversation is off limits. Code mentioning the issue number stays."""
    from app.retrieval.hybrid import HybridRetriever, clear_index_cache

    repo, version = indexed_repo
    clear_index_cache()
    guarded = HybridRetriever(db, version, exclude_issue=6143)
    assert any(e.source_type == "code" for e in guarded.search("average empty list", k=12))
    clear_index_cache()
