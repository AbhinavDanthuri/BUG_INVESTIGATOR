"""Path priors on ranked results.

Written after an eval run on pallets/flask: of the twelve evidence chunks the investigator
received, eleven were `pyproject.toml`, `tests/test_basic.py` and `examples/` - four of them the
same test file. One implementation file got through. Both the run that got the verdict right and
the run that got it wrong had that haul, which is the point: neither was reasoning about the code.
"""
import pytest

from app.retrieval.ranking import path_weight, rerank


@pytest.mark.parametrize(
    "path, expected",
    [
        ("src/flask/app.py", 1.0),
        ("src/click/core.py", 1.0),
        ("docs/api.rst", 1.0),          # docs are evidence of intent; they stay at full weight
        ("tests/test_basic.py", 0.35),
        ("tests/conftest.py", 0.35),
        ("examples/javascript/app.py", 0.25),
        ("pyproject.toml", 0.15),
        ("requirements-dev.txt", 0.15),
    ],
)
def test_path_weight(path, expected):
    assert path_weight(path) == expected


def test_supporting_files_are_demoted_not_dropped():
    """A test file is real evidence of intended behaviour. It just must not outrank the code."""
    scored = [(1, 0.9), (2, 0.8)]  # the test file wins on resemblance alone
    paths = {1: "tests/test_basic.py", 2: "src/flask/app.py"}
    out = rerank(scored, paths.get)
    assert [cid for cid, _ in out] == [2, 1]
    assert len(out) == 2  # still present


def test_one_file_cannot_take_the_whole_budget():
    scored = [(i, 1.0 - i / 100) for i in range(6)]
    paths = {0: "tests/test_basic.py", 1: "tests/test_basic.py", 2: "tests/test_basic.py",
             3: "tests/test_basic.py", 4: "src/flask/app.py", 5: "src/flask/json.py"}
    out = rerank(scored, paths.get)
    kept = [paths[cid] for cid, _ in out]
    assert kept.count("tests/test_basic.py") == 2
    assert "src/flask/app.py" in kept and "src/flask/json.py" in kept


def test_the_cap_keeps_a_files_best_chunks():
    scored = [(1, 0.3), (2, 0.9), (3, 0.6)]
    paths = {1: "src/a.py", 2: "src/a.py", 3: "src/a.py"}
    assert [cid for cid, _ in rerank(scored, paths.get)] == [2, 3]


def test_issue_and_commit_refs_are_untouched():
    """source_ref for those is "#123" or a sha, not a path: no rule should fire on them."""
    assert path_weight("#6115") == 1.0
    assert path_weight("71431a714c5d17344bb9fd82377726a561025207") == 1.0


def test_the_flask_haul_leaves_room_for_code():
    """The real inv-30 evidence list, re-ranked: config and duplicate tests must stop crowding out
    the one implementation file that was there all along."""
    real = [
        "pyproject.toml", "tests/test_basic.py", "examples/tutorial/pyproject.toml",
        "tests/test_basic.py", "examples/javascript/pyproject.toml", "tests/test_basic.py",
        "pyproject.toml", "tests/test_cli.py", "tests/test_basic.py", "tests/test_json.py",
        "src/flask/sansio/app.py", "tests/test_blueprints.py",
    ]
    scored = [(i, 1.0 - i / 100) for i in range(len(real))]
    paths = dict(enumerate(real))
    out = rerank(scored, paths.get)
    assert paths[out[0][0]] == "src/flask/sansio/app.py"  # the only real source file now leads
