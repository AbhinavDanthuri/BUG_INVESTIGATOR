"""Extracting file locations from issue text."""
import pytest

from app.agents.locations import Location, extract_locations


@pytest.mark.parametrize(
    "text,expected",
    [
        ("crash at src/click/core.py:1391", [Location("src/click/core.py", 1391)]),
        ("see core.py line 1391", [Location("core.py", 1391)]),
        ("core.py, line 1391 is wrong", [Location("core.py", 1391)]),
        ("core.py (line 1391)", [Location("core.py", 1391)]),
        ("traceback core.py:1391:5 column", [Location("core.py", 1391)]),
        ("look in `utils.py`", [Location("utils.py", None)]),
    ],
)
def test_extract_common_shapes(text, expected):
    assert extract_locations(text) == expected


def test_a_line_qualified_path_wins_over_a_bare_one():
    """The line number is the useful part; don't let a later bare mention shadow it."""
    text = "see `core.py` for context, the bug is at core.py:1391"
    assert extract_locations(text) == [Location("core.py", 1391)]


def test_non_source_files_are_ignored():
    assert extract_locations("screenshot.png and output.log and data.csv") == []


def test_empty_and_plain_text():
    assert extract_locations("") == []
    assert extract_locations("it just crashes sometimes") == []


def test_results_are_deduplicated_and_capped():
    text = " ".join(f"file{i}.py:{i}" for i in range(20))
    locs = extract_locations(text, limit=3)
    assert len(locs) == 3
    assert len({loc.path for loc in locs}) == 3


def test_real_issue_text():
    """From pallets/click#3832, the issue that exposed the gap."""
    text = (
        "Terminal escape-sequence injection in error messages. "
        "The formatting at src/click/core.py:1391 and src/click/types.py:1010 "
        "interpolates user input directly."
    )
    assert extract_locations(text) == [
        Location("src/click/core.py", 1391),
        Location("src/click/types.py", 1010),
    ]


# ---------- GitHub permalinks ----------
@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "see https://github.com/pallets/click/blob/8.1.7/src/click/core.py#L1391",
            [Location("src/click/core.py", 1391)],
        ),
        (
            "https://github.com/pallets/click/blob/main/src/click/types.py#L1010-L1020",
            [Location("src/click/types.py", 1010)],
        ),
        ("core.py#L1391", [Location("core.py", 1391)]),
    ],
)
def test_github_permalinks(text, expected):
    """How people actually point at code in an issue."""
    assert extract_locations(text) == expected


def test_a_permalink_beats_a_bare_mention_of_the_same_file():
    text = "`core.py` is involved; precisely https://github.com/a/b/blob/main/src/core.py#L99"
    locs = extract_locations(text)
    assert Location("src/core.py", 99) in locs


def test_mixed_formats_in_one_issue():
    text = (
        "Repro: https://github.com/pallets/click/blob/main/src/click/core.py#L1391\n"
        "Also types.py:1010 and utils.py line 55.\n"
        "Context in `exceptions.py`."
    )
    assert extract_locations(text) == [
        Location("src/click/core.py", 1391),
        Location("types.py", 1010),
        Location("utils.py", 55),
        Location("exceptions.py", None),
    ]
