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


# ---------- phrasings found by reading real issues ----------
def test_a_hedged_line_reference_is_still_a_location():
    """flask#6172: "(app.py, around line 258)". The strict pattern saw nothing, so the lookup that
    would have answered the issue never fired, and the investigator judged it on test files."""
    locs = extract_locations("the tuple in `Flask.__init_subclass__` (app.py, around line 258) has 12 methods")
    assert (locs[0].path, locs[0].line) == ("app.py", 258)


def test_the_reference_said_backwards_also_counts():
    locs = extract_locations("the bug is on line 42 of src/flask/app.py")
    assert (locs[0].path, locs[0].line) == ("src/flask/app.py", 42)


def test_a_backticked_path_after_the_line_number():
    locs = extract_locations("line 99 in `src/x/y.py` is wrong")
    assert (locs[0].path, locs[0].line) == ("src/x/y.py", 99)


def test_a_path_and_a_line_in_different_sentences_are_not_joined():
    """Loosening the pattern must not weld unrelated facts together."""
    assert extract_locations("I read config.py yesterday. Separately, line 55 was slow.") == []


def test_prose_between_a_path_and_a_line_breaks_the_link():
    assert extract_locations("app.py is the entry point and does many things before line 258") == []
