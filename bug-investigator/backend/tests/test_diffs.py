"""The fix agent writes `*** Begin Patch` blocks however firmly it is told not to.

This sample is the real output for pallets/click#3832, run through the normalizer.
"""
from app.agents.diffs import normalize_diff

CORE_TEXT = "\n".join(
    [
        "        ctx.args = args",            # 1401
        "        if args and not ctx.allow_extra_args and not ctx.resilient_parsing:",
        "            ctx.fail(",
        "                ngettext(",
        '                    "Got unexpected extra argument ({args})",',
        '                    "Got unexpected extra arguments ({args})",',
        "                    len(args),",
        '                ).format(args=" ".join(map(str, args)))',  # 1408
        "            )",
    ]
)
EVIDENCE = [
    {"id": "E1", "source_type": "code", "ref": "src/click/core.py", "start_line": 1401, "end_line": 1408, "text": CORE_TEXT}
]

APPLY_PATCH = """*** Begin Patch
*** Update File: src/click/core.py
@@
-                ).format(args=" ".join(map(str, args)))
+                ).format(args=" ".join(map(repr, args)))
*** End Patch"""


def test_apply_patch_format_becomes_a_real_unified_diff():
    out, ok = normalize_diff(APPLY_PATCH, "src/click/core.py", EVIDENCE)
    assert ok
    assert out.splitlines()[:3] == ["--- a/src/click/core.py", "+++ b/src/click/core.py", "@@ -1408,1 +1408,1 @@"]
    assert "*** " not in out
    assert out.endswith('+                ).format(args=" ".join(map(repr, args)))\n')


def test_context_lines_are_used_to_find_the_position():
    patch = """*** Begin Patch
*** Update File: src/click/core.py
@@
                     len(args),
-                ).format(args=" ".join(map(str, args)))
+                ).format(args=" ".join(map(repr, args)))
*** End Patch"""
    out, ok = normalize_diff(patch, "src/click/core.py", EVIDENCE)
    assert ok
    assert "@@ -1407,2 +1407,2 @@" in out


def test_a_repo_prefixed_path_still_finds_the_evidence():
    patch = APPLY_PATCH.replace("src/click/core.py", "click/src/click/core.py")
    _, ok = normalize_diff(patch, "click/src/click/core.py", EVIDENCE)
    assert ok


def test_wrong_hunk_counts_are_recomputed():
    diff = "--- a/x.py\n+++ b/x.py\n@@ -10,99 +10,99 @@\n a\n-b\n+c\n"
    out, ok = normalize_diff(diff, "x.py", [])
    assert ok
    assert "@@ -10,2 +10,2 @@" in out


def test_markdown_fences_are_stripped():
    diff = "```diff\n@@ -3,1 +3,1 @@\n-a\n+b\n```"
    out, ok = normalize_diff(diff, "x.py", [])
    assert ok and "```" not in out


def test_a_patch_that_cannot_be_located_is_flagged_not_guessed():
    patch = "*** Begin Patch\n*** Update File: src/click/core.py\n@@\n-no such line\n+x\n*** End Patch"
    out, ok = normalize_diff(patch, "src/click/core.py", EVIDENCE)
    assert not ok
    assert out == patch  # the original comes back untouched


def test_a_pure_addition_with_no_numbers_cannot_be_placed():
    patch = "@@\n+new line\n"
    _, ok = normalize_diff(patch, "x.py", EVIDENCE)
    assert not ok


def test_remove_and_readd_of_identical_lines_is_rejected():
    diff = "@@ -1,1 +1,1 @@\n-same\n+same\n"
    _, ok = normalize_diff(diff, "x.py", [])
    assert not ok


def test_garbage_and_empty_are_invalid():
    assert normalize_diff("", "x.py", [])[1] is False
    assert normalize_diff("-...", "x.py", [])[1] is False


def test_the_fix_node_flags_an_unverifiable_patch():
    """End to end: the report must not present a bad patch as a good one."""
    from app.agents.graph import run_investigation
    from tests.test_graph import ISSUE, FakeLLM, FakeRetriever

    out = run_investigation(FakeLLM(["accept"]), FakeRetriever(), ISSUE)
    assert out["report"]["fix"]["diff_valid"] is False  # FakeLLM's diff is just "-..."
