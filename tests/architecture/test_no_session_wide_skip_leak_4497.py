"""Guards against a conftest silencing tests it does not own (issue #4497).

``tests/unit/codemap/conftest.py`` used to implement
``pytest_collection_modifyitems`` and mark **every item in the session**
skipped when the optional tree-sitter stack was absent. That hook receives
the whole session's items, not just the ones under its own directory, so
co-collecting the codemap directory with anything else silenced the entire
run:

    pytest tests/shared/python/theme                    -> 107 passed
    pytest tests/shared/python/theme tests/unit/codemap -> 198 skipped, 0 passed

A full-suite run reported 8411 skipped and zero passed while still exiting 0.
Skips are not failures and junit is still written, so every existing
"vacuous run" guard (#3324, #3325, #3567) reported green.

The tests below are deliberately *meta*: they assert that the suite still
executes tests, rather than asserting anything about a particular feature.
A check that silently does nothing is the failure mode being defended
against, so the guard has to observe real pytest outcomes.
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CODEMAP_TESTS = REPO_ROOT / "tests" / "unit" / "codemap"

# Any test module that is skipped wholesale for missing optional deps must do
# so in a way that cannot reach outside its own file. Directories listed here
# are co-collected with a sentinel to prove they do not leak.
OPTIONAL_DEP_TEST_DIRS = ("tests/unit/codemap",)


def _run_pytest(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run pytest in a subprocess with repo addopts neutralised where needed."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *args,
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
            "-n",
            "0",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )


@pytest.mark.parametrize("optional_dir", OPTIONAL_DEP_TEST_DIRS)
def test_optional_dep_skips_do_not_leak_outside_their_directory(
    tmp_path: Path, optional_dir: str
) -> None:
    """Co-collecting an optional-dep suite must not skip unrelated tests.

    This is the direct regression test for #4497. The sentinel lives outside
    the repository test tree entirely, so the only way it can be skipped is a
    hook reaching across the whole session.
    """
    sentinel = tmp_path / "test_sentinel_4497.py"
    sentinel.write_text(
        "def test_sentinel_must_execute():\n    assert True\n",
        encoding="utf-8",
    )

    result = _run_pytest([str(sentinel), str(REPO_ROOT / optional_dir)])
    combined = result.stdout + result.stderr

    assert "1 passed" in combined or " passed" in combined, (
        f"Co-collecting {optional_dir} with an unrelated sentinel test "
        f"executed nothing. A conftest in that tree is skipping items it "
        f"does not own (issue #4497).\n\n{combined[-3000:]}"
    )
    assert "skipped" not in combined.split("=")[-1] or " passed" in combined, (
        f"Sentinel test was skipped by {optional_dir}'s conftest.\n\n{combined[-3000:]}"
    )


def test_no_conftest_skips_items_it_does_not_own() -> None:
    """No conftest may blanket-mark session items without a path filter.

    ``pytest_collection_modifyitems`` is handed every item in the session.
    A hook that iterates those items and adds a marker without inspecting
    each item's path silences unrelated directories.
    """
    offenders: list[str] = []

    for conftest in REPO_ROOT.joinpath("tests").rglob("conftest.py"):
        try:
            tree = ast.parse(conftest.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):  # pragma: no cover - unreadable file
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name != "pytest_collection_modifyitems":
                continue

            source = ast.dump(node)
            adds_marker = "add_marker" in source
            # A compliant hook must consult each item's location before
            # deciding to skip it.
            filters_by_path = any(
                token in source
                for token in ("path", "fspath", "nodeid", "location", "module")
            )
            if adds_marker and not filters_by_path:
                offenders.append(str(conftest.relative_to(REPO_ROOT)))

    assert not offenders, (
        "These conftest files add markers to every collected item without "
        "filtering by path, which silences tests they do not own (#4497): "
        + ", ".join(offenders)
        + ". Prefer a module-level `pytestmark = pytest.mark.skipif(...)` in "
        "the affected test modules, which structurally cannot reach outside "
        "its own file."
    )


def test_codemap_tests_keep_real_assertions_despite_being_skipped() -> None:
    """Being skipped must not become licence to hollow the tests out.

    The codemap modules are skipped in CI (the ``codemap`` extra is never
    installed), so nothing would notice if their bodies decayed into
    assertion-free stubs -- the same "check that silently does nothing"
    failure mode as #4497 itself. Reuses the repo's own gate implementation
    rather than a second copy of the rule.
    """
    spec = importlib.util.spec_from_file_location(
        "_check_test_assertions_4497",
        REPO_ROOT / "scripts" / "check_test_assertions.py",
    )
    assert spec is not None and spec.loader is not None
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)

    modules = sorted(CODEMAP_TESTS.glob("test_*.py"))
    assert modules, "no codemap test modules found -- the suite was deleted"

    hollow = [
        module.relative_to(REPO_ROOT).as_posix()
        for module in modules
        if not gate.has_behavioral_assertion(module.read_text(encoding="utf-8"))
    ]

    assert not hollow, (
        "These codemap test modules no longer contain any behavioral "
        "assertion: " + ", ".join(hollow) + ". They are skipped in CI, so a "
        "stubbed-out body would never be noticed."
    )


def test_full_suite_nightly_enforces_a_floor_on_executed_tests() -> None:
    """The nightly guard must count *passed*, not merely *collected*.

    The existing guard rejects ``total < 500``, but ``total`` includes
    skipped tests. An all-skipped session (8411 collected, 8411 skipped)
    sails past it. The workflow computes ``passed`` already; it has to
    assert on it.
    """
    workflow = REPO_ROOT / ".github" / "workflows" / "full-suite-nightly.yml"
    body = workflow.read_text(encoding="utf-8")

    # Parse to confirm the workflow is still valid YAML after editing.
    assert yaml.safe_load(body), "full-suite-nightly.yml failed to parse"

    assert "passed < " in body, (
        "full-suite-nightly.yml does not enforce a floor on executed "
        "(non-skipped) tests. A session where every test is skipped still "
        "writes junit, exits 0, and clears the `total < 500` collection "
        "floor (issue #4497)."
    )
