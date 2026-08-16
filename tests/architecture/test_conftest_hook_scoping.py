"""Guard against conftest collection hooks that silently disable the suite.

``pytest_collection_modifyitems`` is a session-level hook: pytest calls every
loaded implementation once with the *entire* session's item list, regardless of
which directory the defining conftest lives in. A nested conftest that marks
"every item" therefore reaches tests it has no business touching.

``tests/unit/codemap/conftest.py`` did exactly that: when the optional
``codemap`` extra was absent it applied ``pytest.mark.skip`` to every collected
item, so any pytest session that included ``tests/unit/codemap`` reported the
whole run as skipped and still exited 0.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

HOOK_NAME = "pytest_collection_modifyitems"

# Only conftests at a collection root may define the session-wide hook, because
# only there does "every item" legitimately mean "every item in scope".
ALLOWED_HOOK_CONFTESTS = frozenset(
    {
        "conftest.py",
        "tests/conftest.py",
    }
)

EXCLUDED_DIR_PARTS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "site-packages",
        "target",
        "venv",
    }
)


def _iter_conftests() -> list[Path]:
    return [
        path
        for path in REPO_ROOT.rglob("conftest.py")
        if EXCLUDED_DIR_PARTS.isdisjoint(path.relative_to(REPO_ROOT).parts)
    ]


def test_only_collection_root_conftests_define_modifyitems() -> None:
    """Nested conftests must not define the session-wide collection hook."""
    violations: list[str] = []

    for path in _iter_conftests():
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in ALLOWED_HOOK_CONFTESTS:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                and node.name == HOOK_NAME
            ):
                violations.append(f"{rel}:{node.lineno}")

    assert not violations, (
        f"{HOOK_NAME} in a nested conftest receives the whole session's item "
        "list, not just the items under that conftest's directory, so it can "
        "skip or deselect unrelated tests and leave the suite vacuously green. "
        "Use a directory-scoped autouse fixture, a module-level "
        "pytest.importorskip, or collect_ignore instead. Offenders:\n"
        + "\n".join(violations)
    )


@pytest.mark.timeout(120)
def test_codemap_conftest_skip_does_not_leak_to_other_directories(
    tmp_path: Path,
) -> None:
    """A test outside tests/unit/codemap must survive sharing its session."""
    canary = tmp_path / "test_conftest_scoping_canary.py"
    canary.write_text(
        "def test_canary_is_not_skipped_by_a_sibling_directory() -> None:\n"
        "    assert True\n",
        encoding="utf-8",
    )

    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "pytest",
            str(canary),
            str(REPO_ROOT / "tests" / "unit" / "codemap"),
            # Pin the config explicitly: the canary lives outside the repo, so
            # letting pytest infer a rootdir from the arguments could drop the
            # `pythonpath` entries the codemap modules need to import.
            "-c",
            str(REPO_ROOT / "pyproject.toml"),
            # Drop the repo-level addopts (xdist fan-out, marker deselection,
            # coverage) so this stays a cheap, deterministic two-target run.
            "-o",
            "addopts=",
            # The codemap modules import an optional dependency stack. If that
            # import fails outright, keep running the canary rather than
            # aborting the session — a collection error there is a different
            # problem than the scoping regression under test.
            "--continue-on-collection-errors",
            "-p",
            "no:randomly",
            "-p",
            "no:cacheprovider",
            "--import-mode=importlib",
            "-v",
            "--no-header",
            "--tb=no",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    canary_passed = "test_canary_is_not_skipped_by_a_sibling_directory PASSED"
    assert canary_passed in result.stdout, (
        "Collecting tests/unit/codemap in the same session skipped an unrelated "
        "test. A conftest under tests/unit/codemap is applying a session-wide "
        f"skip.\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
