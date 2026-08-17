"""The repository's Python-version declarations must agree with each other.

This repo is deliberately two-tier: the root distribution requires >=3.11, while
several sub-packages and Rust crates declare >=3.10 and ship 3.10 wheels. That is
fine — but only while every declaration says the same thing. When they drift, the
CI matrix ends up running root-package code on an interpreter it does not support
and the failures look like real defects instead of a configuration mistake.

These tests pin the contract so the drift cannot happen silently again.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_PYPROJECT = REPO_ROOT / "pyproject.toml"
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"
CI_STANDARD = REPO_ROOT / ".github" / "workflows" / "ci-standard.yml"

_REQUIRES_PYTHON_RE = re.compile(
    r"""^\s*requires-python\s*=\s*["'][^"']*?>=\s*(\d+)\.(\d+)""",
    re.MULTILINE,
)
_MYPY_PYTHON_RE = re.compile(
    r"""^\s*python_version\s*=\s*["'](\d+)\.(\d+)""",
    re.MULTILINE,
)
_MATRIX_RE = re.compile(r"""python-version:\s*\[([^\]]*)\]""")


def _floor(pyproject: Path) -> tuple[int, int]:
    match = _REQUIRES_PYTHON_RE.search(pyproject.read_text(encoding="utf-8"))
    assert match is not None, (
        f"{pyproject} must declare requires-python with a >= bound"
    )
    return int(match.group(1)), int(match.group(2))


def _path_is_within(candidate: Path, parent: Path) -> bool:
    try:
        candidate.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _sub_pyprojects() -> list[Path]:
    found: list[Path] = []
    for path in REPO_ROOT.rglob("pyproject.toml"):
        if path == ROOT_PYPROJECT:
            continue
        parts = set(path.parts)
        if parts & {".venv", "node_modules", "target", "build", "dist", ".git"}:
            continue
        if any(part.startswith((".codex", ".claude", "_")) for part in path.parts):
            continue
        found.append(path)
    return found


def test_root_declarations_agree_on_the_floor() -> None:
    """requires-python, the mypy target, and the classifiers must not disagree."""
    text = ROOT_PYPROJECT.read_text(encoding="utf-8")
    root_floor = _floor(ROOT_PYPROJECT)

    mypy_match = _MYPY_PYTHON_RE.search(text)
    assert mypy_match is not None, "root pyproject must pin [tool.mypy] python_version"
    mypy_version = (int(mypy_match.group(1)), int(mypy_match.group(2)))
    assert mypy_version == root_floor, (
        f"mypy python_version {mypy_version} does not match "
        f"requires-python floor {root_floor}"
    )

    classifiers = re.findall(r"Programming Language :: Python :: (\d+)\.(\d+)", text)
    assert classifiers, "root pyproject must declare Python version classifiers"
    lowest = min((int(major), int(minor)) for major, minor in classifiers)
    assert lowest == root_floor, (
        f"lowest Python classifier {lowest} does not match "
        f"requires-python floor {root_floor}"
    )


def test_ci_matrix_is_covered_by_some_declared_floor() -> None:
    """Every interpreter CI tests must be supported by at least one package.

    A lane below *every* declared floor tests nothing the repo supports.
    """
    matrix_match = _MATRIX_RE.search(CI_STANDARD.read_text(encoding="utf-8"))
    assert matrix_match is not None, (
        "ci-standard.yml must declare a python-version matrix"
    )
    lanes = [
        tuple(int(part) for part in raw.strip().strip("\"'").split("."))
        for raw in matrix_match.group(1).split(",")
        if raw.strip()
    ]
    assert lanes, "the CI matrix must not be empty"

    floors = {_floor(ROOT_PYPROJECT)}
    floors.update(_floor(path) for path in _sub_pyprojects())
    lowest_floor = min(floors)

    for lane in lanes:
        assert lane >= lowest_floor, (
            f"CI lane {lane} is below every declared requires-python floor "
            f"(lowest is {lowest_floor}); either raise the matrix or lower a floor"
        )


def test_claude_md_states_the_floor_it_actually_enforces() -> None:
    """CLAUDE.md must not advertise a floor the root distribution rejects."""
    root_floor = _floor(ROOT_PYPROJECT)
    text = CLAUDE_MD.read_text(encoding="utf-8")
    advertised = re.findall(r"Python\s+(\d+)\.(\d+)\+", text)
    assert advertised, "CLAUDE.md must state a Python floor"
    versions = {(int(major), int(minor)) for major, minor in advertised}
    assert root_floor in versions, (
        f"CLAUDE.md advertises {sorted(versions)} but the root distribution "
        f"requires >={root_floor[0]}.{root_floor[1]}"
    )


def test_sub_packages_that_claim_310_are_not_root_package_code() -> None:
    """A >=3.10 sub-package must be a real distribution, not a root-code subtree.

    Root-package code is governed by the root floor. If a subtree declares a
    lower floor it must ship as its own distribution, otherwise the 3.10 CI lane
    will collect root code it cannot run.
    """
    root_floor = _floor(ROOT_PYPROJECT)
    for pyproject in _sub_pyprojects():
        if _floor(pyproject) >= root_floor:
            continue
        text = pyproject.read_text(encoding="utf-8")
        assert "[project]" in text or "[tool.maturin]" in text, (
            f"{pyproject.relative_to(REPO_ROOT)} declares a floor below the root "
            "but is not a distribution in its own right"
        )


def test_conftest_reads_each_package_declared_floor() -> None:
    """The guard resolves floors from the nearest pyproject, not a hardcoded list.

    Every package is checked, and mismatches are reported together with the path
    that failed. Path resolution is the part most likely to behave differently
    across platforms, so a failure here should name the package rather than leave
    it to be inferred from a downstream assertion.
    """
    import conftest  # noqa: PLC0415 - guard under test

    root_code = REPO_ROOT / "src" / "p1am_control_system" / "backend" / "tests"
    assert conftest._declared_python_floor(root_code) == _floor(ROOT_PYPROJECT), (
        "root-package code must inherit the root floor"
    )

    mismatches = []
    for pyproject in _sub_pyprojects():
        resolved = conftest._declared_python_floor(pyproject.parent)
        declared = _floor(pyproject)
        if resolved != declared:
            mismatches.append(
                f"{pyproject.relative_to(REPO_ROOT)}: declared {declared}, "
                f"guard resolved {resolved}"
            )
    assert not mismatches, "guard mis-resolved declared floors:\n  " + "\n  ".join(
        mismatches
    )


def test_root_package_tests_are_skipped_below_the_root_floor() -> None:
    """Root-package tests must not be collected on a sub-floor interpreter.

    This is the regression guard for the failure that motivated the contract.
    ``src/p1am_control_system`` is root-package code, so its tests must not run
    on the 3.10 lane — that is where a bare ``tomllib`` import aborted collection
    and where ``asyncio.wait_for`` semantics differ. The interpreter is passed in
    rather than patched, so this runs on every lane including the required 3.11.
    """
    import conftest  # noqa: PLC0415 - guard under test

    root_code = REPO_ROOT / "src" / "p1am_control_system" / "backend" / "tests"
    root_floor = _floor(ROOT_PYPROJECT)
    below = (root_floor[0], root_floor[1] - 1)

    assert conftest._below_declared_floor(root_code, below), (
        f"root-package tests must be excluded on Python {below[0]}.{below[1]}"
    )
    assert not conftest._below_declared_floor(root_code, root_floor), (
        "root-package tests must still be collected at the declared floor"
    )


def test_each_sub_package_is_collected_at_its_own_declared_floor() -> None:
    """The guard must not over-reach and silence the lower lane entirely.

    Each package is checked at *its own* floor. Checking them all at the global
    minimum would be wrong by construction: a package declaring >=3.10 is
    legitimately excluded on 3.9, so a single lower-floored package elsewhere in
    the tree would make this fail for reasons that are not a defect.
    """
    import conftest  # noqa: PLC0415 - guard under test

    root_floor = _floor(ROOT_PYPROJECT)
    lower = [path for path in _sub_pyprojects() if _floor(path) < root_floor]
    if not lower:
        pytest.skip("no sub-package declares a floor below the root")

    for pyproject in lower:
        own_floor = _floor(pyproject)
        package = pyproject.parent
        assert not conftest._below_declared_floor(package, own_floor), (
            f"{package.relative_to(REPO_ROOT)} declares "
            f">={own_floor[0]}.{own_floor[1]} but the guard would exclude it on "
            "that very interpreter"
        )


def test_nested_distributions_do_not_widen_their_parent_tree() -> None:
    """A 3.10 distribution nested under root-package code must not leak upward.

    ``src/shared/python/sidekick/process_calculators/psa_package`` is a real
    distribution declaring >=3.10 while living inside root-package territory.
    The nested package may claim 3.10, but its parent tree must stay at the root
    floor — otherwise the lower lane would start collecting root code again.
    """
    import conftest  # noqa: PLC0415 - guard under test

    root_floor = _floor(ROOT_PYPROJECT)
    for pyproject in _sub_pyprojects():
        if _floor(pyproject) >= root_floor:
            continue
        parent_tree = pyproject.parent.parent
        if not _path_is_within(parent_tree, REPO_ROOT / "src" / "shared"):
            continue
        assert conftest._declared_python_floor(parent_tree) == root_floor, (
            f"{parent_tree.relative_to(REPO_ROOT)} inherited a lowered floor from "
            f"the nested distribution at {pyproject.relative_to(REPO_ROOT)}"
        )
