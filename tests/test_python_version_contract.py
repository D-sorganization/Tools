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
import subprocess
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


def _load_root_conftest():
    import importlib.util
    import sys

    root_conftest_path = REPO_ROOT / "conftest.py"
    spec = importlib.util.spec_from_file_location("root_conftest", root_conftest_path)
    if spec is None or spec.loader is None:
        raise ImportError("Failed to load root conftest.py")
    mod = importlib.util.module_from_spec(spec)
    # Ensure src and root are in sys.path
    if str(REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "src"))
    spec.loader.exec_module(mod)
    return mod


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
    """Return the repository's own sub-package pyprojects.

    Selection is by git tracking rather than a directory denylist. A denylist is
    unbounded: CI runners materialise trees that never exist locally — the cargo
    registry cache under ``.cargo-home`` vendors pyo3's own ``pyproject.toml``
    declaring ``>=3.7``, which is a third-party artifact and not a claim this
    repository makes. Asking git which files it tracks answers "is this ours?"
    directly, and cannot drift as new tool caches appear.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", "-z", "*pyproject.toml"],
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:  # pragma: no cover
        pytest.skip(f"git is required to enumerate tracked pyprojects: {error}")

    tracked = [
        REPO_ROOT / entry
        for entry in result.stdout.decode("utf-8").split("\0")
        if entry
    ]
    return [path for path in tracked if path != ROOT_PYPROJECT and path.is_file()]


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


def _ci_lanes() -> list[tuple[int, ...]]:
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
    return lanes


def test_ci_matrix_starts_at_the_root_floor() -> None:
    """The ci-standard matrix must not test below the root floor.

    This job runs the root-package suite — ``core_tests`` is entirely
    ``tests/**`` and ``src/shared/python/**``. Below the root floor the conftest
    guard excludes all of it, so such a lane collects nothing and reports
    configuration noise rather than a real result. Sub-packages that declare a
    lower floor are gated on it by their own maturin build + parity workflows.
    """
    root_floor = _floor(ROOT_PYPROJECT)
    for lane in _ci_lanes():
        assert lane >= root_floor, (
            f"ci-standard lane {lane} is below the root requires-python floor "
            f"{root_floor}. That lane cannot run the root-package suite. If a "
            "sub-package needs a lower interpreter, gate it in its own maturin "
            "workflow instead of adding a lane here."
        )


def test_lower_floor_packages_keep_a_workflow_that_exercises_them() -> None:
    """Dropping the low lane must not leave a 3.10 claim with nothing behind it.

    Each sub-package declaring a floor below the root has to be covered by some
    workflow matrix that actually runs that interpreter, otherwise the claim is
    untested.
    """
    root_floor = _floor(ROOT_PYPROJECT)
    lower = [path for path in _sub_pyprojects() if _floor(path) < root_floor]
    if not lower:
        pytest.skip("no sub-package declares a floor below the root")

    workflows = REPO_ROOT / ".github" / "workflows"
    covered: set[tuple[int, ...]] = set()
    for workflow in workflows.glob("*.yml"):
        if workflow == CI_STANDARD:
            continue
        for match in _MATRIX_RE.finditer(workflow.read_text(encoding="utf-8")):
            for raw in match.group(1).split(","):
                raw = raw.strip().strip("\"'")
                if raw:
                    covered.add(tuple(int(part) for part in raw.split(".")))

    for pyproject in lower:
        own = _floor(pyproject)
        assert own in covered, (
            f"{pyproject.relative_to(REPO_ROOT)} declares >={own[0]}.{own[1]} but "
            "no workflow outside ci-standard runs that interpreter, so the claim "
            "is untested"
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
    conftest = _load_root_conftest()

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
    conftest = _load_root_conftest()

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
    conftest = _load_root_conftest()

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
    conftest = _load_root_conftest()

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
