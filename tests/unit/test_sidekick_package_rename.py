"""
TDD tests for issue #2868: rename upstream_drift_tools → sidekick with compat shim.

These tests are written BEFORE the implementation:
  - test_sidekick_package_importable: FAILS until sidekick/ is created
  - test_upstream_drift_tools_shim_imports: FAILS until shim emits DeprecationWarning
  - test_shim_and_canonical_are_same_object: shim re-exports must match sidekick
  - test_no_upstream_drift_tools_imports_in_sidekick_source: FAILS until rename done
  - test_no_upstream_drift_tools_in_tools_src_except_shim: FAILS until rename done
"""

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

WORKTREE_ROOT = Path(__file__).resolve().parents[2]


def _run_import_probe(code: str) -> subprocess.CompletedProcess[str]:
    """Run import-identity probes without mutating pytest's module cache."""
    env = os.environ.copy()
    roots = [
        str(WORKTREE_ROOT / "src"),
        str(WORKTREE_ROOT / "src" / "python" / "src"),
    ]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        roots.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(roots)

    return subprocess.run(
        [
            sys.executable,
            "-W",
            "always::DeprecationWarning",
            "-c",
            textwrap.dedent(code),
        ],
        cwd=WORKTREE_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _assert_import_probe_succeeds(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.unit
def test_sidekick_package_importable() -> None:
    """The new canonical name must be importable."""
    result = _run_import_probe(
        """
        import sidekick

        assert sidekick is not None
        """
    )
    _assert_import_probe_succeeds(result)


@pytest.mark.unit
def test_upstream_drift_tools_shim_imports() -> None:
    """Old name still works (backward compat) and emits a DeprecationWarning."""
    result = _run_import_probe(
        """
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import upstream_drift_tools

        deprecation_warnings = [
            warning
            for warning in caught
            if issubclass(warning.category, DeprecationWarning)
            and "deprecated" in str(warning.message).lower()
        ]
        assert deprecation_warnings, (
            "Expected at least one DeprecationWarning about 'deprecated' from "
            f"the shim, but got: {[str(warning.message) for warning in caught]}"
        )
        """
    )
    _assert_import_probe_succeeds(result)


@pytest.mark.unit
def test_shim_and_canonical_are_same_object() -> None:
    """Shim re-exports point to the same canonical sidekick objects (no duplication)."""
    result = _run_import_probe(
        """
        import warnings

        import sidekick.data_processing

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            import upstream_drift_tools.data_processing

        assert sidekick.data_processing is upstream_drift_tools.data_processing, (
            "sidekick.data_processing and upstream_drift_tools.data_processing "
            "must be the same module object (shim must proxy, not copy)"
        )
        """
    )
    _assert_import_probe_succeeds(result)


def _find_import_violations(root_path: Path, exclude_dir_name: str = None) -> list[str]:
    """Find import statements referencing the old name in .py files."""
    pattern = re.compile(r"^\s*(import upstream_drift_tools|from upstream_drift_tools)")
    violations = []
    for py_file in root_path.rglob("*.py"):
        if exclude_dir_name and exclude_dir_name in py_file.parts:
            continue
        try:
            with open(py_file, encoding="utf-8", errors="ignore") as f:
                for line_idx, line in enumerate(f, 1):
                    if pattern.match(line):
                        rel_path = py_file.relative_to(root_path)
                        violations.append(f"{rel_path}:{line_idx}:{line.strip()}")
        except Exception:
            pass
    return violations


@pytest.mark.unit
def test_no_upstream_drift_tools_imports_in_sidekick_source() -> None:
    """The sidekick package itself must not import from upstream_drift_tools.

    Ensures no circular imports via the shim.
    """
    sidekick_src = WORKTREE_ROOT / "src" / "shared" / "python" / "sidekick"
    if not sidekick_src.exists():
        pytest.fail(
            "sidekick package directory does not exist yet: Implement the rename first."
        )
    violations = _find_import_violations(sidekick_src)
    assert not violations, (
        "Found upstream_drift_tools import statements inside sidekick/ source:\n"
        + "\n".join(violations)
    )


@pytest.mark.unit
def test_no_upstream_drift_tools_in_tools_src_except_shim() -> None:
    """Only the shim package directory may use old-name import statements."""
    src_root = WORKTREE_ROOT / "src"
    violations = _find_import_violations(
        src_root, exclude_dir_name="upstream_drift_tools"
    )
    assert not violations, (
        "Found old 'upstream_drift_tools' import statements in src/ "
        "outside shim directory:\n" + "\n".join(violations)
    )
