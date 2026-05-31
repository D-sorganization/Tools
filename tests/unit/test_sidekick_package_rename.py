"""
TDD tests for issue #2868: rename upstream_drift_tools → sidekick with compat shim.

These tests are written BEFORE the implementation:
  - test_sidekick_package_importable: FAILS until sidekick/ is created
  - test_upstream_drift_tools_shim_imports: FAILS until shim emits DeprecationWarning
  - test_shim_and_canonical_are_same_object: shim re-exports must match sidekick
  - test_no_upstream_drift_tools_imports_in_sidekick_source: FAILS until rename done
  - test_no_upstream_drift_tools_in_tools_src_except_shim: FAILS until rename done
"""

import sys
import warnings
from pathlib import Path

import pytest

WORKTREE_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.unit
def test_sidekick_package_importable() -> None:
    """The new canonical name must be importable."""
    # Remove any cached import to get a clean slate
    for key in list(sys.modules.keys()):
        if key == "sidekick" or key.startswith("sidekick."):
            del sys.modules[key]

    import sidekick  # noqa: F401 — should not raise

    assert sidekick is not None


@pytest.mark.unit
def test_upstream_drift_tools_shim_imports() -> None:
    """Old name still works (backward compat) and emits a DeprecationWarning."""
    # Remove cached module so we can observe the warning fresh
    for key in list(sys.modules.keys()):
        if key == "upstream_drift_tools" or key.startswith("upstream_drift_tools."):
            del sys.modules[key]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import upstream_drift_tools  # noqa: F401

        deprecation_warnings = [
            w
            for w in caught
            if issubclass(w.category, DeprecationWarning)
            and "deprecated" in str(w.message).lower()
        ]
        assert deprecation_warnings, (
            "Expected at least one DeprecationWarning about 'deprecated' from "
            f"the shim, but got: {[str(w.message) for w in caught]}"
        )


@pytest.mark.unit
def test_shim_and_canonical_are_same_object() -> None:
    """Shim re-exports point to the same canonical sidekick objects (no duplication)."""
    # Clear caches to guarantee a fresh load order
    for key in list(sys.modules.keys()):
        if key.startswith("sidekick") or key.startswith("upstream_drift_tools"):
            del sys.modules[key]

    # Load via canonical path
    import sidekick.data_processing

    # Load via shim (suppress the deprecation warning — we test it separately)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import upstream_drift_tools.data_processing

    assert sidekick.data_processing is upstream_drift_tools.data_processing, (
        "sidekick.data_processing and upstream_drift_tools.data_processing "
        "must be the same module object (shim must proxy, not copy)"
    )


import re


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
