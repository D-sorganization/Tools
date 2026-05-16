"""
TDD tests for issue #2868: rename upstream_drift_tools → sidekick with compat shim.

These tests are written BEFORE the implementation:
  - test_sidekick_package_importable: FAILS until sidekick/ is created
  - test_upstream_drift_tools_shim_imports: FAILS until shim emits DeprecationWarning
  - test_shim_and_canonical_are_same_object: shim re-exports must match sidekick
  - test_no_upstream_drift_tools_imports_in_sidekick_source: FAILS until rename done
  - test_no_upstream_drift_tools_in_tools_src_except_shim: FAILS until rename done
"""

import subprocess
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


@pytest.mark.unit
def test_no_upstream_drift_tools_imports_in_sidekick_source() -> None:
    """The sidekick package itself must not import from upstream_drift_tools (circular).

    We check only actual Python import statements (lines starting with
    'import upstream_drift_tools' or 'from upstream_drift_tools'), not
    comments or string literals that may mention the old name for
    documentation purposes.
    """
    sidekick_src = WORKTREE_ROOT / "src" / "shared" / "python" / "sidekick"
    if not sidekick_src.exists():
        pytest.fail(
            f"sidekick package directory does not exist yet: {sidekick_src}. "
            "Implement the rename first."
        )

    # Match lines that are actual import statements referencing the old name.
    # Pattern: optional whitespace, then 'import upstream_drift_tools' or
    # 'from upstream_drift_tools'.
    result = subprocess.run(
        [
            "grep",
            "-rn",
            "--include=*.py",
            r"^\s*\(import upstream_drift_tools\|from upstream_drift_tools\)",
            str(sidekick_src),
        ],
        capture_output=True,
        text=True,
    )
    # grep returns 0 when it FINDS matches — that's a failure for us
    assert result.returncode != 0, (
        f"Found upstream_drift_tools import statements inside sidekick/ source "
        f"(would be circular via shim):\n{result.stdout}"
    )


@pytest.mark.unit
def test_no_upstream_drift_tools_in_tools_src_except_shim() -> None:
    """Only the shim package directory may use old-name import statements in Tools src/.

    We check only actual Python import statements (lines starting with
    'import upstream_drift_tools' or 'from upstream_drift_tools'), not
    comments or string literals.
    """
    src_root = WORKTREE_ROOT / "src"

    result = subprocess.run(
        [
            "grep",
            "-rn",
            "--include=*.py",
            r"^\s*\(import upstream_drift_tools\|from upstream_drift_tools\)",
            str(src_root),
            "--exclude-dir=upstream_drift_tools",
        ],
        capture_output=True,
        text=True,
        cwd=str(WORKTREE_ROOT),
    )

    hits = result.stdout.strip()
    assert hits == "", (
        f"Found old 'upstream_drift_tools' import statements in src/ "
        f"outside the shim directory:\n{hits}"
    )
