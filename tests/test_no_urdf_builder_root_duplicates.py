"""Mirror guard test: no root-level urdf_builder_gui duplicates (#3346 / GH1693).

This test mirrors ``src/urdf_builder_gui/tests/test_urdf_builder_gui.py``
``TestNoRootLevelDuplicates`` so it runs under the project's standard
``testpaths = ["tests"]`` collection (pyproject.toml:228).

The guard test in src/ is authoritative; this file just wires it into the
default CI collection so silent re-introductions are caught automatically.

Issue #3346: the original guard existed but lived in src/urdf_builder_gui/tests/
which is outside the default test collection, so the three duplicates were
re-introduced without failing CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Location constants
# ---------------------------------------------------------------------------

# Root of the urdf_builder_gui source directory (contains __init__.py).
# Two parents: tests/ → repo_root; then navigate into src/urdf_builder_gui/.
_URDF_ROOT = Path(__file__).resolve().parent.parent / "src" / "urdf_builder_gui"

# Canonical package directory (single source of truth).
_CANONICAL_PKG = _URDF_ROOT / "python" / "urdf_builder_gui"

# Module names that must NOT exist at the root level (flat copies).
_FORBIDDEN_ROOT_MODULES = [
    "anthropometric_model.py",
    "contracts.py",
    "preview_generator.py",
    "theme.py",
    "urdf_generator.py",
]


class TestNoRootLevelDuplicates:
    """Guard: root-level module copies must not exist (DRY, GH1693, #3346).

    The canonical location for urdf_builder_gui modules is::

        src/urdf_builder_gui/python/urdf_builder_gui/

    Root-level copies shadow the canonical package (depending on __path__
    search order) and create divergent code paths between the GUI and the
    web viewer.  They must never be re-introduced.
    """

    @pytest.mark.parametrize("module_name", _FORBIDDEN_ROOT_MODULES)
    def test_root_level_copy_absent(self, module_name: str) -> None:
        """Each listed module must be absent from the flat root dir (#3346)."""
        flat_copy = _URDF_ROOT / module_name
        assert not flat_copy.exists(), (
            f"Root-level module copy re-introduced (DRY violation, GH1693, #3346): "
            f"{flat_copy.relative_to(_URDF_ROOT.parent.parent)}"
        )

    def test_canonical_package_exists(self) -> None:
        """Sanity check: canonical package directory must be present."""
        assert _CANONICAL_PKG.is_dir(), (
            f"Canonical package directory missing: {_CANONICAL_PKG}. "
            "Tests cannot guard against duplicates if the canonical copy is absent."
        )

    def test_canonical_modules_present(self) -> None:
        """Essential modules must exist in the canonical package directory."""
        essential = [
            "urdf_generator.py",
            "anthropometric_model.py",
            "preview_generator.py",
        ]
        missing = [m for m in essential if not (_CANONICAL_PKG / m).exists()]
        assert not missing, (
            "Canonical package is missing essential modules: " + ", ".join(missing)
        )

    def test_path_bridge_uses_insert_not_append(self) -> None:
        """__init__.py must use __path__.insert(0, …) not append (#3346).

        The insert ensures the canonical tree wins even if a flat copy is
        accidentally re-introduced.
        """
        init_source = (_URDF_ROOT / "__init__.py").read_text(encoding="utf-8")
        assert "__path__.insert(0, _canonical_str)" in init_source, (
            "src/urdf_builder_gui/__init__.py must use __path__.insert(0, …) "
            "to guarantee canonical-tree precedence (issue #3346). "
            "Found append() instead, which lets flat copies shadow canonical modules."
        )
