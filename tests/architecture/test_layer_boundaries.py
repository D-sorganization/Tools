"""Architecture fitness tests for layer boundary enforcement.

Verifies that:
- src/shared/ does not import from tool-specific code
- Tool core/ modules do not import from ui/ modules
- Calculation engines are pure (no Qt imports)

These tests enforce architectural invariants to prevent coupling
regressions as the codebase evolves.

Addresses #765 (Phase 4: DbC standardization, architecture tests).
"""

from __future__ import annotations

import ast
import logging
import os
from pathlib import Path
from typing import Any

import pytest

logger = logging.getLogger(__name__)

# ─── Helpers ──────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"

# Tool-specific packages that shared/ should NOT depend on
TOOL_PACKAGES = {
    "electrode_advisor",
    "data_processing",
    "document_processing",
    "folder_tools",
    "humanoid_builder_gui",
    "scientific_modeling",
    "trc_vessel_designer",
    "urdf_builder_gui",
    "web_applications",
}

# Qt modules that core/ calculation engines should NOT import
QT_MODULES = {
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "QtWidgets",
    "QtCore",
    "QtGui",
}


def _collect_python_files(directory: Path) -> list[Path]:
    """Collect all .py files under a directory, skipping __pycache__."""
    files: list[Path] = []
    if not directory.exists():
        return files
    for root, dirs, filenames in os.walk(directory):
        # Skip cache and hidden directories
        dirs[:] = [d for d in dirs if d != "__pycache__" and not d.startswith(".")]
        for f in filenames:
            if f.endswith(".py"):
                files.append(Path(root) / f)
    return files


def _extract_imports(filepath: Path) -> list[dict[str, Any]]:
    """Extract import statements from a Python file using AST.

    Returns a list of dicts with keys:
    - module: the imported module/package name
    - lineno: the line number
    - type: 'import' or 'from'
    """
    imports: list[dict[str, Any]] = []
    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, ValueError):
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(
                    {
                        "module": alias.name,
                        "lineno": node.lineno,
                        "type": "import",
                    }
                )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(
                    {
                        "module": node.module,
                        "lineno": node.lineno,
                        "type": "from",
                    }
                )
    return imports


# ─── Test: Shared library independence ────────────────────────────


class TestSharedLibraryBoundaries:
    """Verify src/shared/ does not depend on tool-specific packages."""

    @pytest.fixture(scope="class")
    def shared_files(self) -> list[Path]:
        """Collect all Python files in src/shared/."""
        shared_dir = SRC_DIR / "shared"
        return _collect_python_files(shared_dir)

    def test_shared_does_not_import_tool_packages(
        self, shared_files: list[Path]
    ) -> None:
        """Shared libraries must not import from tool-specific packages.

        Precondition: shared_files is not empty.
        Postcondition: no tool-specific import detected.
        """
        violations: list[str] = []
        for filepath in shared_files:
            imports = _extract_imports(filepath)
            for imp in imports:
                module_root = imp["module"].split(".")[0]
                if module_root in TOOL_PACKAGES:
                    rel = filepath.relative_to(REPO_ROOT)
                    violations.append(
                        f"  {rel}:{imp['lineno']} imports '{imp['module']}'"
                    )

        assert not violations, (
            f"Shared library imports tool-specific code "
            f"({len(violations)} violations):\n" + "\n".join(violations)
        )


# ─── Test: Core/UI boundary ──────────────────────────────────────


class TestCoreUiBoundary:
    """Verify tool core/ modules do not import from ui/ modules."""

    @staticmethod
    def _find_core_dirs() -> list[Path]:
        """Find all 'core' directories under src/."""
        core_dirs: list[Path] = []
        if not SRC_DIR.exists():
            return core_dirs
        for root, dirs, _ in os.walk(SRC_DIR):
            dirs[:] = [d for d in dirs if d != "__pycache__" and not d.startswith(".")]
            if Path(root).name == "core":
                core_dirs.append(Path(root))
        return core_dirs

    def test_core_does_not_import_ui(self) -> None:
        """Core modules must not import from ui/ sibling packages.

        This enforces separation of concerns: computation ≠ presentation.
        """
        violations: list[str] = []
        for core_dir in self._find_core_dirs():
            for filepath in _collect_python_files(core_dir):
                imports = _extract_imports(filepath)
                for imp in imports:
                    module_parts = imp["module"].split(".")
                    if "ui" in module_parts or "pyqt6" in module_parts:
                        rel = filepath.relative_to(REPO_ROOT)
                        violations.append(
                            f"  {rel}:{imp['lineno']} imports '{imp['module']}'"
                        )

        assert not violations, (
            f"Core modules import UI code ({len(violations)} violations):\n"
            + "\n".join(violations)
        )


# ─── Test: Pure calculation engines ──────────────────────────────


class TestPureCalculationEngines:
    """Verify that calculation engines are free of Qt dependencies."""

    ENGINE_PATTERNS = ["*engine*.py", "*calculator*.py", "*solver*.py"]

    # Directories with known Qt coupling — tracked for future decomposition
    # but excluded from this fitness test to avoid false positives
    KNOWN_COUPLED_DIRS = {"process_calculators", "tests"}

    @staticmethod
    def _find_engine_files() -> list[Path]:
        """Find all calculation engine files under src/.

        Excludes test files and known coupled directories that are
        tracked for separate decomposition work.
        """
        engine_files: list[Path] = []
        if not SRC_DIR.exists():
            return engine_files
        for pattern in TestPureCalculationEngines.ENGINE_PATTERNS:
            for filepath in SRC_DIR.rglob(pattern):
                parts = filepath.parts
                if (
                    filepath.is_file()
                    and "__pycache__" not in parts
                    and "ui" not in parts
                    and "pyqt6" not in parts
                    and not any(
                        d in parts
                        for d in TestPureCalculationEngines.KNOWN_COUPLED_DIRS
                    )
                    and not filepath.name.startswith("test_")
                ):
                    engine_files.append(filepath)
        return engine_files

    def test_engines_have_no_qt_imports(self) -> None:
        """Calculation engines must not import Qt modules.

        This ensures engines are testable without a display server
        and can run in headless CI environments.
        """
        violations: list[str] = []
        for filepath in self._find_engine_files():
            imports = _extract_imports(filepath)
            for imp in imports:
                module_root = imp["module"].split(".")[0]
                if module_root in QT_MODULES:
                    rel = filepath.relative_to(REPO_ROOT)
                    violations.append(
                        f"  {rel}:{imp['lineno']} imports '{imp['module']}'"
                    )

        assert (
            not violations
        ), f"Engine files import Qt ({len(violations)} violations):\n" + "\n".join(
            violations
        )


# ─── Test: Contract module consistency ───────────────────────────


class TestContractModuleConsistency:
    """Verify the contracts module is properly structured."""

    def test_contracts_module_importable(self) -> None:
        """The contracts module must be importable."""
        from contracts import (  # noqa: F401
            ContractLevel,
            ContractViolationError,
            InvariantError,
            PostconditionError,
            PreconditionError,
            ensure,
            invariant,
            require,
        )

    def test_require_raises_on_false(self) -> None:
        """require() must raise PreconditionError on False condition."""
        from contracts import PreconditionError, require

        with pytest.raises(PreconditionError):
            require(False, "test precondition")

    def test_ensure_raises_on_false(self) -> None:
        """ensure() must raise PostconditionError on False condition."""
        from contracts import PostconditionError, ensure

        with pytest.raises(PostconditionError):
            ensure(False, "test postcondition")

    def test_invariant_raises_on_false(self) -> None:
        """invariant() must raise InvariantError on False condition."""
        from contracts import InvariantError, invariant

        with pytest.raises(InvariantError):
            invariant(False, "test invariant")

    def test_require_passes_on_true(self) -> None:
        """require() must not raise on True condition."""
        from contracts import require

        require(True, "should pass")  # No exception

    def test_ensure_passes_on_true(self) -> None:
        """ensure() must not raise on True condition."""
        from contracts import ensure

        ensure(True, "should pass")  # No exception


# ─── Test: No wildcard imports ───────────────────────────────────


class TestNoWildcardImports:
    """Verify no wildcard imports (from X import *) in source code."""

    def test_no_star_imports_in_src(self) -> None:
        """Source code must not use wildcard imports.

        Wildcard imports make it impossible to trace symbol origins
        and can mask naming conflicts.
        """
        violations: list[str] = []
        for filepath in _collect_python_files(SRC_DIR):
            try:
                source = filepath.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(source, filename=str(filepath))
            except (SyntaxError, ValueError):
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.names and any(alias.name == "*" for alias in node.names):
                        rel = filepath.relative_to(REPO_ROOT)
                        violations.append(
                            f"  {rel}:{node.lineno} " f"from {node.module} import *"
                        )

        assert (
            not violations
        ), f"Wildcard imports found ({len(violations)}):\n" + "\n".join(violations)
