"""Architecture fitness tests for layer boundary enforcement.

Verifies that:
- src/shared/ does not import from tool-specific code
- Tool core/ modules do not import from ui/ modules
- Calculation engines are pure (no Qt imports)
- No circular dependencies between shared sub-packages
- Source libraries use logging, not print()
- Typed exception modules define proper hierarchies

These tests enforce architectural invariants to prevent coupling
regressions as the codebase evolves.

Addresses #765 (Phase 4), #832 (orthogonality/boundary checks).
"""

from __future__ import annotations

import ast
import importlib
import logging
import os
import sys
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
    "humanoid_builder_gui",
    "scientific_modeling",
    "rotation_converter",
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


def _collect_python_files(directory: Path, skip_tests: bool = True) -> list[Path]:
    """Collect all .py files under a directory, skipping __pycache__ and optionally tests."""
    files: list[Path] = []
    if not directory.exists():
        return files
    for root, dirs, filenames in os.walk(directory):
        # Skip cache, hidden, and (optionally) test directories
        dirs[:] = [
            d
            for d in dirs
            if d != "__pycache__"
            and not d.startswith(".")
            and not (skip_tests and d in {"tests", "test"})
        ]
        for f in filenames:
            if f.endswith(".py") and not (skip_tests and f.startswith("test_")):
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
                        "level": node.level,  # 0=absolute, >0=relative
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
                # Skip relative imports — they stay within the package
                # and cannot cross the shared→tool boundary
                if imp.get("level", 0) > 0:
                    continue
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

        msg = f"Engine files import Qt ({len(violations)} violations):\n" + "\n".join(
            violations
        )
        assert not violations, msg


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
                            f"  {rel}:{node.lineno} from {node.module} import *"
                        )

        msg = f"Wildcard imports found ({len(violations)}):\n" + "\n".join(violations)
        assert not violations, msg


# ─── Test: No bare print() in library code ────────────────────────


class TestNoPrintInLibraryCode:
    """Verify that src/shared/ library code uses logging, not print()."""

    # Directories whose code legitimately uses print (CLI entry-points, debug
    # utilities that explicitly support file=... output, etc.)
    _ALLOWED_DIRS = {"tests", "scripts", "cli"}
    _ALLOWED_FILES = {"__main__.py", "cli.py", "mcp_server.py", "watcher.py"}

    def test_shared_library_has_no_bare_print_calls(self) -> None:
        """Library code under src/shared/ must use logging, not print().

        Debug utilities may write to a file argument, but raw print()
        calls to stdout indicate missing logging discipline.
        """
        violations: list[str] = []
        shared_dir = SRC_DIR / "shared"
        for filepath in _collect_python_files(shared_dir):
            # Skip test helpers and CLI entry-points
            parts = set(filepath.parts)
            if parts & self._ALLOWED_DIRS:
                continue
            if filepath.name in self._ALLOWED_FILES:
                continue

            try:
                source = filepath.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(source, filename=str(filepath))
            except (SyntaxError, ValueError):
                continue

            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "print"
                ):
                    rel = filepath.relative_to(REPO_ROOT)
                    violations.append(f"  {rel}:{node.lineno} print()")

        detail = "\n".join(violations)
        msg = f"Shared library uses print() ({len(violations)} calls):\n{detail}"
        assert not violations, msg


# ─── Test: No circular imports between shared sub-packages ────────


class TestNoCrossCouplingInShared:
    """Verify shared sub-packages only use allowed dependency directions.

    The shared layer has sub-packages with an explicit dependency DAG.
    Imports that violate this DAG are architectural regressions.
    """

    # Top-level sub-packages under src/shared/python/
    SHARED_PACKAGES = {
        "model_generation",
        "signal_toolkit",
        "upstream_drift_tools",
        "humanoid_character_builder",
        "plot_theme",
        "plot_engine",
        "chat",
        "theme",
        "calc_backend",
    }

    # Allowed directed dependencies: source -> {allowed targets}
    # calc_backend routers depend on upstream_drift_tools calculators
    # plot_engine depends on plot_theme for colour palettes
    # model_generation.humanoid bridges to humanoid_character_builder
    # upstream_drift_tools.theme is a deliberate re-export wrapper for
    # the sibling `theme` package (PR #896 — intentional coupling)
    ALLOWED_DEPS: dict[str, set[str]] = {
        "calc_backend": {"upstream_drift_tools"},
        "chat": {"theme"},
        "plot_engine": {"plot_theme"},
        "model_generation": {"humanoid_character_builder"},
        "upstream_drift_tools": {"theme"},
    }

    def test_no_unauthorized_cross_package_imports(self) -> None:
        """Sub-packages must not import siblings outside the allowed DAG.

        Allowed dependencies are listed in ALLOWED_DEPS. Any import
        outside that map is flagged as a violation.
        """
        shared_python = SRC_DIR / "shared" / "python"
        violations: list[str] = []

        for pkg in self.SHARED_PACKAGES:
            pkg_dir = shared_python / pkg
            if not pkg_dir.exists():
                continue
            allowed = self.ALLOWED_DEPS.get(pkg, set())
            for filepath in _collect_python_files(pkg_dir):
                imports = _extract_imports(filepath)
                for imp in imports:
                    if imp.get("level", 0) > 0:
                        continue  # skip relative
                    module_root = imp["module"].split(".")[0]
                    if (
                        module_root in self.SHARED_PACKAGES
                        and module_root != pkg
                        and module_root not in allowed
                    ):
                        rel = filepath.relative_to(REPO_ROOT)
                        violations.append(
                            f"  {rel}:{imp['lineno']} "
                            f"'{pkg}' imports unauthorized sibling "
                            f"'{module_root}'"
                        )

        assert not violations, (
            f"Unauthorized cross-package imports in shared/ "
            f"({len(violations)} violations):\n" + "\n".join(violations)
        )


# ─── Test: Exception hierarchy consistency ────────────────────────


class TestExceptionHierarchyConsistency:
    """Verify that custom exception modules define proper hierarchies."""

    def test_data_processing_exceptions_importable(self) -> None:
        """All data-processing exceptions must be importable."""
        pytest.importorskip("numpy")
        pytest.importorskip("scipy")
        from upstream_drift_tools.data_processing.exceptions import (  # noqa: F401
            ColumnNotFoundError,
            DataNotLoadedError,
            DataProcessingError,
            FileIOError,
            FilterError,
            FitError,
            TransformationError,
            UnsupportedOperationError,
        )

    def test_exception_import_does_not_eagerly_import_scipy_signal(self) -> None:
        """Exception-only imports must not initialize SciPy signal internals."""
        sys.modules.pop("upstream_drift_tools.data_processing", None)
        sys.modules.pop("upstream_drift_tools.data_processing.exceptions", None)
        sys.modules.pop("scipy.signal", None)

        importlib.import_module("upstream_drift_tools.data_processing.exceptions")

        assert "scipy.signal" not in sys.modules

    def test_all_data_processing_exceptions_share_base(self) -> None:
        """Every exception must inherit from DataProcessingError."""
        pytest.importorskip("numpy")
        pytest.importorskip("scipy")
        from upstream_drift_tools.data_processing.exceptions import (
            ColumnNotFoundError,
            DataNotLoadedError,
            DataProcessingError,
            FileIOError,
            FilterError,
            FitError,
            TransformationError,
            UnsupportedOperationError,
        )

        for exc_class in (
            DataNotLoadedError,
            ColumnNotFoundError,
            FileIOError,
            FilterError,
            FitError,
            TransformationError,
            UnsupportedOperationError,
        ):
            msg = f"{exc_class.__name__} does not inherit DataProcessingError"
            assert issubclass(exc_class, DataProcessingError), msg
