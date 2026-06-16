"""Import-boundary guard for the #3316 sidekick canonicalization slice."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
SHARED_PYTHON = SRC_ROOT / "shared" / "python"

_ALLOWED_FILES = {
    Path("src/sidekick/__init__.py"),
    Path("src/upstream_drift_tools/__init__.py"),
    Path("src/shared/python/upstream_drift_tools/__init__.py"),
}

_AMBIGUOUS_NON_SHARED_APP_ROOTS = {"data_processing"}


def _shared_python_roots() -> tuple[str, ...]:
    roots = {
        path.name
        for path in SHARED_PYTHON.iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    }
    roots.update(
        path.stem for path in SHARED_PYTHON.glob("*.py") if path.stem != "__init__"
    )
    roots.difference_update(_AMBIGUOUS_NON_SHARED_APP_ROOTS)
    roots.add("src.shared.python")
    return tuple(sorted(roots))


def _is_allowed_path(path: Path) -> bool:
    relative = path.relative_to(REPO_ROOT)
    if "tests" in relative.parts:
        return True
    return relative in _ALLOWED_FILES


def _root_matches(module: str) -> bool:
    return any(
        module == root or module.startswith(f"{root}.")
        for root in _shared_python_roots()
    )


def _duplicate_import_violations() -> list[str]:
    violations: list[str] = []
    for py_file in SRC_ROOT.rglob("*.py"):
        if _is_allowed_path(py_file):
            continue

        source = py_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                module = node.module or ""
                if _root_matches(module):
                    violations.append(
                        f"{py_file.relative_to(REPO_ROOT)}:{node.lineno}: {module}"
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if _root_matches(alias.name):
                        violations.append(
                            f"{py_file.relative_to(REPO_ROOT)}:{node.lineno}: "
                            f"{alias.name}"
                        )
    return violations


def test_production_code_uses_shared_python_imports() -> None:
    """Production consumers must use the canonical shared.python spelling."""
    assert not _duplicate_import_violations()


def test_legacy_sidekick_aliases_share_canonical_module_objects() -> None:
    """Compatibility aliases must not create duplicate sidekick modules."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src"), str(REPO_ROOT / "src" / "python" / "src")]
    )
    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "ignore::DeprecationWarning",
            "-c",
            textwrap.dedent(
                """
                import importlib

                canonical = importlib.import_module(
                    "shared.python.sidekick.ui.tools_sidebar.registry"
                )
                legacy = importlib.import_module("sidekick.ui.tools_sidebar.registry")
                old = importlib.import_module(
                    "upstream_drift_tools.ui.tools_sidebar.registry"
                )

                assert canonical is legacy
                assert canonical is old
                src_alias = __import__("sys").modules.get(
                    "src.shared.python.sidekick.ui.tools_sidebar.registry"
                )
                assert src_alias is None or src_alias is canonical
                """
            ),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
