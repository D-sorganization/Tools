"""Provider-contract tests for the legacy upstream_drift_tools shim."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _run_isolated_import_check(script: str) -> None:
    """Run an import contract without mutating the pytest process."""
    repo_root = Path(__file__).resolve().parents[4]
    env = os.environ.copy()
    source_paths = [repo_root / "src", repo_root / "src/shared/python"]
    env["PYTHONPATH"] = os.pathsep.join(
        [*(str(path) for path in source_paths), env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        check=False,
        timeout=30,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_legacy_package_reexports_sidekick_contracts() -> None:
    """Legacy imports keep resolving to the canonical sidekick API."""
    child = """
import importlib
import importlib.util
import sys
import warnings
from pathlib import Path

from shared.python.import_aliases import install_shared_import_aliases

install_shared_import_aliases()
removed = []
for finder in list(sys.meta_path):
    if finder.__class__.__name__ == "SharedImportAliasFinder":
        sys.meta_path.remove(finder)
        removed.append(finder)
assert removed, "the test did not find its expected redirector"

spec = importlib.util.find_spec("upstream_drift_tools")
expected = (Path.cwd() / "src/upstream_drift_tools/__init__.py").resolve()
assert spec is not None and spec.origin is not None
assert Path(spec.origin).resolve() == expected, (spec.origin, expected)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", DeprecationWarning)
    legacy = importlib.import_module("upstream_drift_tools")
sidekick = importlib.import_module("shared.python.sidekick")

assert legacy is sidekick
assert legacy.__version__ == sidekick.__version__
assert legacy.Calculator is sidekick.Calculator
assert legacy.ValidationResult is sidekick.ValidationResult
assert any(item.category is DeprecationWarning for item in caught)
"""
    _run_isolated_import_check(child)


def test_legacy_submodule_aliases_canonical_sidekick_modules() -> None:
    """Submodule aliases point at the same module objects as sidekick."""
    child = """
import importlib
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    legacy_theme = importlib.import_module("upstream_drift_tools.theme")
sidekick_theme = importlib.import_module("sidekick.theme")
assert legacy_theme is sidekick_theme
"""
    _run_isolated_import_check(child)
