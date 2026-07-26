"""Regression tests for production shared import aliasing (#3316)."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):  # noqa: UP036 - CI still runs a 3.10 lane.
    import tomllib
else:  # pragma: no cover - Python 3.10 compatibility.
    import tomli as tomllib

from shared.python import import_aliases
from shared.python.import_aliases import install_shared_import_aliases


def test_canonical_alias_loader_delegates_runpy_code_lookup() -> None:
    """`python -m <alias>` must execute the canonical module code."""
    expected_code = compile("sentinel = True", "<canonical>", "exec")
    requested: list[str] = []

    class CodeLoader:
        def get_code(self, fullname: str):
            requested.append(fullname)
            return expected_code

    canonical_name = "shared.python.sidekick.__main__"
    spec = ModuleSpec(canonical_name, CodeLoader())
    loader = import_aliases._CanonicalAliasLoader(  # type: ignore[attr-defined]
        spec,
        [canonical_name, "sidekick.__main__"],
    )

    assert loader.get_code("sidekick.__main__") is expected_code
    assert requested == [canonical_name]


def test_sidekick_aliases_share_one_registry_module() -> None:
    install_shared_import_aliases()

    canonical = importlib.import_module(
        "shared.python.sidekick.ui.tools_sidebar.registry"
    )
    importlib.import_module("sidekick.ui.tools_sidebar.registry")
    importlib.import_module("upstream_drift_tools.ui.tools_sidebar.registry")
    importlib.import_module("src.shared.python.sidekick.ui.tools_sidebar.registry")

    aliases = (
        "sidekick.ui.tools_sidebar.registry",
        "upstream_drift_tools.ui.tools_sidebar.registry",
        "src.shared.python.sidekick.ui.tools_sidebar.registry",
    )
    assert all(sys.modules[name] is canonical for name in aliases)
    assert importlib.import_module("sidekick.ui.tools_sidebar.registry") is canonical
    assert (
        importlib.import_module("upstream_drift_tools.ui.tools_sidebar.registry")
        is canonical
    )
    assert (
        importlib.import_module("src.shared.python.sidekick.ui.tools_sidebar.registry")
        is canonical
    )


def test_theme_and_compatibility_aliases_share_identity() -> None:
    install_shared_import_aliases()

    theme = importlib.import_module("shared.python.theme.theme_manager")
    importlib.import_module("theme.theme_manager")
    importlib.import_module("src.shared.python.theme.theme_manager")
    assert sys.modules["theme.theme_manager"] is theme
    assert sys.modules["src.shared.python.theme.theme_manager"] is theme

    compatibility = importlib.import_module("shared.python.compatibility")
    importlib.import_module("compatibility")
    importlib.import_module("src.shared.python.compatibility")
    assert sys.modules["compatibility"] is compatibility
    assert sys.modules["src.shared.python.compatibility"] is compatibility


def test_installer_coalesces_preloaded_legacy_alias() -> None:
    install_shared_import_aliases()

    canonical = importlib.import_module("shared.python.theme.theme_manager")
    stale_alias = type(sys)("theme.theme_manager")
    sys.modules["theme.theme_manager"] = stale_alias

    install_shared_import_aliases()

    assert sys.modules["theme.theme_manager"] is canonical
    assert importlib.import_module("theme.theme_manager") is canonical


def test_bootstrap_uses_src_root_not_shared_python_root() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = """
from _bootstrap import bootstrap
import sys
root = bootstrap('UnifiedToolsLauncher.py')
assert str(root / 'src') in sys.path
assert str(root / 'src' / 'shared' / 'python') not in sys.path
"""
    env = {**os.environ, "PYTHONPATH": str(repo_root / "src")}
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        check=True,
    )


def test_sidekick_bootstrap_uses_src_root_not_shared_python_root() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = """
from shared.python.sidekick.bootstrap import ensure_paths
import sys
root = ensure_paths()
assert str(root / 'src') in sys.path
assert str(root / 'src' / 'shared' / 'python') not in sys.path
"""
    env = {**os.environ, "PYTHONPATH": str(repo_root / "src")}
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        check=True,
    )


def test_setuptools_no_longer_discovers_shared_python_as_package_root() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pyproject: dict[str, Any] = tomllib.loads(
        (repo_root / "pyproject.toml").read_text()
    )
    package_roots = pyproject["tool"]["setuptools"]["packages"]["find"]["where"]

    assert "src/shared/python" not in package_roots
    assert "src" in package_roots


def test_legacy_package_shims_share_canonical_package_identity() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = """
import importlib
import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore', DeprecationWarning)
    sidekick = importlib.import_module('sidekick')
    upstream = importlib.import_module('upstream_drift_tools')

shared_sidekick = importlib.import_module('shared.python.sidekick')
theme = importlib.import_module('theme')
shared_theme = importlib.import_module('shared.python.theme')
chat = importlib.import_module('chat')
shared_chat = importlib.import_module('shared.python.chat')
sidekick_process = importlib.import_module('sidekick.process_calculators')
shared_process = importlib.import_module('shared.python.sidekick.process_calculators')
upstream_process = importlib.import_module('upstream_drift_tools.process_calculators')

assert sidekick is shared_sidekick
assert upstream is shared_sidekick
assert theme is shared_theme
assert chat is shared_chat
assert sidekick_process is shared_process
assert upstream_process is shared_process
"""
    env = {**os.environ, "PYTHONPATH": str(repo_root / "src")}
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        check=True,
    )
