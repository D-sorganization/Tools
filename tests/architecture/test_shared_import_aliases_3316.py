"""Regression tests for production shared import aliasing (#3316)."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Any

if sys.version_info >= (3, 11):  # noqa: UP036 - CI still runs a 3.10 lane.
    import tomllib
else:  # pragma: no cover - Python 3.10 compatibility.
    import tomli as tomllib

from shared.python import import_aliases
from shared.python.import_aliases import install_shared_import_aliases


def _write_downstream_src_package(tmp_path: Path) -> Path:
    downstream_root = tmp_path / "downstream"
    package_root = downstream_root / "src" / "shared" / "python"
    package_root.mkdir(parents=True)
    (downstream_root / "src" / "__init__.py").write_text(
        "DOWNSTREAM_SRC = True\n",
        encoding="utf-8",
    )
    (downstream_root / "src" / "shared" / "__init__.py").write_text(
        "DOWNSTREAM_SHARED = True\n",
        encoding="utf-8",
    )
    (package_root / "__init__.py").write_text(
        "DOWNSTREAM_PYTHON = True\n",
        encoding="utf-8",
    )
    (package_root / "domain_module.py").write_text(
        "VALUE = 'downstream-owned'\n",
        encoding="utf-8",
    )
    for leaf in ("sidekick", "chat"):
        copied_leaf = package_root / leaf
        copied_leaf.mkdir()
        (copied_leaf / "__init__.py").write_text(
            "DOWNSTREAM_COPY = True\n",
            encoding="utf-8",
        )
    downstream_config = package_root / "config"
    downstream_config.mkdir()
    (downstream_config / "__init__.py").write_text(
        "DOWNSTREAM_CONFIG = True\n",
        encoding="utf-8",
    )
    return downstream_root


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


def test_finder_does_not_redirect_internal_test_modules() -> None:
    """``<root>.tests.<...>`` must load normally instead of being aliased.

    Downstream repos import the shared packages' own test modules directly
    (e.g. ``from sidekick.tests.calculators.conversion.test_conversion import
    ...``). Redirecting those names through the alias machinery resolves them
    to the wrong module object, so the finder must decline them outright.
    """
    finder = import_aliases.SharedImportAliasFinder()

    declined = (
        "sidekick.tests",
        "sidekick.tests.calculators",
        "sidekick.tests.calculators.conversion.test_conversion",
        "upstream_drift_tools.tests.test_data_io",
        "src.shared.python.sidekick.tests.test_data_io",
        "ai.tests.test_access_policy",
        "shared.python.ai.tests.test_access_policy",
    )
    for fullname in declined:
        assert finder.find_spec(fullname) is None, fullname
        assert finder._parse(fullname) == (None, ""), fullname


def test_finder_still_redirects_non_test_submodules() -> None:
    """The ``.tests`` carve-out must not disable ordinary alias redirection."""
    finder = import_aliases.SharedImportAliasFinder()

    for fullname in (
        "sidekick.process_calculators",
        "sidekick.ui.tools_sidebar.registry",
    ):
        assert finder.find_spec(fullname) is not None, fullname


def test_internal_test_package_imports_under_alias_root() -> None:
    """A shared package's own ``tests`` subpackage is importable, not aliased."""
    install_shared_import_aliases()

    module = importlib.import_module("sidekick.tests")

    assert module.__name__ == "sidekick.tests"
    canonical_root = Path(
        importlib.import_module("shared.python.sidekick").__file__ or ""
    ).parent
    assert Path(module.__file__ or "").parent == canonical_root / "tests"


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


def test_installer_binds_legacy_src_parent_namespaces(
    monkeypatch,
) -> None:
    """Installed apps must reach canonical modules through ``src.shared``."""
    shared = importlib.import_module("shared")
    shared_python = importlib.import_module("shared.python")
    legacy_src = ModuleType("src")
    legacy_src.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src", legacy_src)
    monkeypatch.delitem(sys.modules, "src.shared", raising=False)
    monkeypatch.delitem(sys.modules, "src.shared.python", raising=False)

    install_shared_import_aliases()

    assert sys.modules["src.shared"] is shared
    assert sys.modules["src.shared.python"] is shared_python
    assert legacy_src.shared is shared  # type: ignore[attr-defined]


def test_installer_preserves_downstream_src_packages_and_aliases_tools_leaves(
    tmp_path: Path,
) -> None:
    """A downstream ``src.shared.python`` package must survive Tools bootstrap."""
    repo_root = Path(__file__).resolve().parents[2]
    downstream_root = _write_downstream_src_package(tmp_path)
    script = """
import importlib
import sys

downstream_shared = importlib.import_module("src.shared")
downstream_python = importlib.import_module("src.shared.python")
domain_module = importlib.import_module("src.shared.python.domain_module")

from shared.python.import_aliases import install_shared_import_aliases

install_shared_import_aliases()

assert sys.modules["src.shared"] is downstream_shared
assert sys.modules["src.shared.python"] is downstream_python
assert downstream_shared.DOWNSTREAM_SHARED is True
assert downstream_python.DOWNSTREAM_PYTHON is True
assert domain_module.VALUE == "downstream-owned"
assert importlib.import_module("src.shared.python.config").DOWNSTREAM_CONFIG is True

for leaf in ("sidekick", "chat"):
    canonical = importlib.import_module(f"shared.python.{leaf}")
    direct = importlib.import_module(leaf)
    legacy = importlib.import_module(f"src.shared.python.{leaf}")
    assert direct is canonical
    assert legacy is canonical
    assert not hasattr(canonical, "DOWNSTREAM_COPY")

assert importlib.import_module("src.shared.python.domain_module") is domain_module
"""
    python_path = os.pathsep.join((str(downstream_root), str(repo_root / "src")))
    env = {**os.environ, "PYTHONPATH": python_path}

    subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        check=True,
    )


def test_installer_does_not_shadow_an_available_downstream_src_package(
    tmp_path: Path,
) -> None:
    """Tools bootstrap must preserve downstream packages regardless of import order."""
    repo_root = Path(__file__).resolve().parents[2]
    downstream_root = _write_downstream_src_package(tmp_path)
    script = """
import importlib

downstream_src = importlib.import_module("src")
assert downstream_src.DOWNSTREAM_SRC is True

from shared.python.import_aliases import install_shared_import_aliases

install_shared_import_aliases()

downstream_shared = importlib.import_module("src.shared")
downstream_python = importlib.import_module("src.shared.python")
domain_module = importlib.import_module("src.shared.python.domain_module")
assert downstream_shared.DOWNSTREAM_SHARED is True
assert downstream_python.DOWNSTREAM_PYTHON is True
assert domain_module.VALUE == "downstream-owned"
assert importlib.import_module("src.shared.python.config").DOWNSTREAM_CONFIG is True

for leaf in ("sidekick", "chat"):
    canonical = importlib.import_module(f"shared.python.{leaf}")
    assert importlib.import_module(leaf) is canonical
    assert importlib.import_module(f"src.shared.python.{leaf}") is canonical
    assert not hasattr(canonical, "DOWNSTREAM_COPY")
"""
    python_path = os.pathsep.join((str(downstream_root), str(repo_root / "src")))
    env = {**os.environ, "PYTHONPATH": python_path}

    subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        check=True,
    )


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
src_sidekick = importlib.import_module('src.shared.python.sidekick')
src_chat = importlib.import_module('src.shared.python.chat')
sidekick_process = importlib.import_module('sidekick.process_calculators')
shared_process = importlib.import_module('shared.python.sidekick.process_calculators')
upstream_process = importlib.import_module('upstream_drift_tools.process_calculators')

assert sidekick is shared_sidekick
assert src_sidekick is shared_sidekick
assert upstream is shared_sidekick
assert theme is shared_theme
assert chat is shared_chat
assert src_chat is shared_chat
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
