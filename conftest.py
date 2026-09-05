"""Repository-level pytest hooks for cross-tree test discovery."""

from __future__ import annotations

import importlib
import re
import sys
import types
from pathlib import Path

import pytest

from shared.python.import_aliases import install_shared_import_aliases

# Use the same aliasing path in tests and production bootstraps.
install_shared_import_aliases()

REPO_ROOT = Path(__file__).resolve().parent


def _setup_global_stubs(repo_root: Path) -> None:
    """Setup global stubs for missing shared modules in the Tools repository.

    This ensures that when shared code imports config or logging_pkg (which
    are not physically present in Tools), they resolve to safe mock/stub modules.
    """
    import logging

    for import_root in (
        repo_root / "src",
        repo_root / "src" / "python" / "src",
        repo_root / "src" / "shared" / "python",
    ):
        import_path = str(import_root)
        if import_path in sys.path:
            sys.path.remove(import_path)
        sys.path.insert(0, import_path)

    def ensure_package_path(name: str, path: str) -> None:
        package_path = str(repo_root / path)
        module = sys.modules.get(name)
        if module is None:
            # Prefer the real package. This root conftest loads before
            # tests/conftest.py, and the synthetic ``ModuleType`` it used to
            # install here has a ``__path__`` but never executes
            # ``__init__.py`` -- so ``shared.python.config`` lacked the names
            # its ``__init__`` re-exports and ``from config import
            # EnvironmentError`` failed at collection (Tools #4913). Stub only
            # when the package genuinely cannot be imported in this checkout.
            try:
                module = importlib.import_module(name)
            except ImportError:
                module = types.ModuleType(name)
                module.__path__ = [package_path]  # type: ignore[attr-defined]
                sys.modules[name] = module
        existing_paths = list(getattr(module, "__path__", []))
        module.__path__ = [  # type: ignore[attr-defined]
            package_path,
            *(entry for entry in existing_paths if entry != package_path),
        ]

        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = sys.modules.get(parent_name)
            if parent is not None and not hasattr(parent, child_name):
                setattr(parent, child_name, module)

    # Define the package paths that must prefer this checkout.
    checkout_packages = [
        ("shared", "src/shared"),
        ("shared.python", "src/shared/python"),
        ("shared.python.calc_backend", "src/shared/python/calc_backend"),
        ("shared.python.config", "src/shared/python/config"),
        ("shared.python.logging_pkg", "src/shared/python/logging_pkg"),
    ]

    for name, path in checkout_packages:
        ensure_package_path(name, path)

    # Specifically stub logging_config
    if "shared.python.logging_pkg.logging_config" not in sys.modules:
        logging_config = types.ModuleType("shared.python.logging_pkg.logging_config")
        logging_config.get_logger = logging.getLogger  # type: ignore
        logging_config.setup_logging = lambda *a, **kw: None  # type: ignore
        sys.modules["shared.python.logging_pkg.logging_config"] = logging_config

    # ``shared.python.config.environment`` is a real module in this checkout;
    # the placeholder that used to be installed here (``get_env`` returning
    # its default) shadowed it and broke tests/shared/python/config (#4913).
    importlib.import_module("shared.python.config.environment")


_setup_global_stubs(REPO_ROOT)


@pytest.fixture
def repo_root() -> Path:
    """Return the repository root for tests that inspect source files."""
    return REPO_ROOT


BRIDGED_EMBEDDED_TEST_DIRS = {
    REPO_ROOT / "src" / "pendulum_simulator" / "tests",
    REPO_ROOT / "src" / "solar_system_model" / "solar_system" / "tests",
}


def _path_is_within(candidate: Path, parent: Path) -> bool:
    """Return whether ``candidate`` is inside ``parent``."""
    try:
        candidate.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


# --------------------------------------------------------------------------- #
# Python floor enforcement                                                     #
#                                                                              #
# This repository is deliberately two-tier. The root distribution declares      #
# ``requires-python = ">=3.11"``, while ten sub-packages and Rust crates        #
# (movement_optimizer, pendulum_simulator, rotation_converter, tools-core,      #
# swing-core, ...) declare ``>=3.10`` and ship 3.10 wheels from the maturin     #
# workflows. The CI matrix therefore runs a 3.10 lane on purpose.               #
#                                                                              #
# That lane must only exercise code that actually claims 3.10 support. Running  #
# the whole suite there tests root-package code against an interpreter it does  #
# not support, which surfaces as failures that look like real defects but are   #
# not: a bare ``import tomllib`` (stdlib only on 3.11+) aborting collection,    #
# or ``asyncio.wait_for`` cancellation semantics that changed in 3.11.          #
#                                                                              #
# The floor is read from each package's own ``pyproject.toml`` rather than      #
# hardcoded here, so adding a sub-package or moving a floor needs no edit to    #
# this file. ``requires-python`` is parsed with a regex on purpose: ``tomllib`` #
# does not exist on the very interpreter this guard has to run on.              #
# --------------------------------------------------------------------------- #

_REQUIRES_PYTHON_RE = re.compile(
    r"""^\s*requires-python\s*=\s*["'][^"']*?>=\s*(\d+)\.(\d+)""",
    re.MULTILINE,
)

_floor_cache: dict[Path, tuple[int, int]] = {}


def _declared_python_floor(directory: Path) -> tuple[int, int]:
    """Return the ``requires-python`` floor governing ``directory``.

    Walks upward to the nearest ``pyproject.toml`` inside the repository and
    returns its declared minimum. Falls back to the root declaration when no
    nearer one exists, and to ``(3, 11)`` if nothing can be parsed — failing
    closed rather than silently widening support.
    """
    cached = _floor_cache.get(directory)
    if cached is not None:
        return cached

    floor = (3, 11)
    for parent in (directory, *directory.parents):
        if not _path_is_within(parent, REPO_ROOT) and parent != REPO_ROOT:
            continue
        pyproject = parent / "pyproject.toml"
        if pyproject.is_file():
            try:
                match = _REQUIRES_PYTHON_RE.search(
                    pyproject.read_text(encoding="utf-8")
                )
            except OSError:
                match = None
            if match is not None:
                floor = (int(match.group(1)), int(match.group(2)))
            break
        if parent == REPO_ROOT:
            break

    _floor_cache[directory] = floor
    return floor


def _below_declared_floor(
    candidate: Path, running: tuple[int, int] | None = None
) -> bool:
    """Return whether ``running`` is below ``candidate``'s declared floor.

    ``running`` defaults to the live interpreter. It is a parameter so tests can
    exercise other interpreters without patching ``sys.version_info``, which is
    process-global and read by unrelated library code during a parallel run.
    """
    directory = candidate if candidate.is_dir() else candidate.parent
    version = sys.version_info[:2] if running is None else running
    return version < _declared_python_floor(directory)


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    """Avoid double-collecting embedded suites that are bridged into ``tests/``.

    If a developer explicitly targets one of the embedded directories, preserve
    the direct path-based behavior.
    """
    candidate = Path(collection_path)

    # Never collect code whose own pyproject declares a floor above the running
    # interpreter. See the Python floor enforcement note above.
    if _below_declared_floor(candidate):
        return True

    explicit_targets = [
        (config.rootpath / arg).resolve()
        for arg in config.args
        if arg and not arg.startswith("-")
    ]

    for embedded_tests_dir in BRIDGED_EMBEDDED_TEST_DIRS:
        if not _path_is_within(candidate, embedded_tests_dir):
            continue
        if any(
            _path_is_within(target, embedded_tests_dir) for target in explicit_targets
        ):
            return None
        return True

    return None
