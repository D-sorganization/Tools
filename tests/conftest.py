"""Shared pytest configuration for entire Tools repo."""

import importlib
import sys
import types
from pathlib import Path

import pytest

from shared.python.import_aliases import install_shared_import_aliases

# Use the same aliasing path in tests and production bootstraps.
install_shared_import_aliases()

REPO_ROOT = Path(__file__).resolve().parent.parent


def _setup_global_stubs(repo_root: Path) -> None:
    """Setup global stubs for missing shared modules in the Tools repository.

    This ensures that when shared code imports config or logging_pkg (which
    are not physically present in Tools), they resolve to safe mock/stub modules.
    """
    for import_root in (
        repo_root / "src",
        repo_root / "src" / "python" / "src",
    ):
        import_path = str(import_root)
        if import_path in sys.path:
            sys.path.remove(import_path)
        sys.path.insert(0, import_path)

    def ensure_package_path(name: str, path: str) -> None:
        package_path = str(repo_root / path)
        module = sys.modules.get(name)
        if module is None:
            # Prefer the real package: a synthetic ``ModuleType`` stub has a
            # ``__path__`` but never executes ``__init__.py``, so names the
            # package re-exports (``shared.python.config.EnvironmentError``)
            # were missing and ``from config import EnvironmentError`` failed
            # at collection time (Tools #4913). Fall back to a stub only when
            # the package genuinely is not importable in this checkout.
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
    ]

    for name, path in checkout_packages:
        ensure_package_path(name, path)

    # Use the real logging package rather than a placeholder package. The
    # top-level ``logging_pkg`` shim aliases to this module object, so a stub
    # here hides public exports such as DEFAULT_SEED during CI collection.
    sys.modules.pop("shared.python.logging_pkg", None)
    sys.modules.pop("shared.python.logging_pkg.logging_config", None)

    # ``shared.python.config.environment`` is a real module in this checkout;
    # the placeholder that used to be installed here (``get_env`` returning
    # its default) shadowed it and broke tests/shared/python/config. Import
    # it so downstream ``from config import get_env`` sees the real helpers.
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


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    """Avoid double-collecting embedded suites that are bridged into ``tests/``.

    If a developer explicitly targets one of the embedded directories, preserve
    the direct path-based behavior.
    """
    candidate = Path(collection_path)
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


try:
    import pytest
    from PyQt6.QtWidgets import QApplication

    @pytest.fixture(scope="session")
    def qapp():
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        yield app

except ImportError:
    pass


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--regenerate-api-baseline",
        action="store_true",
        default=False,
        help=(
            "Regenerate the public-API stability baselines "
            "(tests/sidekick_api_baseline.json and tests/api_baselines/*.json)"
        ),
    )
