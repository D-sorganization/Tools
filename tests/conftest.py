"""Shared pytest configuration for entire Tools repo."""

import importlib.util
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
    import logging

    for import_root in (
        repo_root / "src" / "shared" / "python",
        repo_root / "src",
        repo_root,
    ):
        import_path = str(import_root)
        if import_path in sys.path:
            sys.path.remove(import_path)
        sys.path.insert(0, import_path)

    def ensure_package_path(name: str, path: str) -> None:
        package_path = str(repo_root / path)
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            module.__path__ = [package_path]  # type: ignore[attr-defined]
            sys.modules[name] = module
        else:
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
        ("src", "src"),
        ("src.shared", "src/shared"),
        ("src.shared.python", "src/shared/python"),
        ("shared", "src/shared"),
        ("shared.python", "src/shared/python"),
        ("src.shared.python.calc_backend", "src/shared/python/calc_backend"),
        ("shared.python.calc_backend", "src/shared/python/calc_backend"),
        ("src.shared.python.config", "src/shared/python/config"),
        ("src.shared.python.logging_pkg", "src/shared/python/logging_pkg"),
    ]

    for name, path in checkout_packages:
        ensure_package_path(name, path)

    # Specifically stub logging_config
    if "src.shared.python.logging_pkg.logging_config" not in sys.modules:
        config_name = "src.shared.python.logging_pkg.logging_config"
        logging_config = types.ModuleType(config_name)
        logging_config.get_logger = logging.getLogger  # type: ignore
        logging_config.setup_logging = lambda *a, **kw: None  # type: ignore
        sys.modules[config_name] = logging_config

    # Specifically stub environment
    if "src.shared.python.config.environment" not in sys.modules:
        env = types.ModuleType("src.shared.python.config.environment")
        env.get_env = lambda key, default=None, required=False: default  # type: ignore
        env.get_env_float = lambda key, default=0.0: float(default)  # type: ignore
        sys.modules["src.shared.python.config.environment"] = env


def _preload_ai_exception_aliases(repo_root: Path) -> None:
    """Bind all supported AI exception import paths to the real module file."""
    module_names = (
        "src.shared.python.ai.exceptions",
        "shared.python.ai.exceptions",
        "ai.exceptions",
    )
    existing = next(
        (
            sys.modules[name]
            for name in module_names
            if hasattr(sys.modules.get(name), "AIConnectionError")
        ),
        None,
    )
    if existing is not None:
        for name in module_names:
            sys.modules[name] = existing
        return

    module_path = repo_root / "src" / "shared" / "python" / "ai" / "exceptions.py"
    spec = importlib.util.spec_from_file_location(module_names[0], module_path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    for name in module_names:
        sys.modules[name] = module
    spec.loader.exec_module(module)
    for name in module_names:
        sys.modules[name] = module


def _preload_ai_type_aliases(repo_root: Path) -> None:
    """Bind all supported AI type import paths to the real module file."""
    module_names = (
        "src.shared.python.ai.types",
        "shared.python.ai.types",
        "ai.types",
    )
    existing = next(
        (
            sys.modules[name]
            for name in module_names
            if hasattr(sys.modules.get(name), "ConversationContext")
        ),
        None,
    )
    if existing is not None:
        for name in module_names:
            sys.modules[name] = existing
        return

    module_path = repo_root / "src" / "shared" / "python" / "ai" / "types.py"
    spec = importlib.util.spec_from_file_location(module_names[0], module_path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    for name in module_names:
        sys.modules[name] = module
    spec.loader.exec_module(module)
    for name in module_names:
        sys.modules[name] = module


_setup_global_stubs(REPO_ROOT)
_preload_ai_exception_aliases(REPO_ROOT)
_preload_ai_type_aliases(REPO_ROOT)


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
        help="Regenerate Sidekick API stability baseline json file",
    )
