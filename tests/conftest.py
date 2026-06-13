"""Shared pytest configuration for entire Tools repo."""

import importlib.util
import sys
import types
import warnings
from collections.abc import Sequence
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any

import pytest


class AliasLoader(Loader):
    """A loader that simply returns an existing module object."""

    def __init__(self, module: types.ModuleType) -> None:
        self.module = module

    def create_module(self, spec: ModuleSpec) -> types.ModuleType:
        return self.module

    def exec_module(self, module: types.ModuleType) -> None:
        pass


class WrappedCanonicalLoader(Loader):
    """A loader that delegates to the canonical loader and sets aliases."""

    def __init__(self, canonical_spec: ModuleSpec, aliases: list[str]) -> None:
        """Initialize with the canonical spec and its aliases."""
        self.canonical_spec = canonical_spec
        self.aliases = aliases

    def create_module(self, spec: ModuleSpec) -> types.ModuleType | None:
        """Create the module using the canonical spec's loader if available."""
        if self.canonical_spec.name in sys.modules:
            return sys.modules[self.canonical_spec.name]

        if hasattr(self.canonical_spec.loader, "create_module"):
            module = self.canonical_spec.loader.create_module(self.canonical_spec)
        else:
            module = None

        if module is None:
            module = types.ModuleType(self.canonical_spec.name)
            module.__spec__ = self.canonical_spec
            if self.canonical_spec.submodule_search_locations is not None:
                module.__path__ = self.canonical_spec.submodule_search_locations

        return module

    def exec_module(self, module: types.ModuleType) -> None:
        """Execute the module and register it under all its aliases in sys.modules."""
        for alias in self.aliases:
            sys.modules[alias] = module
        if self.canonical_spec.loader is not None:
            self.canonical_spec.loader.exec_module(module)
        for alias in self.aliases:
            sys.modules[alias] = module


class RobustImportRedirector(MetaPathFinder):
    """Import redirector that ensures shared modules are mapped to the same object.

    This prevents duplicate module loading when modules in src/shared/python
    are imported via different import paths (e.g. 'ai.types' vs
    'src.shared.python.ai.types').
    """

    def __init__(self) -> None:
        """Initialize the redirector with the set of top-level shared packages."""
        self.top_level_packages: set[str] = {
            "ai",
            "theme",
            "contracts",
            "cors",
            "deprecation",
            "safe_eval",
            "notes",
            "sidekick",
            "signal_toolkit",
            "upstream_drift_tools",
        }

    def _parse_fullname(self, parts: list[str]) -> tuple[str | None, list[str] | None]:
        """Parse parts to find top level package and remainder."""
        if (
            len(parts) >= 4
            and parts[:3] == ["src", "shared", "python"]
            and parts[3] in self.top_level_packages
        ):
            return parts[3], parts[4:]
        if (
            len(parts) >= 3
            and parts[:2] == ["shared", "python"]
            and parts[2] in self.top_level_packages
        ):
            return parts[2], parts[3:]
        if parts[0] in self.top_level_packages:
            return parts[0], parts[1:]
        return None, None

    def get_aliases(self, fullname: str) -> list[str]:
        """Get all alternative import names for the given fullname."""
        parts = fullname.split(".")
        if not parts:
            return []
        pkg, remainder = self._parse_fullname(parts)
        if pkg is None:
            return []
        suffix = "." + ".".join(remainder) if remainder else ""
        if pkg in ("sidekick", "upstream_drift_tools"):
            return [
                f"src.shared.python.sidekick{suffix}",
                f"shared.python.sidekick{suffix}",
                f"sidekick{suffix}",
                f"src.shared.python.upstream_drift_tools{suffix}",
                f"shared.python.upstream_drift_tools{suffix}",
                f"upstream_drift_tools{suffix}",
            ]
        return [
            f"src.shared.python.{pkg}{suffix}",
            f"shared.python.{pkg}{suffix}",
            f"{pkg}{suffix}",
        ]

    def _redirect_spec(
        self, fullname: str, loader: Any, canonical_spec: ModuleSpec
    ) -> ModuleSpec:
        """Create a redirected ModuleSpec from canonical."""
        spec = ModuleSpec(
            fullname,
            loader,
            origin=canonical_spec.origin,
            loader_state=canonical_spec.loader_state,
        )
        spec.submodule_search_locations = canonical_spec.submodule_search_locations
        spec.cached = canonical_spec.cached
        spec.has_location = canonical_spec.has_location
        return spec

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: types.ModuleType | None = None,
    ) -> ModuleSpec | None:
        """Find the module spec, redirecting to existing aliases if loaded."""
        aliases = self.get_aliases(fullname)
        if not aliases:
            return None

        # Emit deprecation warning if trying to import via the deprecated package name
        is_deprecated = fullname == "upstream_drift_tools" or fullname.startswith(
            "upstream_drift_tools."
        )
        if is_deprecated:
            warnings.warn(
                "upstream_drift_tools is deprecated and will be removed "
                "in a future release. Import from sidekick instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        # 1. Check if any alias is already loaded in sys.modules
        for alias in aliases:
            if alias in sys.modules and sys.modules[alias] is not None:
                module = sys.modules[alias]
                for a in aliases:
                    if a not in sys.modules or sys.modules[a] is None:
                        sys.modules[a] = module
                canonical = self._find_canonical_spec(aliases)
                dummy_spec = canonical or ModuleSpec(fullname, None)
                return self._redirect_spec(fullname, AliasLoader(module), dummy_spec)

        # 2. Find which alias is importable by the default machinery
        canonical_spec = self._find_canonical_spec(aliases)
        if canonical_spec is None:
            return None

        # 3. Return a spec that loads the canonical module and sets aliases
        return self._redirect_spec(
            fullname, WrappedCanonicalLoader(canonical_spec, aliases), canonical_spec
        )

    def _find_canonical_spec(self, aliases: list[str]) -> ModuleSpec | None:
        """Find the canonical module spec from aliases using the default machinery."""
        canonical_spec = None
        removed = False
        if self in sys.meta_path:
            sys.meta_path.remove(self)
            removed = True
        try:
            for alias in aliases:
                try:
                    spec = importlib.util.find_spec(alias)
                    if spec is not None and spec.loader is not None:
                        canonical_spec = spec
                        break
                except Exception:
                    continue
        finally:
            if removed:
                sys.meta_path.insert(0, self)
        return canonical_spec


# Register the redirector at the beginning of the meta path
sys.meta_path.insert(0, RobustImportRedirector())

REPO_ROOT = Path(__file__).resolve().parent.parent


def _setup_global_stubs(repo_root: Path) -> None:
    """Setup global stubs for missing shared modules in the Tools repository.

    This ensures that when shared code imports config or logging_pkg (which
    are not physically present in Tools), they resolve to safe mock/stub modules.
    """
    import logging

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Define the missing package paths
    missing_packages = [
        ("src", "src"),
        ("src.shared", "src/shared"),
        ("src.shared.python", "src/shared/python"),
        ("src.shared.python.config", "src/shared/python/config"),
        ("src.shared.python.logging_pkg", "src/shared/python/logging_pkg"),
    ]

    for name, path in missing_packages:
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.__path__ = [str(repo_root / path)]
            sys.modules[name] = stub

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
