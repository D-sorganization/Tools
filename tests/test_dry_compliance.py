"""Tests for DRY compliance: shared modules, launcher deduplication, and path utilities.

Verifies:
- All launch_web.py files use the shared launch_web_from_gui_info pattern
- All gui_registration.py files with a sibling launch_web.py define a "web" config
- launch_web_app and launch_web_from_gui_info are importable and well-formed
- get_repo_root implementations agree across modules
- launch_pyqt6.py files follow the thin-wrapper pattern
"""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── Path constants ────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"

# PSA package uses Streamlit, not the React/npm pattern
_NON_REACT_WEB_LAUNCHERS = {"psa_package"}


# ── Helper ────────────────────────────────────────────────────────────────


def _load_module_from_path(name: str, path: Path) -> types.ModuleType | None:
    """Dynamically load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return None


def _collect_launch_web_files() -> list[Path]:
    """Find all launch_web.py files under src/."""
    return sorted(SRC_DIR.rglob("launch_web.py"))


def _collect_launch_pyqt6_files() -> list[Path]:
    """Find all launch_pyqt6.py files under src/."""
    return sorted(SRC_DIR.rglob("launch_pyqt6.py"))


def _collect_gui_registrations() -> list[Path]:
    """Find all gui_registration.py files under src/."""
    return sorted(SRC_DIR.rglob("gui_registration.py"))


# ── Web launcher DRY tests ───────────────────────────────────────────────


class TestWebLauncherDRY:
    """All React launch_web.py files should use the shared pattern."""

    def test_launch_web_files_use_shared_function(self) -> None:
        """Every React launch_web.py should import launch_web_from_gui_info."""
        files = _collect_launch_web_files()
        assert len(files) > 0, "No launch_web.py files found"

        problems = []
        for path in files:
            # Skip non-React launchers (e.g. Streamlit)
            tool_dir = path.parent.name
            if tool_dir in _NON_REACT_WEB_LAUNCHERS:
                continue

            content = path.read_text(encoding="utf-8")
            if "launch_web_from_gui_info" not in content:
                rel = str(path.relative_to(REPO_ROOT))
                problems.append(rel)

        assert problems == [], (
            "launch_web.py files not using shared launch_web_from_gui_info:\n"
            + "\n".join(f"  - {p}" for p in problems)
        )

    def test_launch_web_files_import_gui_info(self) -> None:
        """Every React launch_web.py should import GUI_INFO from gui_registration."""
        files = _collect_launch_web_files()

        problems = []
        for path in files:
            tool_dir = path.parent.name
            if tool_dir in _NON_REACT_WEB_LAUNCHERS:
                continue

            content = path.read_text(encoding="utf-8")
            if "GUI_INFO" not in content:
                rel = str(path.relative_to(REPO_ROOT))
                problems.append(rel)

        msg = "launch_web.py files not importing GUI_INFO:\n" + "\n".join(
            f"  - {p}" for p in problems
        )
        assert problems == [], msg


class TestGUIInfoWebConfig:
    """gui_registration.py files with launch_web.py siblings should have web config."""

    def test_web_config_exists_when_launch_web_exists(self) -> None:
        """If a tool has launch_web.py, its GUI_INFO should have a 'web' key."""
        launch_web_files = _collect_launch_web_files()
        gui_reg_files = _collect_gui_registrations()

        # Build set of directories that have launch_web.py
        web_dirs = {f.parent for f in launch_web_files}

        # Build map of directories that have gui_registration.py
        reg_by_dir = {f.parent: f for f in gui_reg_files}

        problems = []
        for web_dir in web_dirs:
            tool_dir = web_dir.name
            if tool_dir in _NON_REACT_WEB_LAUNCHERS:
                continue

            reg_file = reg_by_dir.get(web_dir)
            if reg_file is None:
                continue  # No gui_registration.py at all -- different issue

            try:
                module = _load_module_from_path(f"gui_reg_{tool_dir}", reg_file)
            except Exception:  # noqa: BLE001
                continue

            if module is None:
                continue

            gui_info = getattr(module, "GUI_INFO", None)
            if gui_info is None:
                continue

            if "web" not in gui_info:
                rel = str(reg_file.relative_to(REPO_ROOT))
                problems.append(f"{rel}: missing 'web' config")

        assert problems == [], (
            "gui_registration.py files with sibling launch_web.py but no web config:\n"
            + "\n".join(f"  - {p}" for p in problems)
        )

    def test_web_config_has_port(self) -> None:
        """Every web config should specify a port."""
        gui_reg_files = _collect_gui_registrations()

        problems = []
        for path in gui_reg_files:
            try:
                module = _load_module_from_path(f"gui_reg_{path.stem}", path)
            except Exception:  # noqa: BLE001
                continue

            if module is None:
                continue

            gui_info = getattr(module, "GUI_INFO", None)
            if gui_info is None or "web" not in gui_info:
                continue

            web = gui_info["web"]
            if not isinstance(web, dict):
                continue
            if "port" not in web:
                rel = str(path.relative_to(REPO_ROOT))
                problems.append(f"{rel}: web config missing 'port'")

        assert problems == [], "web config problems:\n" + "\n".join(
            f"  - {p}" for p in problems
        )


# ── Shared launcher function tests ──────────────────────────────────────


class TestLaunchWebApp:
    """launch_web_app should be importable and well-formed."""

    def test_importable(self) -> None:
        from gui_launcher import launch_web_app

        assert callable(launch_web_app)

    def test_launch_web_from_gui_info_importable(self) -> None:
        from gui_launcher import launch_web_from_gui_info

        assert callable(launch_web_from_gui_info)

    def test_returns_error_for_missing_web_dir(self, tmp_path: Path) -> None:
        from gui_launcher.launcher import launch_web_app

        result = launch_web_app(
            tool_name="test",
            web_dir=tmp_path / "nonexistent",
            port=9999,
            auto_open_browser=False,
        )
        assert result == 1

    @patch("gui_launcher.launcher.subprocess.run")
    def test_returns_error_when_node_missing(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        from gui_launcher.launcher import launch_web_app

        mock_run.side_effect = FileNotFoundError("node not found")
        result = launch_web_app(
            tool_name="test",
            web_dir=tmp_path,
            port=9999,
            auto_open_browser=False,
        )
        assert result == 1

    def test_launch_web_from_gui_info_returns_error_without_web_config(
        self,
    ) -> None:
        """If GUI_INFO has no web config, should still run with defaults."""
        import tempfile

        from gui_launcher.launcher import launch_web_from_gui_info

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            caller_file = f.name

        # No web dir exists, so it should error
        gui_info = {"name": "Test", "tool_name": "test"}
        result = launch_web_from_gui_info(gui_info, caller_file)
        assert result == 1


# ── PyQt6 launcher DRY tests ────────────────────────────────────────────


class TestPyQt6LauncherDRY:
    """All launch_pyqt6.py files should follow the thin-wrapper pattern."""

    # Launchers with custom logic that do not follow the standard pattern
    _CUSTOM_LAUNCHERS = {
        "folder_packer_pro",  # calls main() directly
        "folder_tool",  # cross-platform script launcher
        "pdf_renamer",  # custom logging setup
        "glass_bath_fea",  # custom main() with error handling
        "lower_body_model",  # custom MuJoCo viewer/control panel
        "movement_optimizer",  # launches via movement_optimizer.__main__
        "optimizer_gui",  # compatibility shim delegating to movement_optimizer
        "rate_of_closure",  # custom launcher with Morris authority lifecycle
    }

    def test_launch_pyqt6_files_use_factory_or_bootstrap(self) -> None:
        """Standard launch_pyqt6.py files should use a shared launcher factory."""
        files = _collect_launch_pyqt6_files()
        assert len(files) > 0, "No launch_pyqt6.py files found"

        problems = []
        for path in files:
            # Skip known custom launchers
            if path.parent.name in self._CUSTOM_LAUNCHERS:
                continue

            content = path.read_text(encoding="utf-8")
            if "make_pyqt6_launcher" not in content and "make_launcher" not in content:
                rel = str(path.relative_to(REPO_ROOT))
                problems.append(rel)

        msg = (
            "launch_pyqt6.py files not using a shared launcher factory:\n"
            + "\n".join(f"  - {p}" for p in problems)
        )
        assert problems == [], msg

    def test_make_pyqt6_launcher_importable(self) -> None:
        """make_pyqt6_launcher should be importable from gui_launcher."""
        from gui_launcher import make_pyqt6_launcher

        assert callable(make_pyqt6_launcher)


# ── get_repo_root consistency ────────────────────────────────────────────


class TestGetRepoRootConsistency:
    """All get_repo_root implementations should agree."""

    def test_canonical_implementation_exists(self) -> None:
        from upstream_drift_tools.utils.paths import get_repo_root

        result = get_repo_root()
        assert result.is_absolute()
        assert (result / ".git").exists() or (result / "pyproject.toml").exists()

    def test_launch_utils_delegates_to_canonical(self) -> None:
        from upstream_drift_tools.utils.paths import get_repo_root as canonical

        from tools.launch_utils import get_repo_root as launch_get_root

        assert launch_get_root() == canonical()

    def test_path_setup_delegates_to_canonical(self) -> None:
        from upstream_drift_tools.utils.paths import get_repo_root as canonical
        from utils.path_setup import get_repo_root as setup_get_root

        assert setup_get_root() == canonical()
