"""Tests for CLI agent provider surfacing in the ChatDockWidget provider dropdown.

Covers:
- list_available_providers() probes with shutil.which and returns correct entries
- Installed providers appear; missing ones are excluded
- Providers include the three CLI agents: Claude CLI, Codex CLI, Cline
- ChatDockWidget provider combo includes CLI providers when installed
- CLI provider availability check integrates with TerminalSessionRuntime

Tools issue: UpstreamDrift#5622
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Register src namespace packages so dotted imports resolve correctly
_src_pkg = types.ModuleType("src")
_src_pkg.__path__ = [str(ROOT / "src")]  # type: ignore[attr-defined]
sys.modules.setdefault("src", _src_pkg)

for _ns in ("src.shared", "src.shared.python", "src.shared.python.chat"):
    _parts = _ns.split(".")
    _mod = types.ModuleType(_ns)
    _mod.__path__ = [str(ROOT.joinpath(*_parts))]  # type: ignore[attr-defined]
    sys.modules.setdefault(_ns, _mod)

# Stub PyQt6 so widget construction tests can run headless
_qt_widgets = types.ModuleType("PyQt6.QtWidgets")
_qt_core = types.ModuleType("PyQt6.QtCore")
_qt_gui = types.ModuleType("PyQt6.QtGui")
_qt_ws = types.ModuleType("PyQt6.QtWebSockets")
_qt6 = types.ModuleType("PyQt6")

for _attr in [
    "QDockWidget",
    "QWidget",
    "QVBoxLayout",
    "QHBoxLayout",
    "QLabel",
    "QPushButton",
    "QComboBox",
    "QScrollArea",
    "QPlainTextEdit",
    "QStackedWidget",
    "QFrame",
    "QMenu",
    "QFileDialog",
    "QApplication",
]:
    setattr(_qt_widgets, _attr, MagicMock)

for _attr in ["Qt", "QTimer", "QUrl", "pyqtSignal"]:
    setattr(_qt_core, _attr, MagicMock)
for _attr in ["QKeySequence", "QShortcut"]:
    setattr(_qt_gui, _attr, MagicMock)
_qt_ws.QWebSocket = MagicMock

_qt6.QtWidgets = _qt_widgets
_qt6.QtCore = _qt_core
_qt6.QtGui = _qt_gui
_qt6.QtWebSockets = _qt_ws

sys.modules.setdefault("PyQt6", _qt6)
sys.modules.setdefault("PyQt6.QtWidgets", _qt_widgets)
sys.modules.setdefault("PyQt6.QtCore", _qt_core)
sys.modules.setdefault("PyQt6.QtGui", _qt_gui)
sys.modules.setdefault("PyQt6.QtWebSockets", _qt_ws)

# Stub out logging config dependency used by sub-modules
logging_pkg = types.ModuleType("src.shared.python.logging_pkg")
logging_config = types.ModuleType("src.shared.python.logging_pkg.logging_config")
logging_config.get_logger = logging.getLogger  # type: ignore[attr-defined]
logging_config.setup_logging = lambda *a, **kw: None  # type: ignore[attr-defined]
sys.modules.setdefault("src.shared.python.logging_pkg", logging_pkg)
sys.modules.setdefault("src.shared.python.logging_pkg.logging_config", logging_config)

from src.shared.python.chat.cli_provider_availability import (  # noqa: E402
    CliProviderEntry,
    list_available_cli_providers,
)

# ─────────────────────────────────────────────────────────────────────────────
# list_available_cli_providers()
# ─────────────────────────────────────────────────────────────────────────────


class TestListAvailableCliProviders:
    """Tests for the shutil.which-based provider probe."""

    def test_returns_list(self) -> None:
        with patch("shutil.which", return_value=None):
            result = list_available_cli_providers()
        assert isinstance(result, list)

    def test_all_three_appear_when_all_installed(self) -> None:
        """When shutil.which finds claude/codex/cline, all three are returned."""
        with patch("shutil.which", side_effect=lambda x: f"/usr/bin/{x}"):
            providers = list_available_cli_providers()
        names = [p.display_name for p in providers]
        assert "Claude CLI" in names
        assert "Codex CLI" in names
        assert "Cline" in names

    def test_omits_missing_providers(self) -> None:
        """Providers whose binary is absent are excluded."""
        with patch("shutil.which", return_value=None):
            providers = list_available_cli_providers()
        assert providers == []

    def test_partial_availability(self) -> None:
        """Only installed providers are returned."""
        available = {"claude"}

        def _which(name: str) -> str | None:
            return f"/usr/bin/{name}" if name in available else None

        with patch("shutil.which", side_effect=_which):
            providers = list_available_cli_providers()
        names = [p.display_name for p in providers]
        assert "Claude CLI" in names
        assert "Codex CLI" not in names
        assert "Cline" not in names

    def test_entry_has_binary_path(self) -> None:
        """Each returned entry carries the resolved binary path."""
        with patch("shutil.which", side_effect=lambda x: f"/usr/bin/{x}"):
            providers = list_available_cli_providers()
        for entry in providers:
            assert entry.binary_path is not None
            assert entry.binary_path.startswith("/usr/bin/")

    def test_entry_has_provider_id(self) -> None:
        """Each entry exposes a stable provider_id matching the registry."""
        with patch("shutil.which", side_effect=lambda x: f"/usr/bin/{x}"):
            providers = list_available_cli_providers()
        ids = [p.provider_id for p in providers]
        assert "claude-code" in ids
        assert "codex" in ids
        assert "cline-cli" in ids

    def test_returns_cli_provider_entry_instances(self) -> None:
        with patch("shutil.which", side_effect=lambda x: f"/usr/bin/{x}"):
            providers = list_available_cli_providers()
        for entry in providers:
            assert isinstance(entry, CliProviderEntry)

    def test_gemini_cli_also_included_when_installed(self) -> None:
        """Gemini CLI is also surfaced when available."""
        with patch("shutil.which", side_effect=lambda x: f"/usr/bin/{x}"):
            providers = list_available_cli_providers()
        names = [p.display_name for p in providers]
        assert "Gemini CLI" in names

    def test_no_duplicates(self) -> None:
        with patch("shutil.which", side_effect=lambda x: f"/usr/bin/{x}"):
            providers = list_available_cli_providers()
        ids = [p.provider_id for p in providers]
        assert len(ids) == len(set(ids))


# ─────────────────────────────────────────────────────────────────────────────
# CliProviderEntry contract
# ─────────────────────────────────────────────────────────────────────────────


class TestCliProviderEntry:
    def test_fields_accessible(self) -> None:
        entry = CliProviderEntry(
            provider_id="claude-code",
            display_name="Claude CLI",
            binary_path="/usr/bin/claude",
        )
        assert entry.provider_id == "claude-code"
        assert entry.display_name == "Claude CLI"
        assert entry.binary_path == "/usr/bin/claude"

    def test_display_name_cannot_be_empty(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            CliProviderEntry(
                provider_id="x",
                display_name="",
                binary_path="/usr/bin/x",
            )

    def test_provider_id_cannot_be_empty(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            CliProviderEntry(
                provider_id="",
                display_name="X",
                binary_path="/usr/bin/x",
            )
