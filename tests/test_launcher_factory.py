"""Tests for the shared launcher factory (TDD — tests written first).

Verifies:
- LauncherConfig validation contracts
- launch_app orchestration logic
- Error handling for missing PyQt6
- Logging integration

Addresses #763 (Phase 2: DRY consolidation).
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
from upstream_drift_tools.launcher_factory import (
    LauncherConfig,
    LauncherError,
    create_launcher_config,
    launch_app,
    validate_launcher_config,
)


class TestLauncherConfig:
    """Test LauncherConfig creation and validation."""

    def test_create_config_with_defaults(self) -> None:
        """Config with just app_module and window_title uses sensible defaults."""
        config = create_launcher_config(
            app_module="my_app.main",
            window_title="My Application",
        )
        assert config.app_module == "my_app.main"
        assert config.window_title == "My Application"
        assert config.min_width == 800
        assert config.min_height == 600
        assert config.icon_path is None

    def test_create_config_with_custom_size(self) -> None:
        """Config accepts custom minimum window size."""
        config = create_launcher_config(
            app_module="my_app.main",
            window_title="My App",
            min_width=1200,
            min_height=900,
        )
        assert config.min_width == 1200
        assert config.min_height == 900

    def test_create_config_with_icon(self) -> None:
        """Config accepts an icon path."""
        config = create_launcher_config(
            app_module="my_app.main",
            window_title="My App",
            icon_path="assets/icon.png",
        )
        assert config.icon_path == "assets/icon.png"


class TestValidateLauncherConfig:
    """Test launch configuration validation (DbC preconditions)."""

    def test_empty_app_module_rejected(self) -> None:
        """Empty app_module violates precondition."""
        config = LauncherConfig(
            app_module="",
            window_title="Test",
            min_width=800,
            min_height=600,
        )
        with pytest.raises(LauncherError, match="app_module"):
            validate_launcher_config(config)

    def test_empty_window_title_rejected(self) -> None:
        """Empty window_title violates precondition."""
        config = LauncherConfig(
            app_module="my_app",
            window_title="",
            min_width=800,
            min_height=600,
        )
        with pytest.raises(LauncherError, match="window_title"):
            validate_launcher_config(config)

    def test_negative_width_rejected(self) -> None:
        """Negative min_width violates precondition."""
        config = LauncherConfig(
            app_module="my_app",
            window_title="Test",
            min_width=-1,
            min_height=600,
        )
        with pytest.raises(LauncherError, match="min_width"):
            validate_launcher_config(config)

    def test_negative_height_rejected(self) -> None:
        """Negative min_height violates precondition."""
        config = LauncherConfig(
            app_module="my_app",
            window_title="Test",
            min_width=800,
            min_height=-1,
        )
        with pytest.raises(LauncherError, match="min_height"):
            validate_launcher_config(config)

    def test_valid_config_passes(self) -> None:
        """Valid config must not raise."""
        config = LauncherConfig(
            app_module="my_app",
            window_title="Test App",
            min_width=800,
            min_height=600,
        )
        validate_launcher_config(config)  # Should not raise


class TestLaunchApp:
    """Test the launch_app orchestration function."""

    @patch("upstream_drift_tools.launcher_factory._import_pyqt6")
    def test_launch_returns_exit_code(self, mock_import: MagicMock) -> None:
        """launch_app should return the application exit code."""
        mock_app = MagicMock()
        mock_app.exec.return_value = 0
        mock_import.return_value = (mock_app, MagicMock)

        config = create_launcher_config(
            app_module="test.app",
            window_title="Test",
        )
        exit_code = launch_app(config, window_factory=lambda: MagicMock())
        assert exit_code == 0

    @patch("upstream_drift_tools.launcher_factory._import_pyqt6")
    def test_launch_sets_window_title(self, mock_import: MagicMock) -> None:
        """launch_app should set the window title from config."""
        mock_app = MagicMock()
        mock_app.exec.return_value = 0
        mock_import.return_value = (mock_app, MagicMock)

        mock_window = MagicMock()
        config = create_launcher_config(
            app_module="test.app",
            window_title="Expected Title",
        )
        launch_app(config, window_factory=lambda: mock_window)
        mock_window.setWindowTitle.assert_called_once_with("Expected Title")

    @patch("upstream_drift_tools.launcher_factory._import_pyqt6")
    def test_launch_sets_minimum_size(self, mock_import: MagicMock) -> None:
        """launch_app should set minimum window size from config."""
        mock_app = MagicMock()
        mock_app.exec.return_value = 0
        mock_import.return_value = (mock_app, MagicMock)

        mock_window = MagicMock()
        config = create_launcher_config(
            app_module="test.app",
            window_title="Test",
            min_width=1024,
            min_height=768,
        )
        launch_app(config, window_factory=lambda: mock_window)
        mock_window.setMinimumSize.assert_called_once_with(1024, 768)

    @patch("upstream_drift_tools.launcher_factory._import_pyqt6")
    def test_launch_shows_window(self, mock_import: MagicMock) -> None:
        """launch_app should call window.show()."""
        mock_app = MagicMock()
        mock_app.exec.return_value = 0
        mock_import.return_value = (mock_app, MagicMock)

        mock_window = MagicMock()
        config = create_launcher_config(
            app_module="test.app",
            window_title="Test",
        )
        launch_app(config, window_factory=lambda: mock_window)
        mock_window.show.assert_called_once()

    def test_launch_handles_missing_pyqt6(self) -> None:
        """launch_app should return 1 when PyQt6 is not installed."""
        config = create_launcher_config(
            app_module="test.app",
            window_title="Test",
        )
        with patch(
            "upstream_drift_tools.launcher_factory._import_pyqt6",
            side_effect=ImportError("No module named 'PyQt6'"),
        ):
            exit_code = launch_app(config, window_factory=lambda: MagicMock())
            assert exit_code == 1

    def test_launch_invalid_config_raises(self) -> None:
        """launch_app should raise LauncherError for invalid config."""
        config = LauncherConfig(
            app_module="",
            window_title="Test",
            min_width=800,
            min_height=600,
        )
        with pytest.raises(LauncherError):
            launch_app(config, window_factory=lambda: MagicMock())


class TestLaunchAppLogging:
    """Test that launch_app integrates with logging."""

    @patch("upstream_drift_tools.launcher_factory._import_pyqt6")
    def test_launch_logs_startup(
        self, mock_import: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """launch_app should log the application startup."""
        mock_app = MagicMock()
        mock_app.exec.return_value = 0
        mock_import.return_value = (mock_app, MagicMock)

        config = create_launcher_config(
            app_module="test.app",
            window_title="My Test App",
        )
        with caplog.at_level(logging.INFO):
            launch_app(config, window_factory=lambda: MagicMock())

        assert any("My Test App" in record.message for record in caplog.records)
