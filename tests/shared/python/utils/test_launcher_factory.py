"""Tests for upstream_drift_tools.launcher_factory module.

Covers:
- LauncherConfig dataclass creation
- create_launcher_config factory
- validate_launcher_config preconditions
- LauncherError exception
- Frozen dataclass immutability
"""

from __future__ import annotations

import pytest
from upstream_drift_tools.launcher_factory import (
    LauncherConfig,
    LauncherError,
    create_launcher_config,
    validate_launcher_config,
)

# ── LauncherConfig ───────────────────────────────────────────────────────


class TestLauncherConfig:
    """Test LauncherConfig dataclass."""

    def test_construction(self) -> None:
        config = LauncherConfig(
            app_module="my_tool.main",
            window_title="My Tool",
        )
        assert config.app_module == "my_tool.main"
        assert config.window_title == "My Tool"

    def test_defaults(self) -> None:
        config = LauncherConfig(
            app_module="test",
            window_title="Test",
        )
        assert config.min_width == 800
        assert config.min_height == 600
        assert config.icon_path is None
        assert config.extra == {}

    def test_custom_dimensions(self) -> None:
        config = LauncherConfig(
            app_module="a",
            window_title="b",
            min_width=1024,
            min_height=768,
        )
        assert config.min_width == 1024
        assert config.min_height == 768

    def test_frozen(self) -> None:
        config = LauncherConfig(
            app_module="a",
            window_title="b",
        )
        with pytest.raises(AttributeError):
            config.window_title = "changed"  # type: ignore[misc]

    def test_icon_path(self) -> None:
        config = LauncherConfig(
            app_module="a",
            window_title="b",
            icon_path="/path/to/icon.png",
        )
        assert config.icon_path == "/path/to/icon.png"


# ── create_launcher_config ───────────────────────────────────────────────


class TestCreateLauncherConfig:
    """Test create_launcher_config factory function."""

    def test_basic_creation(self) -> None:
        config = create_launcher_config(
            app_module="my_tool.main",
            window_title="My Tool",
        )
        assert isinstance(config, LauncherConfig)
        assert config.app_module == "my_tool.main"

    def test_extra_kwargs(self) -> None:
        config = create_launcher_config(
            app_module="a",
            window_title="b",
            debug=True,
            port=8080,
        )
        assert config.extra["debug"] is True
        assert config.extra["port"] == 8080

    def test_all_parameters(self) -> None:
        config = create_launcher_config(
            app_module="pkg.module",
            window_title="Full Test",
            min_width=1280,
            min_height=720,
            icon_path="/icon.svg",
            custom_option="value",
        )
        assert config.min_width == 1280
        assert config.icon_path == "/icon.svg"
        assert config.extra["custom_option"] == "value"


# ── validate_launcher_config ─────────────────────────────────────────────


class TestValidateLauncherConfig:
    """Test validation preconditions."""

    def test_valid_config_passes(self) -> None:
        config = LauncherConfig(
            app_module="valid.module",
            window_title="Valid Title",
        )
        validate_launcher_config(config)  # Should not raise

    def test_empty_app_module_rejected(self) -> None:
        config = LauncherConfig(
            app_module="",
            window_title="Title",
        )
        with pytest.raises(LauncherError, match="app_module"):
            validate_launcher_config(config)

    def test_whitespace_app_module_rejected(self) -> None:
        config = LauncherConfig(
            app_module="   ",
            window_title="Title",
        )
        with pytest.raises(LauncherError, match="app_module"):
            validate_launcher_config(config)

    def test_empty_window_title_rejected(self) -> None:
        config = LauncherConfig(
            app_module="mod",
            window_title="",
        )
        with pytest.raises(LauncherError, match="window_title"):
            validate_launcher_config(config)

    def test_whitespace_title_rejected(self) -> None:
        config = LauncherConfig(
            app_module="mod",
            window_title="   ",
        )
        with pytest.raises(LauncherError, match="window_title"):
            validate_launcher_config(config)

    def test_negative_width_rejected(self) -> None:
        config = LauncherConfig(
            app_module="mod",
            window_title="title",
            min_width=-1,
        )
        with pytest.raises(LauncherError, match="min_width"):
            validate_launcher_config(config)

    def test_negative_height_rejected(self) -> None:
        config = LauncherConfig(
            app_module="mod",
            window_title="title",
            min_height=-100,
        )
        with pytest.raises(LauncherError, match="min_height"):
            validate_launcher_config(config)

    def test_zero_dimensions_valid(self) -> None:
        config = LauncherConfig(
            app_module="mod",
            window_title="title",
            min_width=0,
            min_height=0,
        )
        validate_launcher_config(config)  # Should not raise


# ── LauncherError ────────────────────────────────────────────────────────


class TestLauncherError:
    """Test LauncherError exception."""

    def test_is_exception(self) -> None:
        assert issubclass(LauncherError, Exception)

    def test_message(self) -> None:
        err = LauncherError("test message")
        assert str(err) == "test message"
