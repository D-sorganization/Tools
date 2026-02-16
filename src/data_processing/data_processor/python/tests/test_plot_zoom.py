"""Tests for data_processor.core.plot_zoom module."""

from __future__ import annotations

from data_processor.core.plot_zoom import (
    InteractivePlotManager,
    MouseWheelZoom,
    ZoomConfig,
)


class TestZoomConfig:
    """Tests for ZoomConfig dataclass."""

    def test_defaults(self) -> None:
        config = ZoomConfig()
        assert config.zoom_in_factor == 1.2
        assert config.zoom_out_factor == 0.8
        assert config.center_on_cursor is True
        assert config.smooth_animation is False
        assert config.animation_duration_ms == 100
        assert config.maintain_aspect_ratio is False
        assert config.allow_horizontal_zoom is True
        assert config.allow_vertical_zoom is True
        assert config.min_zoom_range == 1e-10
        assert config.max_zoom_range == 1e10

    def test_custom_values(self) -> None:
        config = ZoomConfig(
            zoom_in_factor=1.5,
            zoom_out_factor=0.5,
            center_on_cursor=False,
            maintain_aspect_ratio=True,
        )
        assert config.zoom_in_factor == 1.5
        assert config.zoom_out_factor == 0.5
        assert config.center_on_cursor is False
        assert config.maintain_aspect_ratio is True

    def test_zoom_factors_inverse(self) -> None:
        """Default zoom factors should be approximate inverses."""
        config = ZoomConfig()
        product = config.zoom_in_factor * config.zoom_out_factor
        # 1.2 * 0.8 = 0.96, close to 1.0
        assert abs(product - 1.0) < 0.1


class TestMouseWheelZoom:
    """Tests for MouseWheelZoom class."""

    def test_default_construction(self) -> None:
        zoom = MouseWheelZoom()
        assert zoom is not None

    def test_custom_config(self) -> None:
        config = ZoomConfig(zoom_in_factor=2.0)
        zoom = MouseWheelZoom(config=config)
        assert zoom is not None

    def test_add_callback(self) -> None:
        zoom = MouseWheelZoom()
        events: list[str] = []
        zoom.add_zoom_callback(lambda e: events.append("zoom"))
        # Callbacks are stored but not triggered without an actual event
        assert zoom is not None

    def test_remove_callback(self) -> None:
        zoom = MouseWheelZoom()
        cb = lambda e: None  # noqa: E731
        zoom.add_zoom_callback(cb)
        zoom.remove_zoom_callback(cb)
        assert zoom is not None


class TestInteractivePlotManager:
    """Tests for InteractivePlotManager class."""

    def test_construction(self) -> None:
        manager = InteractivePlotManager()
        assert manager is not None

    def test_zoom_handler_property(self) -> None:
        manager = InteractivePlotManager()
        handler = manager.zoom_handler
        assert isinstance(handler, MouseWheelZoom)

    def test_reset_all_zoom_no_figures(self) -> None:
        """Reset should not raise when no figures are managed."""
        manager = InteractivePlotManager()
        manager.reset_all_zoom()  # Should not raise
