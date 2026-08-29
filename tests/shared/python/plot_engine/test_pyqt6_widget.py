from __future__ import annotations

from typing import Any

import pytest
from plot_engine import pyqt6_widget as widget_module
from plot_engine.specs import PlotSpec, SeriesData


class _ThemeSignal:
    def __init__(self) -> None:
        self.callbacks: list[Any] = []

    def connect(self, callback: Any) -> None:
        self.callbacks.append(callback)


class _SharedThemeManager:
    def __init__(self) -> None:
        self.themeChanged = _ThemeSignal()

    def get_current_colors(self) -> dict[str, str]:
        return {"background": "#ffffff", "text": "#111111"}

    def get_theme_colors(self, name: str) -> dict[str, str] | None:
        if name == "known":
            return {"background": "#000000", "text": "#eeeeee"}
        return None


class _PlotThemeManager:
    def __init__(self) -> None:
        self.callbacks: list[Any] = []

    def add_theme_change_callback(self, callback: Any) -> None:
        self.callbacks.append(callback)


class _Renderer:
    def __init__(self, theme_manager: _PlotThemeManager | None) -> None:
        self.theme_manager = theme_manager
        self.render_calls: list[tuple[PlotSpec, Any]] = []
        self.image_calls: list[tuple[PlotSpec, str, int]] = []

    def render(self, spec: PlotSpec, *, fig: Any) -> Any:
        self.render_calls.append((spec, fig))
        return fig

    def to_image(self, spec: PlotSpec, *, fmt: str, dpi: int) -> bytes:
        self.image_calls.append((spec, fmt, dpi))
        return f"{fmt}:{dpi}:{spec.title}".encode()


@pytest.fixture()
def widget_harness(monkeypatch: pytest.MonkeyPatch, qtbot: Any):
    shared_theme_manager = _SharedThemeManager()
    plot_theme_manager = _PlotThemeManager()
    renderers: list[_Renderer] = []
    applied_theme_colors: list[dict[str, str]] = []

    def renderer_factory(theme_manager: _PlotThemeManager | None = None) -> _Renderer:
        renderer = _Renderer(theme_manager)
        renderers.append(renderer)
        return renderer

    def apply_plot_theme(_figure: Any, colors: dict[str, str]) -> None:
        applied_theme_colors.append(colors)

    monkeypatch.setattr(widget_module, "MatplotlibRenderer", renderer_factory)
    monkeypatch.setattr(
        widget_module, "get_theme_manager", lambda: shared_theme_manager
    )
    monkeypatch.setattr(widget_module, "apply_plot_theme", apply_plot_theme)

    plot_widget = widget_module.PlotWidget(theme_manager=plot_theme_manager)
    qtbot.addWidget(plot_widget)

    return (
        plot_widget,
        renderers[0],
        plot_theme_manager,
        shared_theme_manager,
        applied_theme_colors,
    )


def _line_spec(title: str = "demo") -> PlotSpec:
    return PlotSpec(
        title=title,
        series=[SeriesData(name="feed", x=[0.0, 1.0], y=[1.0, 3.0])],
    )


def test_constructor_wires_theme_managers_and_controls(widget_harness: Any) -> None:
    widget, renderer, plot_theme_manager, shared_theme_manager, applied = widget_harness

    assert renderer.theme_manager is plot_theme_manager
    assert plot_theme_manager.callbacks == [widget._on_theme_changed]
    assert len(shared_theme_manager.themeChanged.callbacks) == 1
    assert [widget._format_combo.itemText(i) for i in range(3)] == ["PNG", "SVG", "PDF"]
    assert applied == [{"background": "#ffffff", "text": "#111111"}]

    shared_theme_manager.themeChanged.callbacks[0]("known")

    assert applied[-1] == {"background": "#000000", "text": "#eeeeee"}


def test_set_spec_renders_and_emits_signal(widget_harness: Any, qtbot: Any) -> None:
    widget, renderer, *_ = widget_harness
    spec = _line_spec()

    with qtbot.waitSignal(widget.spec_changed, timeout=1000):
        widget.set_spec(spec)

    assert widget.get_spec() is spec
    assert renderer.render_calls == [(spec, widget._figure)]


def test_set_spec_rejects_none(widget_harness: Any) -> None:
    widget, *_ = widget_harness

    with pytest.raises(ValueError, match="spec must be provided"):
        widget.set_spec(None)  # type: ignore[arg-type]


def test_refresh_and_theme_change_rerender_current_spec(widget_harness: Any) -> None:
    widget, renderer, *_ = widget_harness
    spec = _line_spec()

    widget.refresh()
    assert renderer.render_calls == []

    widget.set_spec(spec)
    widget.refresh()
    widget._on_theme_changed(object())

    assert renderer.render_calls == [
        (spec, widget._figure),
        (spec, widget._figure),
        (spec, widget._figure),
    ]


def test_export_plot_uses_selected_format_and_chosen_path(
    widget_harness: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    widget, *_ = widget_harness
    export_path = tmp_path / "plot.svg"
    saved: list[dict[str, Any]] = []

    monkeypatch.setattr(
        widget_module.QFileDialog,
        "getSaveFileName",
        lambda *args: (str(export_path), "SVG Files (*.svg)"),
    )
    monkeypatch.setattr(
        widget._figure,
        "savefig",
        lambda path, **kwargs: saved.append({"path": path, **kwargs}),
    )

    widget.set_spec(_line_spec())
    widget._format_combo.setCurrentText("SVG")
    widget._export_plot()

    assert len(saved) == 1
    call = saved[0]
    # The selected format, chosen path, and render settings are still honoured
    # verbatim; #4740 additionally routes the export through
    # ``plotting.export_figure``, which embeds provenance metadata.
    assert call["path"] == str(export_path)
    assert call["format"] == "svg"
    assert call["dpi"] == 150
    assert call["bbox_inches"] == "tight"
    assert call["transparent"] is False
    # Date is wall-clock, so assert its presence and shape rather than a value.
    assert call["metadata"]["Creator"] == "Tools"
    assert isinstance(call["metadata"]["Date"], str)


def test_export_plot_returns_without_spec_or_path(
    widget_harness: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    widget, *_ = widget_harness
    dialog_calls: list[str] = []
    save_calls: list[str] = []

    monkeypatch.setattr(
        widget_module.QFileDialog,
        "getSaveFileName",
        lambda *args: (dialog_calls.append("dialog"), ("", ""))[1],
    )
    monkeypatch.setattr(
        widget._figure,
        "savefig",
        lambda *args, **kwargs: save_calls.append("save"),
    )

    widget._export_plot()
    assert dialog_calls == []

    widget.set_spec(_line_spec())
    widget._export_plot()

    assert dialog_calls == ["dialog"]
    assert save_calls == []


def test_get_image_bytes_validates_format_and_delegates(widget_harness: Any) -> None:
    widget, renderer, *_ = widget_harness
    spec = _line_spec("image")

    assert widget.get_image_bytes() == b""

    with pytest.raises(ValueError, match="fmt must be provided"):
        widget.get_image_bytes(fmt=None)  # type: ignore[arg-type]

    widget.set_spec(spec)

    assert widget.get_image_bytes(fmt="pdf", dpi=96) == b"pdf:96:image"
    assert renderer.image_calls == [(spec, "pdf", 96)]
