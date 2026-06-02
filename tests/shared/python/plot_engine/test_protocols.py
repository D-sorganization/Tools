from __future__ import annotations

from typing import Any

from plot_engine.protocols import PlotConverter, PlotRenderer, ThemeColorProvider


class CompleteRenderer:
    def render(self, spec: object, **kwargs: Any) -> str:
        return f"rendered:{spec}:{kwargs.get('mode', 'default')}"

    def to_image(self, spec: object, fmt: str = "png", dpi: int = 150) -> bytes:
        return f"{spec}:{fmt}:{dpi}".encode()


class ConvertOnly:
    def convert(self, spec: object) -> dict[str, Any]:
        return {"spec": spec}


class ThemeProvider:
    def __init__(self) -> None:
        self.applied_to: list[object] = []

    def get_colors(self) -> dict[str, Any]:
        return {"primary": "#123456", "accent": "#abcdef"}

    def apply_to_figure(self, fig: object) -> None:
        self.applied_to.append(fig)


def test_plot_renderer_protocol_accepts_structural_implementations() -> None:
    renderer = CompleteRenderer()

    assert isinstance(renderer, PlotRenderer)
    assert renderer.render("plot", mode="preview") == "rendered:plot:preview"
    assert renderer.to_image("plot", fmt="svg", dpi=72) == b"plot:svg:72"


def test_plot_renderer_protocol_rejects_partial_implementations() -> None:
    assert not isinstance(ConvertOnly(), PlotRenderer)


def test_plot_converter_protocol_accepts_convert_only_implementations() -> None:
    converter = ConvertOnly()

    assert isinstance(converter, PlotConverter)
    assert converter.convert("plot") == {"spec": "plot"}


def test_theme_color_provider_protocol_accepts_theme_like_objects() -> None:
    provider = ThemeProvider()
    fig = object()

    assert isinstance(provider, ThemeColorProvider)
    assert provider.get_colors() == {"primary": "#123456", "accent": "#abcdef"}
    provider.apply_to_figure(fig)
    assert provider.applied_to == [fig]


def test_protocol_stub_methods_are_noop_contract_placeholders() -> None:
    assert PlotRenderer.render(object(), object()) is None
    assert PlotRenderer.to_image(object(), object()) is None
    assert PlotConverter.convert(object(), object()) is None
    assert ThemeColorProvider.get_colors(object()) is None
    assert ThemeColorProvider.apply_to_figure(object(), object()) is None
