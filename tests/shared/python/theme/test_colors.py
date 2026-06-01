from __future__ import annotations

import builtins

import pytest

from src.shared.python.theme import colors


class _FakeThemePath:
    def __init__(self, *, exists: bool) -> None:
        self._exists = exists

    @property
    def parent(self) -> _FakeThemePath:
        return self

    def __truediv__(self, _part: str) -> _FakeThemePath:
        return self

    def __fspath__(self) -> str:
        return "themes.json"

    def exists(self) -> bool:
        return self._exists


@pytest.mark.parametrize(
    "value",
    [
        "#fff",
        "fff",
        "#ffff",
        "#ffffff",
        "ffffff",
        "#ffffffff",
        "AaBbCc",
    ],
)
def test_is_valid_hex_color_accepts_supported_hex_lengths(value: str) -> None:
    assert colors.is_valid_hex_color(value) is True


@pytest.mark.parametrize("value", ["", "   ", "#ff", "#fffff", "#ggg", "blue"])
def test_is_valid_hex_color_rejects_invalid_values(value: str) -> None:
    assert colors.is_valid_hex_color(value) is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("#f00", "#ff0000"),
        ("0f0", "#00ff00"),
        ("#ABCDEF", "#abcdef"),
        ("abcdef", "#abcdef"),
    ],
)
def test_normalise_hex_color_returns_lowercase_six_digit_hex(
    value: str,
    expected: str,
) -> None:
    assert colors.normalise_hex_color(value) == expected


def test_normalise_hex_color_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="Invalid colour value"):
        colors.normalise_hex_color("not-a-colour")


def test_get_rgba_converts_six_digit_hex_and_uses_supplied_alpha() -> None:
    assert colors.get_rgba("#336699", alpha=0.25) == pytest.approx(
        (0x33 / 255, 0x66 / 255, 0x99 / 255, 0.25)
    )


def test_get_rgba_multiplies_embedded_alpha_by_supplied_alpha() -> None:
    assert colors.get_rgba("#33669980", alpha=0.5) == pytest.approx(
        (0x33 / 255, 0x66 / 255, 0x99 / 255, (0x80 / 255) * 0.5)
    )


def test_get_rgba_requires_color_value() -> None:
    with pytest.raises(ValueError, match="hex_color must be provided"):
        colors.get_rgba(None)  # type: ignore[arg-type]


def test_get_matplotlib_colors_uses_dark_theme_grid_alpha() -> None:
    palette = colors.get_matplotlib_colors(colors.BUILTIN_THEMES["Dark"])

    assert palette["figure.facecolor"] == colors.BUILTIN_THEMES["Dark"]["bg"]
    assert palette["axes.facecolor"] == colors.BUILTIN_THEMES["Dark"]["group_bg"]
    assert palette["grid.alpha"] == pytest.approx(0.3)


def test_get_matplotlib_colors_uses_light_theme_grid_alpha() -> None:
    palette = colors.get_matplotlib_colors(colors.BUILTIN_THEMES["Light"])

    assert palette["figure.facecolor"] == colors.BUILTIN_THEMES["Light"]["bg"]
    assert palette["legend.edgecolor"] == colors.BUILTIN_THEMES["Light"]["border"]
    assert palette["grid.alpha"] == pytest.approx(0.5)


def test_is_dark_theme_handles_known_and_unknown_theme_names() -> None:
    assert colors.is_dark_theme("Dark") is True
    assert colors.is_dark_theme("Light") is False
    assert colors.is_dark_theme("Definitely Missing") is False


def test_private_is_dark_theme_falls_back_for_short_background_value() -> None:
    assert colors._is_dark_theme({"bg": "#fff"}) is False


def test_get_qcolor_returns_qt_color() -> None:
    qcolor = colors.get_qcolor("#336699")

    assert qcolor.isValid()
    assert qcolor.name() == "#336699"


def test_load_theme_helpers_return_none_when_json_file_is_missing(monkeypatch) -> None:
    missing_path = _FakeThemePath(exists=False)
    monkeypatch.setattr(colors, "Path", lambda _path: missing_path)

    assert colors._load_themes_from_json() is None
    assert colors._load_chart_colors_from_json() is None


def test_load_theme_helpers_return_none_when_json_file_cannot_open(monkeypatch) -> None:
    existing_path = _FakeThemePath(exists=True)
    monkeypatch.setattr(colors, "Path", lambda _path: existing_path)

    def raise_permission_error(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(builtins, "open", raise_permission_error)

    assert colors._load_themes_from_json() is None
    assert colors._load_chart_colors_from_json() is None
