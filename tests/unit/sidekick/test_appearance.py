"""Tests for the shared PanelAppearance core (no Qt required)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

_SHARED = Path(__file__).resolve().parents[3] / "src" / "shared" / "python"
_TEST_PKG = Path(__file__).resolve().parent


def _import_appearance() -> Any:
    shared_str = str(_SHARED)
    if shared_str in sys.path:
        sys.path.remove(shared_str)
    sys.path.insert(0, shared_str)
    top_mod = sys.modules.get("sidekick")
    if top_mod is not None:
        top_mod_file = getattr(top_mod, "__file__", None)
        if top_mod_file is not None and str(_TEST_PKG) in str(
            Path(top_mod_file).resolve().parent
        ):
            del sys.modules["sidekick"]
    from sidekick.ui.tools_sidebar import appearance

    return appearance


# ─── is_hex_color ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("#fff", True),
        ("#FFFFFF", True),
        ("#1e1e2e", True),
        ("  #abc  ", True),
        ("fff", False),
        ("#gggggg", False),
        ("#1234", False),
        (123, False),
        (None, False),
    ],
)
def test_is_hex_color(value: Any, expected: bool) -> None:
    ap = _import_appearance()
    assert ap.is_hex_color(value) is expected


# ─── PanelAppearance validation (DbC) ────────────────────────────


def test_valid_appearance_round_trips_to_dict() -> None:
    ap = _import_appearance()
    a = ap.PanelAppearance("#fff", "#000", "#3b82f6", border_width=3, border_radius=8)
    assert a.to_dict() == {
        "foreground": "#fff",
        "background": "#000",
        "border_color": "#3b82f6",
        "border_width": 3,
        "border_radius": 8,
    }


@pytest.mark.parametrize("bad", ["red", "#xyz", "", "12ffaa"])
def test_invalid_color_raises(bad: str) -> None:
    ap = _import_appearance()
    with pytest.raises(ValueError):
        ap.PanelAppearance(bad, "#000", "#fff")


@pytest.mark.parametrize("width", [-1, 9, 100])
def test_border_width_out_of_range_raises(width: int) -> None:
    ap = _import_appearance()
    with pytest.raises(ValueError):
        ap.PanelAppearance("#fff", "#000", "#fff", border_width=width)


def test_border_radius_out_of_range_raises() -> None:
    ap = _import_appearance()
    with pytest.raises(ValueError):
        ap.PanelAppearance("#fff", "#000", "#fff", border_radius=99)


def test_with_overrides_revalidates() -> None:
    ap = _import_appearance()
    base = ap.DEFAULT_DARK_PANEL_APPEARANCE
    changed = base.with_overrides(border_width=4)
    assert changed.border_width == 4
    assert changed.foreground == base.foreground
    with pytest.raises(ValueError):
        base.with_overrides(foreground="not-a-color")


# ─── coerce_appearance ───────────────────────────────────────────


def test_coerce_none_returns_base() -> None:
    ap = _import_appearance()
    assert ap.coerce_appearance(None) is ap.DEFAULT_DARK_PANEL_APPEARANCE


def test_coerce_non_mapping_raises() -> None:
    ap = _import_appearance()
    with pytest.raises(TypeError):
        ap.coerce_appearance([1, 2, 3])


def test_coerce_falls_back_on_invalid_fields() -> None:
    ap = _import_appearance()
    base = ap.DEFAULT_LIGHT_PANEL_APPEARANCE
    out = ap.coerce_appearance(
        {"foreground": "bogus", "background": "#222222", "border_width": 999},
        base=base,
    )
    assert out.foreground == base.foreground  # invalid -> fallback
    assert out.background == "#222222"  # valid -> kept
    assert out.border_width == ap.MAX_BORDER_WIDTH  # clamped


def test_coerce_clamps_negative_width() -> None:
    ap = _import_appearance()
    out = ap.coerce_appearance({"border_width": -5})
    assert out.border_width == 0


# ─── panel_qss ───────────────────────────────────────────────────


def test_panel_qss_contains_border_and_scope() -> None:
    ap = _import_appearance()
    a = ap.PanelAppearance("#e6e6e6", "#1e1e2e", "#89b4fa", border_width=2)
    qss = ap.panel_qss("MyPanel", a)
    assert "QWidget#MyPanel QPlainTextEdit" in qss
    assert "QWidget#MyPanel QTableView" in qss
    assert "border: 2px solid #89b4fa" in qss
    assert "#1e1e2e" in qss


def test_panel_qss_rejects_empty_object_name() -> None:
    ap = _import_appearance()
    with pytest.raises(ValueError):
        ap.panel_qss("", ap.DEFAULT_DARK_PANEL_APPEARANCE)


def test_panel_qss_rejects_wrong_type() -> None:
    ap = _import_appearance()
    with pytest.raises(TypeError):
        ap.panel_qss("Panel", {"foreground": "#fff"})


def test_panel_qss_accepts_source_qualified_appearance_alias() -> None:
    ap = _import_appearance()
    from src.shared.python.sidekick.ui.tools_sidebar.appearance import (
        PanelAppearance,
    )

    alias_appearance = PanelAppearance("#fff", "#000", "#3b82f6")

    assert ap.is_panel_appearance(alias_appearance) is True
    assert "border: 2px solid #3b82f6" in ap.panel_qss("Panel", alias_appearance)
