"""Functional tests for programmatic_pid.rendering primitives.

Tests cover: layer management, text wrapping, drawing primitives,
and DbC precondition enforcement (#3172 epic — programmatic_pid coverage).
"""

from __future__ import annotations

import pytest

ezdxf = pytest.importorskip("ezdxf", reason="ezdxf not installed")


from programmatic_pid.rendering import (  # noqa: E402
    add_box,
    add_text,
    ensure_layer,
    ensure_layers,
    wrap_text_lines,
)


def _fresh_doc():  # type: ignore[no-untyped-def]
    return ezdxf.new("R2010")


class TestWrapTextLines:
    def test_short_text_unchanged(self) -> None:
        lines = wrap_text_lines("Hello", width=20)
        assert lines == ["Hello"]

    def test_long_text_produces_multiple_lines(self) -> None:
        text = "word " * 10
        lines = wrap_text_lines(text.strip(), width=15)
        assert len(lines) > 1

    def test_empty_string_returns_single_empty_element(self) -> None:
        lines = wrap_text_lines("", width=20)
        assert lines == [""]

    def test_raises_on_none_text(self) -> None:
        with pytest.raises(ValueError):
            wrap_text_lines(None, width=20)

    def test_very_narrow_width_still_returns_list(self) -> None:
        lines = wrap_text_lines("Hello World", width=1)
        assert isinstance(lines, list)
        assert len(lines) >= 1


class TestEnsureLayer:
    def test_creates_new_layer(self) -> None:
        doc = _fresh_doc()
        ensure_layer(doc, "PROCESS", color=5)
        assert "PROCESS" in doc.layers

    def test_idempotent_on_duplicate_call(self) -> None:
        doc = _fresh_doc()
        ensure_layer(doc, "VESSELS", color=30)
        ensure_layer(doc, "VESSELS", color=30)
        assert "VESSELS" in doc.layers

    def test_raises_on_none_name(self) -> None:
        doc = _fresh_doc()
        with pytest.raises(ValueError):
            ensure_layer(doc, None)

    def test_empty_name_skipped_silently(self) -> None:
        doc = _fresh_doc()
        before = set(doc.layers)
        ensure_layer(doc, "", color=7)
        after = set(doc.layers)
        assert after == before


class TestEnsureLayers:
    def test_creates_standard_default_layers(self) -> None:
        doc = _fresh_doc()
        spec: dict = {"project": {"id": "X", "title": "Y"}, "equipment": []}
        ensure_layers(doc, spec)
        for name in ("TEXT", "NOTES", "EQUIPMENT", "INSTRUMENTS", "PROCESS"):
            assert name in doc.layers, f"Expected layer '{name}' to exist"

    def test_raises_on_none_spec(self) -> None:
        doc = _fresh_doc()
        with pytest.raises(ValueError):
            ensure_layers(doc, None)


class TestAddText:
    def test_returns_text_entity(self) -> None:
        doc = _fresh_doc()
        msp = doc.modelspace()
        entity = add_text(msp, "Hello", x=0.0, y=0.0, h=2.5)
        assert entity is not None

    def test_raises_on_none_text(self) -> None:
        doc = _fresh_doc()
        msp = doc.modelspace()
        with pytest.raises(ValueError):
            add_text(msp, None, x=0.0, y=0.0, h=2.5)

    def test_entity_added_to_modelspace(self) -> None:
        doc = _fresh_doc()
        msp = doc.modelspace()
        before = len(list(msp))
        add_text(msp, "Test", x=5.0, y=5.0, h=2.0)
        after = len(list(msp))
        assert after > before


class TestAddBox:
    def test_adds_entity_to_modelspace(self) -> None:
        doc = _fresh_doc()
        msp = doc.modelspace()
        ensure_layer(doc, "EQUIPMENT")
        before = len(list(msp))
        add_box(msp, x=0.0, y=0.0, w=10.0, h=5.0, layer="EQUIPMENT")
        after = len(list(msp))
        assert after > before

    def test_raises_on_none_x(self) -> None:
        doc = _fresh_doc()
        msp = doc.modelspace()
        with pytest.raises(ValueError):
            add_box(msp, x=None, y=0.0, w=10.0, h=5.0, layer="EQUIPMENT")
