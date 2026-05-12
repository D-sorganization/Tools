"""Deeper behavioural tests for ``programmatic_pid.profiles``.

These complement :mod:`test_profiles.py` by exercising:

* Profile name normalisation (whitespace, case, surrounding spaces).
* Independence of consecutive ``apply_profile`` calls (no shared state).
* Spec is not mutated regardless of structure.
* Layout/defaults merge semantics on specs that already contain values.
* All three preset profiles roundtrip and produce monotonically smaller
  panels for ``compact`` vs ``review``.
* Every preset advertises a known set of layout knobs.
"""

from __future__ import annotations

import pytest

pytest.importorskip("ezdxf")
from programmatic_pid.profiles import PROFILE_PRESETS, apply_profile


def _full_spec() -> dict:
    return {
        "project": {
            "id": "P-1",
            "title": "Test",
            "drawing": {
                "text_height": 2.0,
                "layout": {"gap": 1.0, "panel_text_chars": 99},
            },
        },
        "defaults": {"instrument_bubble_radius": 9.9},
        "equipment": [{"id": "E-1", "x": 0, "y": 0, "width": 10, "height": 10}],
    }


# ── Name normalisation ──────────────────────────────────────────────────


class TestProfileNameNormalisation:
    @pytest.mark.parametrize(
        "name",
        [
            "review",
            "REVIEW",
            "Review",
            "  review  ",
            "ReView",
        ],
    )
    def test_case_and_whitespace_insensitive(self, name: str) -> None:
        result = apply_profile(_full_spec(), name)
        assert result["meta"]["profile"] == "review"

    def test_unknown_profile_lists_valid(self) -> None:
        with pytest.raises(ValueError) as ex:
            apply_profile(_full_spec(), "fancy")
        msg = str(ex.value)
        assert "Unknown profile" in msg
        # All known profiles should be advertised in the error message.
        for known in PROFILE_PRESETS:
            assert known in msg


# ── Spec immutability ───────────────────────────────────────────────────


class TestSpecImmutability:
    def test_input_spec_not_mutated(self) -> None:
        original = _full_spec()
        snapshot = {
            "gap": original["project"]["drawing"]["layout"]["gap"],
            "bubble": original["defaults"]["instrument_bubble_radius"],
        }
        apply_profile(original, "compact")
        # Original layout values must be unchanged after applying a profile.
        assert original["project"]["drawing"]["layout"]["gap"] == snapshot["gap"]
        assert original["defaults"]["instrument_bubble_radius"] == snapshot["bubble"]
        assert "meta" not in original  # meta is added to the copy only

    def test_independent_apply_calls(self) -> None:
        spec = _full_spec()
        a = apply_profile(spec, "review")
        b = apply_profile(spec, "compact")
        # The two outputs must not share inner dicts.
        a_layout = a["project"]["drawing"]["layout"]
        b_layout = b["project"]["drawing"]["layout"]
        a_layout["gap"] = -111.0
        assert b_layout["gap"] != -111.0


# ── Merge semantics ─────────────────────────────────────────────────────


class TestProfileMergeSemantics:
    def test_existing_layout_values_overwritten_by_preset(self) -> None:
        spec = _full_spec()
        spec["project"]["drawing"]["layout"]["gap"] = 999.0
        result = apply_profile(spec, "compact")
        assert (
            result["project"]["drawing"]["layout"]["gap"]
            == PROFILE_PRESETS["compact"]["layout"]["gap"]
        )

    def test_existing_unrelated_layout_keys_preserved(self) -> None:
        spec = _full_spec()
        spec["project"]["drawing"]["layout"]["custom_extra"] = "keep me"
        result = apply_profile(spec, "presentation")
        assert result["project"]["drawing"]["layout"]["custom_extra"] == "keep me"

    def test_drawing_created_when_missing(self) -> None:
        spec = {"project": {"id": "P-1"}, "equipment": []}
        result = apply_profile(spec, "compact")
        assert isinstance(result["project"]["drawing"], dict)
        assert "layout" in result["project"]["drawing"]
        assert result["project"]["drawing"]["layout"]["gap"] == 6.0

    def test_drawing_replaced_when_not_a_dict(self) -> None:
        spec = {"project": {"id": "P-1", "drawing": "not-a-dict"}, "equipment": []}
        result = apply_profile(spec, "compact")
        assert isinstance(result["project"]["drawing"], dict)

    def test_layout_replaced_when_not_a_dict(self) -> None:
        spec = {
            "project": {"id": "P-1", "drawing": {"layout": ["bad"]}},
            "equipment": [],
        }
        result = apply_profile(spec, "compact")
        layout = result["project"]["drawing"]["layout"]
        assert isinstance(layout, dict)
        # Preset values must be applied.
        assert layout["gap"] == 6.0

    def test_defaults_replaced_when_not_a_dict(self) -> None:
        spec = {"project": {"id": "P-1"}, "defaults": "weird", "equipment": []}
        result = apply_profile(spec, "review")
        assert isinstance(result["defaults"], dict)
        assert (
            result["defaults"]["instrument_bubble_radius"]
            == PROFILE_PRESETS["review"]["defaults"]["instrument_bubble_radius"]
        )

    def test_meta_replaced_when_not_a_dict(self) -> None:
        spec = {"project": {"id": "P-1"}, "meta": 42, "equipment": []}
        result = apply_profile(spec, "review")
        assert isinstance(result["meta"], dict)
        assert result["meta"]["profile"] == "review"

    def test_existing_meta_preserved_alongside_profile(self) -> None:
        spec = _full_spec()
        spec["meta"] = {"author": "alice"}
        result = apply_profile(spec, "review")
        assert result["meta"]["author"] == "alice"
        assert result["meta"]["profile"] == "review"


# ── Profile relative semantics ──────────────────────────────────────────


class TestProfileRelativeSizes:
    """Sanity checks: compact MUST be tighter than review on every shared knob."""

    @pytest.mark.parametrize(
        "key",
        [
            "gap",
            "right_panel_width",
            "bottom_panel_height",
            "title_block_height",
            "panel_text_chars",
            "stream_label_scale",
            "instrument_spacing_factor",
            "controls_row_height_scale",
        ],
    )
    def test_compact_smaller_than_review(self, key: str) -> None:
        review = PROFILE_PRESETS["review"]["layout"]
        compact = PROFILE_PRESETS["compact"]["layout"]
        assert compact[key] < review[key], f"{key}: compact must be < review"

    def test_review_shows_inline_notes_compact_does_not(self) -> None:
        review = PROFILE_PRESETS["review"]["layout"]
        compact = PROFILE_PRESETS["compact"]["layout"]
        presentation = PROFILE_PRESETS["presentation"]["layout"]
        assert review["show_inline_equipment_notes"] is True
        assert compact["show_inline_equipment_notes"] is False
        assert presentation["show_inline_equipment_notes"] is False

    def test_all_presets_share_same_layout_keys(self) -> None:
        keysets = [
            set(p["layout"].keys()) for p in PROFILE_PRESETS.values() if "layout" in p
        ]
        assert all(k == keysets[0] for k in keysets), (
            "all presets must declare the same layout keys for predictable merging"
        )

    def test_presentation_has_no_defaults_section(self) -> None:
        # presentation preset deliberately omits a defaults block.
        assert "defaults" not in PROFILE_PRESETS["presentation"]


# ── Empty spec ──────────────────────────────────────────────────────────


class TestEmptySpec:
    def test_empty_spec_apply_profile_produces_full_skeleton(self) -> None:
        result = apply_profile({}, "review")
        assert result["meta"]["profile"] == "review"
        assert "project" in result
        assert isinstance(result["project"]["drawing"], dict)
        assert isinstance(result["defaults"], dict)

    def test_apply_profile_none_returns_independent_copy(self) -> None:
        spec = _full_spec()
        result = apply_profile(spec, None)
        result["project"]["drawing"]["layout"]["gap"] = -1.0
        # Mutating the copy must not touch the original.
        assert spec["project"]["drawing"]["layout"]["gap"] == 1.0
