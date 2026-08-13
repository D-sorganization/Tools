"""Drift guard for the hand-maintained theme fallback (issue #3188).

``theme.colors`` loads themes from the canonical ``themes.json`` and keeps
``_HARDCODED_BUILTIN_THEMES`` only as an emergency fallback for when the
JSON is absent. That parallel dict can silently diverge from the JSON. This
test asserts the fallback's theme names and base color values stay equal to
the ``themes.json``-derived set, so ``themes.json`` remains the single source
of truth.
"""

from __future__ import annotations

import pytest

from src.shared.python.theme import colors


def _json_themes() -> dict[str, dict[str, str]]:
    themes = colors._load_themes_from_json()
    if themes is None:
        pytest.skip("themes.json not available in this environment")
    return themes


def test_fallback_theme_names_match_json() -> None:
    """The fallback exposes exactly the themes defined in themes.json."""
    json_themes = _json_themes()
    assert set(colors._HARDCODED_BUILTIN_THEMES) == set(json_themes), (
        "Hardcoded fallback theme set drifted from themes.json"
    )


def test_fallback_base_colors_match_json() -> None:
    """Every fallback base color value equals the themes.json value."""
    json_themes = _json_themes()
    mismatches: list[str] = []
    for name, fallback in colors._HARDCODED_BUILTIN_THEMES.items():
        json_theme = json_themes.get(name, {})
        for key, value in fallback.items():
            json_value = json_theme.get(key)
            if json_value != value:
                mismatches.append(
                    f"{name}.{key}: fallback={value!r} json={json_value!r}"
                )
    assert not mismatches, "Fallback drifted from themes.json:\n" + "\n".join(
        mismatches
    )


def test_chart_colors_fallback_matches_json() -> None:
    """The hardcoded chart-color fallback equals themes.json chartColors."""
    json_chart = colors._load_chart_colors_from_json()
    if json_chart is None:
        pytest.skip("themes.json not available in this environment")
    assert colors._HARDCODED_CHART_COLORS == json_chart, (
        "Hardcoded chart-color fallback drifted from themes.json"
    )


def test_builtin_themes_is_json_derived_when_available() -> None:
    """When themes.json is present, BUILTIN_THEMES is the JSON-derived set."""
    json_themes = _json_themes()
    assert set(colors.BUILTIN_THEMES) == set(json_themes)
