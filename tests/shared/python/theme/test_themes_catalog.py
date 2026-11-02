"""Built-in theme catalog guards (issue #5915).

No built-in theme is named after, or color-matched to, a commercial product,
and every retired theme id still resolves so saved preferences keep working.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from uuid import uuid4

import pytest
from PyQt6.QtCore import QSettings

THEMES_JSON = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "shared"
    / "theme-definitions"
    / "themes.json"
)
PRODUCT_NAME = re.compile(r"(ms[-_ ]?)?(word|excel|office|outlook|powerpoint)", re.I)
BRAND_HEXES = {"#2b579a", "#217346", "#185abd", "#107c41"}


@pytest.fixture(scope="module")
def catalog() -> dict:
    return json.loads(THEMES_JSON.read_text(encoding="utf-8"))


def test_no_product_names(catalog: dict) -> None:
    offenders = [
        f"{theme_id}: {value}"
        for theme_id, theme in catalog["themes"].items()
        for value in (theme_id, theme["name"], theme["category"])
        if PRODUCT_NAME.search(value)
    ]
    assert not offenders, offenders


def test_no_brand_hex(catalog: dict) -> None:
    offenders = [
        f"{theme_id}.{key}={theme['colors'][key]}"
        for theme_id, theme in catalog["themes"].items()
        for key in ("accent", "focus")
        if theme["colors"][key].lower() in BRAND_HEXES
    ]
    assert not offenders, offenders


def test_legacy_ids_resolve(catalog: dict) -> None:
    legacy = catalog["legacyIds"]
    for old_id in ("ms-word", "office-blue", "corporate-blue", "ms-excel"):
        assert old_id in legacy
    assert not set(legacy) & set(catalog["themes"]), "a legacy id shadows a theme"
    missing = {old: new for old, new in legacy.items() if new not in catalog["themes"]}
    assert not missing, missing


@pytest.mark.parametrize(
    ("saved", "expected"),
    [
        ("ms-word", "Cobalt Light"),
        ("office-blue", "Cobalt Light"),
        ("Office Blue", "Cobalt Light"),
        ("Corporate Blue", "Cobalt Light"),
        ("Office Green", "Fern Light"),
        ("legal-pad", "Sandstone Yellow"),
        ("Dracula", "Vampire Dark"),
        ("Light", "Light"),
        ("My Custom Theme", "My Custom Theme"),
    ],
)
def test_resolve_theme_name(saved: str, expected: str) -> None:
    from shared.python.theme import colors

    assert colors.resolve_theme_name(saved) == expected


@pytest.mark.parametrize("saved", ["ms-word", "Office Blue", "corporate-blue"])
def test_saved_legacy_preference_loads_renamed_theme(
    saved: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from shared.python.theme import theme_manager

    monkeypatch.setattr(
        theme_manager.QStandardPaths, "writableLocation", lambda _loc: str(tmp_path)
    )
    org, app = f"ThemeCatalogTests-{uuid4().hex}", "App"
    QSettings(org, app).setValue("theme", saved)

    manager = theme_manager.ThemeManager(settings_org=org, settings_app=app)
    assert manager.get_current_theme_name() == "Cobalt Light"

    manager.change_theme("office-green")
    assert manager.get_current_theme_name() == "Fern Light"


def test_validator_rejects_dangling_legacy_id(
    catalog: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "validate_themes", THEMES_JSON.parents[3] / "scripts" / "validate_themes.py"
    )
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    broken = dict(catalog, legacyIds={"ms-word": "no-such-theme", "light": "dark"})
    bad_json = tmp_path / "themes.json"
    bad_json.write_text(json.dumps(broken), encoding="utf-8")
    monkeypatch.setattr(validator, "THEMES_JSON", bad_json)

    errors = validator.validate()
    assert any("no-such-theme" in err for err in errors)
    assert any("shadows" in err for err in errors)
