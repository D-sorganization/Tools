"""Unit tests for ``sidekick.ui.design_tokens``.

Qt-free design-token loader: reads a JSON token file and renders QSS variable
snippets / flat token dicts / placeholder substitution. Tests are hermetic —
``get_tokens_path`` is monkeypatched to a temp JSON so they never depend on the
shipped ``design_tokens.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sidekick.ui import design_tokens as dt

_TOKENS = {
    "themes": {
        "light": {"background": "#ffffff", "text": "#000000"},
        "dark": {"background": "#000000", "text": "#ffffff"},
    },
    "spacing": {"sm": "4px", "md": "8px"},
    "radii": {"sm": "2px"},
}


@pytest.fixture
def tokens_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "design_tokens.json"
    path.write_text(json.dumps(_TOKENS), encoding="utf-8")
    monkeypatch.setattr(dt, "get_tokens_path", lambda: path)
    return path


def test_get_tokens_path_points_at_json() -> None:
    # The real resolver returns a Path ending in design_tokens.json.
    assert dt.get_tokens_path().name == "design_tokens.json"


def test_load_design_tokens_returns_dict(tokens_file: Path) -> None:
    data = dt.load_design_tokens()
    assert data["themes"]["light"]["background"] == "#ffffff"


def test_load_design_tokens_missing_file_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dt, "get_tokens_path", lambda: tmp_path / "absent.json")
    with pytest.raises(FileNotFoundError, match="Design tokens not found"):
        dt.load_design_tokens()


def test_load_design_tokens_non_dict_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "design_tokens.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    monkeypatch.setattr(dt, "get_tokens_path", lambda: path)
    with pytest.raises(ValueError, match="Expected dictionary"):
        dt.load_design_tokens()


def test_get_qss_variables_renders_qproperties(tokens_file: Path) -> None:
    qss = dt.get_qss_variables("light")
    assert "qproperty-background: #ffffff;" in qss
    assert "qproperty-spacing_sm: 4px;" in qss
    assert "qproperty-radius_sm: 2px;" in qss


def test_get_qss_variables_unknown_theme_raises(tokens_file: Path) -> None:
    with pytest.raises(ValueError, match="not found in design tokens"):
        dt.get_qss_variables("neon")


def test_get_token_dict_flattens_with_prefixes(tokens_file: Path) -> None:
    flat = dt.get_token_dict("dark")
    assert flat["@color_background"] == "#000000"
    assert flat["@spacing_md"] == "8px"
    assert flat["@radius_sm"] == "2px"


def test_get_token_dict_unknown_theme_raises(tokens_file: Path) -> None:
    with pytest.raises(ValueError, match="not found in design tokens"):
        dt.get_token_dict("neon")


def test_apply_tokens_to_qss_substitutes(tokens_file: Path) -> None:
    qss = "QWidget { background: @color_background; padding: @spacing_md; }"
    result = dt.apply_tokens_to_qss(qss, "light")
    assert "#ffffff" in result
    assert "8px" in result
    assert "@color_background" not in result
