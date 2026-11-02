import json
from pathlib import Path
from typing import Any

__all__ = [
    "apply_tokens_to_qss",
    "get_qss_variables",
    "get_token_dict",
    "get_tokens_path",
    "load_design_tokens",
]


def get_tokens_path() -> Path:
    """Return the absolute path to the shared design_tokens.json file."""
    # ui -> sidekick -> python -> shared
    return Path(__file__).resolve().parents[3] / "design_tokens.json"


# themes.json is the single source of theme colors (Tools #5917).
_THEMES_JSON = get_tokens_path().parent / "theme-definitions" / "themes.json"


def _load_catalog_themes() -> dict[str, dict[str, str]]:
    """Flatten each themes.json theme's colors and semantic maps, keyed by id."""
    with open(_THEMES_JSON, encoding="utf-8") as f:
        catalog = json.load(f)
    return {
        theme_id: {**theme["colors"], **theme["semantic"]}
        for theme_id, theme in catalog["themes"].items()
    }


def load_design_tokens() -> dict[str, Any]:
    """Load the design tokens, with theme colors read from themes.json.

    The ``themes`` table in design_tokens.json is a generated copy
    (``scripts/sync_design_tokens.py``); the catalog is read directly so a
    color changed in themes.json alone reaches every token consumer.
    """
    path = get_tokens_path()
    if not path.exists():
        raise FileNotFoundError(f"Design tokens not found at {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Expected dictionary in design_tokens.json")
    tokens = dict(data)
    tokens["themes"] = _load_catalog_themes()
    return tokens


def get_qss_variables(theme_name: str = "light") -> str:
    """
    Generate PyQt6 QSS variables from the design tokens for a specific theme.
    Returns a QSS snippet defining global properties (e.g. on QWidget or root).
    """
    tokens = load_design_tokens()

    if theme_name not in tokens.get("themes", {}):
        raise ValueError(f"Theme '{theme_name}' not found in design tokens.")

    theme_colors = tokens["themes"][theme_name]
    spacing = tokens.get("spacing", {})
    radii = tokens.get("radii", {})

    # Generate QSS snippet
    # Note: PyQt6 doesn't fully support CSS custom properties natively.
    # We provide a formatted string for string replacement in stylesheets.
    # We will format it as a dictionary of values or standard CSS variables.

    qss_lines = ["/* Design Tokens */"]
    for key, value in theme_colors.items():
        qss_lines.append(f"qproperty-{key}: {value};")

    for key, value in spacing.items():
        qss_lines.append(f"qproperty-spacing_{key}: {value};")

    for key, value in radii.items():
        qss_lines.append(f"qproperty-radius_{key}: {value};")

    return "\n".join(qss_lines)


def get_token_dict(theme_name: str = "light") -> dict[str, str]:
    """Return a flat dictionary of tokens for string replacement in QSS."""
    tokens = load_design_tokens()
    if theme_name not in tokens.get("themes", {}):
        raise ValueError(f"Theme '{theme_name}' not found in design tokens.")

    theme_colors = tokens["themes"][theme_name]
    spacing = tokens.get("spacing", {})
    radii = tokens.get("radii", {})

    flat_tokens = {}
    for k, v in theme_colors.items():
        flat_tokens[f"@color_{k}"] = v
    for k, v in spacing.items():
        flat_tokens[f"@spacing_{k}"] = v
    for k, v in radii.items():
        flat_tokens[f"@radius_{k}"] = v

    return flat_tokens


def apply_tokens_to_qss(qss_content: str, theme_name: str = "light") -> str:
    """Replace token placeholders with actual values in a QSS string."""
    tokens = get_token_dict(theme_name)
    result = qss_content
    for key, value in tokens.items():
        result = result.replace(key, value)
    return result
