"""Configuration loader utility for generic tool launchers."""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Preferred order for tool categories in launchers
CATEGORY_ORDER = [
    "Media Processing",
    "Data Processing",
    "Scientific Modeling",
    "Web Applications",
    "Development Tools",
]


def load_tools_config(repo_root: Path) -> dict[str, list[Any]]:
    """Load tools configuration from tools.json.

    Args:
        repo_root: Root directory of the repository.

    Returns:
        Dictionary mapping categories to lists of tools.
    """
    json_path = repo_root / "tools.json"
    if not json_path.exists():
        logger.warning(f"tools.json not found at {json_path}")
        return {}

    try:
        with open(json_path, encoding="utf-8") as f:
            config = json.load(f)
        return validate_tools_config(config)
    except Exception as e:
        logger.error(f"Error loading tools.json: {e}")
        return {}


def validate_tools_config(
    tools_dict: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Validate and sanitize tools configuration.

    Args:
        tools_dict: Dictionary of tool categories and lists of tools.

    Returns:
        Validated dictionary with invalid entries removed.
    """
    validated = {}

    for category, tools in tools_dict.items():
        if not isinstance(tools, list):
            continue

        valid_tools = []
        for tool in tools:
            # Basic validation
            if not isinstance(tool, dict):
                continue

            if "name" not in tool or "path" not in tool:
                continue

            valid_tools.append(tool)

        if valid_tools:
            validated[category] = valid_tools

    return validated
