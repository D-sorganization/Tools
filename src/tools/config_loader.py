"""Configuration loader utility for generic tool launchers.

This module provides tool-specific configuration loading functionality
that builds on the centralized ConfigLoader from utils.
"""

import logging
import sys
from pathlib import Path
from typing import Any

# Try to import from centralized utils
try:
    from utils.file_utils import safe_read_json
except ImportError:
    # Fallback if utils not in path
    _src_path = Path(__file__).resolve().parent.parent / "python" / "src"
    if str(_src_path) not in sys.path:
        sys.path.insert(0, str(_src_path))
    from utils.file_utils import safe_read_json

logger = logging.getLogger(__name__)

# Preferred order for tool categories in launchers
CATEGORY_ORDER = [
    "Media Processing",
    "Data Processing",
    "Scientific Modeling",
    "Web Applications",
    "Development Tools",
]


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
        valid_tools = []
        for tool in tools:
            # Runtime validation for JSON data that might not match the type hint
            if not isinstance(tool, dict):
                continue

            if "name" not in tool or "path" not in tool:
                continue

            # Security: Sanitize path input
            path = str(tool["path"])
            if ".." in path:
                logger.warning(f"Skipping potentially unsafe tool path: {path}")
                continue

            valid_tools.append(tool)

        if valid_tools:
            validated[category] = valid_tools

    return validated


def load_tools_config(repo_root: Path) -> dict[str, list[Any]]:
    """Load tools configuration from tools.json.

    Uses the centralized safe_read_json for consistent error handling.

    Args:
        repo_root: Root directory of the repository.

    Returns:
        Dictionary mapping categories to lists of tools.
    """
    json_path = repo_root / "tools.json"

    config = safe_read_json(json_path, default={})

    if not config:
        logger.warning(f"tools.json not found or empty at {json_path}")
        return {}

    return validate_tools_config(config)
