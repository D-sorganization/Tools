"""Configuration loader utility for generic tool launchers.

This module provides tool-specific configuration loading functionality
that builds on the centralized ConfigLoader from utils.
"""

import logging
from pathlib import Path
from typing import Any

from _bootstrap import bootstrap  # noqa: E402

_REPO_ROOT = bootstrap(__file__)

from utils.file_utils import safe_read_json  # noqa: E402

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
    repo_root: Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Validate and sanitize tools configuration.

    Args:
        tools_dict: Dictionary of tool categories and lists of tools.
        repo_root: Root directory of the repository for path validation.

    Returns:
        Validated dictionary with invalid entries removed.
    """
    if not (tools_dict is not None):
        raise ValueError("tools_dict must be provided")
    from src.shared.python.contracts import require

    require(isinstance(tools_dict, dict), "tools_dict must be a dictionary")

    validated = {}

    # Resolve repo_root once for consistent comparison
    resolved_root = repo_root.resolve() if repo_root is not None else None

    for category, tools in tools_dict.items():
        if not isinstance(tools, list):
            # Log warning or skip invalid structure
            continue

        valid_tools = []
        for tool in tools:
            # Runtime validation for JSON data that might not match the type hint
            if not isinstance(tool, dict):
                continue

            if "name" not in tool or "path" not in tool:
                continue

            # Security: Validate path does not escape the repository root
            path = str(tool["path"])
            if resolved_root is not None:
                resolved_path = (resolved_root / path).resolve()
                if not str(resolved_path).startswith(str(resolved_root)):
                    logger.warning(f"Skipping tool path that escapes repo root: {path}")
                    continue
            else:
                # Fallback: reject paths containing ".." when no repo_root provided
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
    from src.shared.python.contracts import require

    require(isinstance(repo_root, Path), "repo_root must be a Path object")

    json_path = repo_root / "tools.json"

    config = safe_read_json(json_path, default={})

    if not config:
        logger.warning(f"tools.json not found or empty at {json_path}")
        return {}

    return validate_tools_config(config, repo_root=repo_root)
