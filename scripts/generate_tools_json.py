# ruff: noqa: T201
"""Generate tools.json and tool_surface_contract.json from gui_registration.py sources.

Usage:
    python scripts/generate_tools_json.py [--repo-root PATH]

Scans all gui_registration.py files under src/ and generates:
1. tools.json - Manifest grouped by category with launch paths
2. tool_surface_contract.json - Contract listing surface availability per tool
"""

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _load_gui_info(registration_path: Path) -> dict[str, Any] | None:
    """Load GUI_INFO dict from a gui_registration.py file.

    Args:
        registration_path: Path to a gui_registration.py file.

    Returns:
        The GUI_INFO dict or None if it can't be loaded.
    """
    try:
        spec = importlib.util.spec_from_file_location(
            f"gui_reg_{registration_path.parent.name}", registration_path
        )
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, "GUI_INFO", None)
    except Exception as exc:
        logger.warning("Could not load %s: %s", registration_path, exc)
        return None


def _find_gui_registrations(repo_root: Path) -> list[Path]:
    """Find all gui_registration.py files under src/.

    Args:
        repo_root: Repository root directory.

    Returns:
        Sorted list of gui_registration.py file paths.
    """
    src_dir = repo_root / "src"
    if not src_dir.exists():
        return []
    return sorted(src_dir.rglob("gui_registration.py"))


def generate_manifest_data(repo_root: Path) -> dict[str, list[dict[str, Any]]]:
    """Generate tools.json manifest data.

    Scans gui_registration.py files, and for dual-surface tools
    (pyqt6 + web), expands them into two manifest entries.

    Args:
        repo_root: Repository root directory.

    Returns:
        Dict mapping category names to lists of tool entries.
    """
    manifest: dict[str, list[dict[str, Any]]] = {}

    for reg_path in _find_gui_registrations(repo_root):
        gui_info = _load_gui_info(reg_path)
        if gui_info is None:
            continue

        name = gui_info.get("name", reg_path.parent.name)
        category = gui_info.get("category", "Uncategorized")
        tool_dir = reg_path.parent

        if category not in manifest:
            manifest[category] = []

        has_pyqt6 = "pyqt6" in gui_info
        has_web = "web" in gui_info

        if has_pyqt6:
            launch_path = tool_dir / "launch_pyqt6.py"
            display_name = f"{name} (PyQt6)" if has_web else name
            manifest[category].append(
                {
                    "name": display_name,
                    "type": "python",
                    "path": str(launch_path.relative_to(repo_root)),
                }
            )

        if has_web:
            launch_path = tool_dir / "launch_web.py"
            display_name = f"{name} (Web)" if has_pyqt6 else name
            manifest[category].append(
                {
                    "name": display_name,
                    "type": "web",
                    "path": str(launch_path.relative_to(repo_root)),
                }
            )

    # Sort entries within each category
    for category in manifest:
        manifest[category].sort(key=lambda t: t["name"])

    return manifest


def generate_contract_data(repo_root: Path) -> dict[str, Any]:
    """Generate tool_surface_contract.json data.

    Each tool gets a single entry with boolean surface flags.

    Args:
        repo_root: Repository root directory.

    Returns:
        Contract dict with version and tools list.
    """
    tools: list[dict[str, Any]] = []

    for reg_path in _find_gui_registrations(repo_root):
        gui_info = _load_gui_info(reg_path)
        if gui_info is None:
            continue

        tool_name = gui_info.get("tool_name", reg_path.parent.name)
        tools.append(
            {
                "id": tool_name,
                "name": gui_info.get("name", tool_name),
                "description": gui_info.get("description", ""),
                "category": gui_info.get("category", "Uncategorized"),
                "surfaces": {
                    "pyqt6": "pyqt6" in gui_info,
                    "web": "web" in gui_info,
                },
            }
        )

    # Sort by ID for deterministic output
    tools.sort(key=lambda t: t["id"])

    return {
        "version": "1.0.0",
        "tools": tools,
    }


def main() -> None:
    """CLI entry point for generating tools.json and contract."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate tools.json manifest")
    parser.add_argument(
        "--repo-root",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root directory.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    repo_root = Path(args.repo_root).resolve()

    # Generate manifest
    manifest = generate_manifest_data(repo_root)
    manifest_path = repo_root / "tools.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info("Wrote %s", manifest_path)

    # Generate contract
    contract = generate_contract_data(repo_root)
    contract_path = repo_root / "tool_surface_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    logger.info("Wrote %s", contract_path)

    print(
        f"Generated manifest ({len(manifest)} categories) and contract ({len(contract['tools'])} tools)"
    )


if __name__ == "__main__":
    main()
