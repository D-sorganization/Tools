#!/usr/bin/env python3
"""Generate tools.json and tool_surface_contract.json from gui_registration.py files.

This script scans all gui_registration.py files under src/ and produces:
1. tools.json — Unified Launcher manifest (categorized, surface-expanded entries)
2. tool_surface_contract.json — Cross-repo parity contract (one entry per logical tool)
"""

import importlib.util
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CONTRACT_VERSION = "1.0.0"


def _emit_stdout(message: str) -> None:
    """Write a single line to stdout for the CLI contract."""
    sys.stdout.write(f"{message}\n")


@dataclass(frozen=True)
class ToolRegistration:
    """Internal representation of a discovered tool registration.

    Invariants (Design by Contract):
    - id is always a non-empty snake_case string
    - name is always a non-empty string
    - category is always a non-empty string
    - At least one surface (has_pyqt6 or has_web) is True
    """

    id: str
    name: str
    description: str
    category: str
    has_pyqt6: bool
    has_web: bool
    has_legacy_gui: bool = False  # Tkinter / non-PyQt6 GUI (launch_gui.py)


def load_gui_info(path: Path) -> dict[str, Any] | None:
    """Load GUI_INFO from a python file.

    Pre-condition: path is a valid file path.
    Post-condition: Returns the GUI_INFO dict or None if not loadable.
    """
    try:
        spec = importlib.util.spec_from_file_location(
            f"gui_reg_{path.parent.name}", path
        )
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, "GUI_INFO", None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load %s: %s", path, exc)
        return None


def _is_catalog_visible(info: dict[str, Any]) -> bool:
    """Return whether a registration participates in generated catalogs."""
    return info.get("catalog_visible", True) is not False


def _discover_registrations(repo_root: Path) -> list[ToolRegistration]:
    """Discover all gui_registration.py files and extract ToolRegistration entries.

    Pre-condition: repo_root / 'src' exists.
    Post-condition: Returns a sorted (by id) list of unique ToolRegistrations.
    """
    src_dir = repo_root / "src"
    seen_ids: dict[str, ToolRegistration] = {}

    for reg_file in sorted(src_dir.glob("**/gui_registration.py")):
        info = load_gui_info(reg_file)
        if not info:
            continue
        if not _is_catalog_visible(info):
            continue

        tool_dir = reg_file.parent

        # Determine tool ID: prefer explicit 'tool_name', fall back to directory name
        tool_id = info.get("tool_name") or tool_dir.name

        # Determine surface availability
        has_pyqt6 = "pyqt6" in info and (tool_dir / "launch_pyqt6.py").exists()
        has_web = "web" in info and (tool_dir / "launch_web.py").exists()
        # Legacy/Tkinter GUI: explicit launch_gui.py with no PyQt6 surface
        has_legacy_gui = (
            not has_pyqt6
            and "tkinter" in info
            and (tool_dir / "launch_gui.py").exists()
        )

        # A registration that declares a surface it cannot launch is a
        # contradiction, not an absence: `src/p1am_control_system` declared a
        # full "pyqt6" block but shipped no launch_pyqt6.py, so it vanished from
        # tools.json, tool_surface_contract.json and the README with a green
        # --check and no output at all (Tools#4916). Dropping it stays the
        # behaviour -- the catalog cannot advertise something with no entry
        # point -- but it is now reported, so deleting a launcher shim can no
        # longer silently delete a tool.
        if not has_pyqt6 and not has_web and not has_legacy_gui:
            declared = sorted(k for k in ("pyqt6", "web", "tkinter") if k in info)
            if declared:
                logger.warning(
                    "Registration %s declares %s but ships no matching launcher "
                    "script; omitting '%s' from generated catalogs",
                    reg_file,
                    ", ".join(declared),
                    tool_id,
                )
            continue

        registration = ToolRegistration(
            id=tool_id,
            name=info.get("name", "Unknown Tool"),
            description=info.get("description", ""),
            category=info.get("category", "Uncategorized"),
            has_pyqt6=has_pyqt6,
            has_web=has_web,
            has_legacy_gui=has_legacy_gui,
        )

        if tool_id in seen_ids:
            logger.warning(
                "Duplicate tool_name '%s' found, skipping %s", tool_id, reg_file
            )
        else:
            seen_ids[tool_id] = registration

    return sorted(seen_ids.values(), key=lambda r: r.id)


def generate_manifest_data(repo_root: Path) -> dict[str, list[dict[str, Any]]]:
    """Generate manifest data (tools.json format) from gui_registration.py files.

    Pre-condition: repo_root is a valid repo root with src/ directory.
    Post-condition: Returns a dict keyed by category with sorted tool entries.
    """
    registrations = _discover_registrations(repo_root)
    manifest: dict[str, list[dict[str, Any]]] = {}

    for reg in registrations:
        if reg.category not in manifest:
            manifest[reg.category] = []

        if reg.has_pyqt6:
            name = f"{reg.name} (PyQt6)" if reg.has_web else reg.name
            src_dir = repo_root / "src"
            # We need to find the actual registration file to get the right path
            tool_dir = _find_tool_dir(src_dir, reg.id)
            if tool_dir:
                launch_script = tool_dir / "launch_pyqt6.py"
                manifest[reg.category].append(
                    {
                        "name": name,
                        "path": str(launch_script.relative_to(repo_root)).replace(
                            "\\", "/"
                        ),
                        "type": "python",
                        "desc": reg.description,
                    }
                )

        if reg.has_web:
            name = f"{reg.name} (Web)" if reg.has_pyqt6 else f"{reg.name} (Web)"
            src_dir = repo_root / "src"
            tool_dir = _find_tool_dir(src_dir, reg.id)
            if tool_dir:
                launch_script = tool_dir / "launch_web.py"
                manifest[reg.category].append(
                    {
                        "name": name,
                        "path": str(launch_script.relative_to(repo_root)).replace(
                            "\\", "/"
                        ),
                        "type": "python",
                        "desc": reg.description,
                    }
                )

        if reg.has_legacy_gui:
            src_dir = repo_root / "src"
            tool_dir = _find_tool_dir(src_dir, reg.id)
            if tool_dir:
                launch_script = tool_dir / "launch_gui.py"
                manifest[reg.category].append(
                    {
                        "name": reg.name,
                        "path": str(launch_script.relative_to(repo_root)).replace(
                            "\\", "/"
                        ),
                        "type": "python",
                        "desc": reg.description,
                    }
                )

    # Sort categories and tools for determinism
    sorted_manifest: dict[str, list[dict[str, Any]]] = {}
    for cat in sorted(manifest.keys()):
        tools = manifest[cat]
        tools.sort(key=lambda x: x["name"])
        sorted_manifest[cat] = tools

    return sorted_manifest


def _find_tool_dir(src_dir: Path, tool_id: str) -> Path | None:
    """Find the directory containing a tool's gui_registration.py by its ID.

    Searches gui_registration.py files for a matching tool_name or directory name.
    """
    for reg_file in sorted(src_dir.glob("**/gui_registration.py")):
        info = load_gui_info(reg_file)
        if not info:
            continue
        if not _is_catalog_visible(info):
            continue
        candidate_id = info.get("tool_name") or reg_file.parent.name
        if candidate_id == tool_id:
            return reg_file.parent
    return None


def generate_contract_data(repo_root: Path) -> dict[str, Any]:
    """Generate tool surface contract data for cross-repo parity checking.

    Pre-condition: repo_root is a valid repo root with src/ directory.
    Post-condition: Returns a dict conforming to tool_surface_contract.schema.json:
      - "version": semver string
      - "tools": list of tool entries sorted by id, each with:
        - "id": snake_case tool identifier
        - "name": human-readable display name
        - "description": tool description
        - "category": tool category
        - "surfaces": {"pyqt6": bool, "web": bool}
    """
    registrations = _discover_registrations(repo_root)

    tools: list[dict[str, Any]] = []
    for reg in registrations:
        tools.append(
            {
                "id": reg.id,
                "name": reg.name,
                "description": reg.description,
                "category": reg.category,
                "surfaces": {
                    "pyqt6": reg.has_pyqt6,
                    "web": reg.has_web,
                    "legacy_gui": reg.has_legacy_gui,
                },
            }
        )

    return {
        "version": CONTRACT_VERSION,
        "tools": tools,
    }


def main() -> int:
    """Generate tools.json and optionally tool_surface_contract.json."""
    repo_root = Path(__file__).resolve().parents[1]

    # Generate manifest
    manifest = generate_manifest_data(repo_root)
    tools_json_path = repo_root / "tools.json"
    with open(tools_json_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=4)
        f.write("\n")  # POSIX newline

    tool_count = sum(len(v) for v in manifest.values())
    logger.info("Generated tools.json with %d tools.", tool_count)

    # Generate contract
    contract = generate_contract_data(repo_root)
    contract_path = repo_root / "tool_surface_contract.json"
    with open(contract_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(contract, f, indent=2)
        f.write("\n")

    logger.info(
        "Generated tool_surface_contract.json with %d tools.",
        len(contract["tools"]),
    )

    # Keep the CLI contract explicit for CI callers without using raw print().
    _emit_stdout(f"Generated tools.json with {tool_count} tools.")
    _emit_stdout(
        f"Generated tool_surface_contract.json with {len(contract['tools'])} tools."
    )

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
