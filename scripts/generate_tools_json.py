#!/usr/bin/env python3
"""Generate tools.json from gui_registration.py files."""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

def load_gui_info(path: Path) -> Optional[Dict[str, Any]]:
    """Load GUI_INFO from a python file."""
    try:

        spec = importlib.util.spec_from_file_location(f"gui_reg_{path.stem}", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, "GUI_INFO", None)
    except Exception as e:
        print(f"Warning: Failed to load {path}: {e}", file=sys.stderr)
        return None

def generate_manifest_data(repo_root: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Generate manifest data from gui_registration.py files."""
    src_dir = repo_root / "src"
    manifest: Dict[str, List[Dict[str, Any]]] = {}

    # Find all gui_registration.py files
    for reg_file in src_dir.glob("**/gui_registration.py"):
        info = load_gui_info(reg_file)
        if not info:
            continue
            
        category = info.get("category", "Uncategorized")
        if category not in manifest:
            manifest[category] = []
            
        base_name = info.get("name", "Unknown Tool")
        tool_dir = reg_file.parent
        
        # Check for PyQt6 surface
        if "pyqt6" in info:
            # Check if launch script exists
            launch_script = tool_dir / "launch_pyqt6.py"
            if launch_script.exists():
                # Determine name: if web also exists, append (PyQt6)
                name = base_name
                if "web" in info:
                    name = f"{base_name} (PyQt6)"
                    
                entry = {
                    "name": name,
                    "path": str(launch_script.relative_to(repo_root)).replace("\\", "/"),
                    "type": "python",
                    "desc": info.get("description", "")
                }
                manifest[category].append(entry)

        # Check for Web surface
        if "web" in info:
            launch_script = tool_dir / "launch_web.py"
            if launch_script.exists():
                name = base_name
                if "pyqt6" in info:
                    name = f"{base_name} (Web)"
                else:
                    name = f"{base_name} (Web)" # Web usually implies distinct runtime, so explicit suffix is good

                entry = {
                    "name": name,
                    "path": str(launch_script.relative_to(repo_root)).replace("\\", "/"),
                    "type": "python",
                    "desc": info.get("description", "")
                }
                manifest[category].append(entry)

    # Sort categories and tools for determinism
    sorted_manifest = {}
    for cat in sorted(manifest.keys()):
        tools = manifest[cat]
        # Sort tools by name
        tools.sort(key=lambda x: x["name"])
        sorted_manifest[cat] = tools
        
    return sorted_manifest

def main():
    repo_root = Path(__file__).resolve().parents[1]
    manifest = generate_manifest_data(repo_root)
    
    tools_json_path = repo_root / "tools.json"
    with open(tools_json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)
        f.write("\n") # POSIX newline
    
    print(f"Generated tools.json with {sum(len(v) for v in manifest.values())} tools.")

if __name__ == "__main__":
    main()
