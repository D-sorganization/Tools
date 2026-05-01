# Plugin System Documentation

## Overview

The Tools repository supports two methods for tool registration:

1. **Centralized Registration** (`tools.json`) — Manual, explicit tool registration
2. **Automatic Discovery** (`tool_manifest.json`) — Per-tool manifest files for auto-discovery

Both methods are supported simultaneously. Discovered tools are merged with `tools.json` entries at launcher startup.

---

## Manifest Format Reference

### `tool_manifest.json` (per-tool auto-discovery)

Place this file in the tool's root directory (e.g. `src/my_tool/tool_manifest.json`):

```json
{
  "name": "My Awesome Tool",
  "path": "launch_pyqt6.py",
  "type": "python",
  "description": "A tool that does amazing things",
  "category": "Development Tools"
}
```

**Fields:**

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `name` | Yes | string | — | Display name shown in the launcher |
| `path` | No | string | auto-detected | Relative path to the tool's entry point. Omit to let the system scan for `*.py` files |
| `type` | No | string | `"python"` | Tool type: `python`, `matlab`, `web`, `browser`, `bat` |
| `description` | No | string | `""` | Short description shown in the launcher tooltip |
| `category` | No | string | `"Development Tools"` | Category for grouping tools in the launcher UI |

### `tools.json` (centralized registry)

The root `tools.json` groups tools by category. Each entry has:

```json
{
  "Category Name": [
    {
      "name": "Tool Display Name",
      "path": "src/my_tool/launch_pyqt6.py",
      "type": "python",
      "desc": "Short description"
    }
  ]
}
```

**Fields:**

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `name` | Yes | string | Display name |
| `path` | Yes | string | Path relative to repo root |
| `type` | Yes | string | `python`, `matlab`, `web`, `browser`, `bat` |
| `desc` | No | string | Short description |

---

## GUI Registration (`gui_registration.py`)

Every PyQt6 tool that appears in the unified launcher must expose a `gui_registration.py`
module at its root directory. This file provides metadata the launcher uses to
instantiate the tool's window.

### Required Contract

```python
"""GUI registration for My Tool."""

from __future__ import annotations
from typing import Any

GUI_INFO = {
    "name": "My Tool",                   # Display name in the launcher
    "tool_name": "my_tool",              # Unique snake_case identifier
    "description": "Does something useful",
    "category": "Process Simulation",    # Launcher tab/group
    "icon": "wrench",                    # Icon name (see src/shared/python/icon_utils.py)
    "pyqt6": {
        "module": "my_tool.python.my_tool.ui.pyqt6.main_window",  # Fully-qualified module
        "class": "MyToolWidget",          # Class to instantiate
        "dependencies": ["PyQt6", "numpy"],
        "settings_app": "MyTool",         # QSettings app name
        "min_size": [900, 600],           # [width, height] in pixels
    },
    "web": {                             # Optional: omit if no web interface
        "port": 5175,
        "auto_open_browser": True,
    },
}


def get_gui_info() -> dict[str, Any]:
    """Return GUI registration information."""
    return GUI_INFO
```

**Key requirements:**
- The `get_gui_info()` function is the only public API the launcher calls
- `tool_name` must be unique across all tools (the launcher uses it as a key)
- `pyqt6.module` must be importable after `pip install -e .` or `python3 _bootstrap.py`
- `pyqt6.class` must be a `QWidget` subclass that can be instantiated with no arguments

---

## Tool Discovery Flow

```
UnifiedToolsLauncher startup
         │
         ├─► Load tools.json  (centralized registry)
         │         │
         │         └─► Build initial tool list
         │
         ├─► Scan src/ for tool_manifest.json  (auto-discovery)
         │         │
         │         └─► Merge discovered tools into tool list
         │                (duplicate names are deduplicated — tools.json wins)
         │
         ├─► Load gui_registration.py for each PyQt6 tool
         │         │
         │         └─► Populate launcher tabs with GUI_INFO metadata
         │
         └─► Present launcher UI
```

---

## Adding a New Tool (Step-by-Step)

### 1. Create the Tool Directory

```bash
mkdir -p src/my_tool/python/my_tool/ui/pyqt6
touch src/my_tool/__init__.py
touch src/my_tool/python/my_tool/__init__.py
touch src/my_tool/python/my_tool/ui/__init__.py
touch src/my_tool/python/my_tool/ui/pyqt6/__init__.py
```

### 2. Implement Core Logic

```python
# src/my_tool/python/my_tool/core.py
"""Core calculation logic for My Tool."""

from __future__ import annotations


def calculate(inputs: dict) -> dict:
    """Run the main calculation.

    Args:
        inputs: Dict with validated input parameters.

    Returns:
        Dict with calculation results.

    Raises:
        ValueError: If inputs fail validation.
    """
    # ... implementation ...
    return {"result": 42.0}
```

### 3. Create the PyQt6 Main Window

```python
# src/my_tool/python/my_tool/ui/pyqt6/main_window.py
"""PyQt6 main window for My Tool."""

from __future__ import annotations
import logging
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel

logger = logging.getLogger(__name__)


class MyToolWidget(QWidget):
    """Main window widget for My Tool."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("My Tool")
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("My Tool"))
```

### 4. Create `launch_pyqt6.py`

```python
# src/my_tool/launch_pyqt6.py
"""Entry-point for launching My Tool as a standalone PyQt6 application."""

from __future__ import annotations
import logging
import sys
from pathlib import Path

# Bootstrap: add repo paths before any package imports
_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root / "src" / "shared" / "python"))
sys.path.insert(0, str(_root / "src"))

from upstream_drift_tools.bootstrap import ensure_paths
ensure_paths(_root)

from PyQt6.QtWidgets import QApplication
from my_tool.ui.pyqt6.main_window import MyToolWidget

logger = logging.getLogger(__name__)


def main() -> None:
    """Launch My Tool."""
    logging.basicConfig(level=logging.INFO)
    app = QApplication(sys.argv)
    window = MyToolWidget()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

### 5. Create `gui_registration.py`

See the [GUI Registration](#gui-registration-gui_registrationpy) section above for the required format.

### 6. Create `tool_manifest.json`

```json
{
  "name": "My Tool",
  "path": "launch_pyqt6.py",
  "type": "python",
  "description": "Does something useful",
  "category": "Development Tools"
}
```

### 7. Add to `tools.json` (Optional)

If you want explicit control over ordering in the launcher:

```json
{
  "Development Tools": [
    {
      "name": "My Tool",
      "path": "src/my_tool/launch_pyqt6.py",
      "type": "python",
      "desc": "Does something useful"
    }
  ]
}
```

### 8. Write Tests

```bash
mkdir -p tests/my_tool
touch tests/my_tool/__init__.py
touch tests/my_tool/test_core.py
touch tests/my_tool/test_contracts.py
```

Contract tests guard the public API surface that downstream repos depend on:

```python
# tests/my_tool/test_contracts.py
import pytest
from my_tool.core import calculate

@pytest.mark.contract
def test_calculate_returns_dict():
    """calculate() must always return a dict — downstream repos depend on this."""
    result = calculate({"input": 1.0})
    assert isinstance(result, dict)
    assert "result" in result
```

### 9. Validate CI Requirements

```bash
# Lint and format
python3 -m ruff check .
python3 -m ruff format .

# Type checking
python3 -m mypy src/my_tool/

# Tests
python3 -m pytest tests/my_tool/ -v -m "unit or contract"
```

---

## Automatic Discovery vs Manual Registration

| Aspect | `tool_manifest.json` | `tools.json` |
|--------|---------------------|--------------|
| Location | Per-tool directory | Repo root |
| Ordering | Not guaranteed | Explicit |
| Maintenance | Zero (self-describing) | Manual update required |
| Discovery | Automatic at startup | Loaded once at startup |
| Duplicates | Deduplicated (tools.json wins) | Authoritative |
| Use case | New tools, rapid iteration | Stable tools needing ordered position |

**Recommendation:** Use `tool_manifest.json` for new tools. Add to `tools.json` only
if you need to control ordering or the tool is part of a stable public API.

---

## Troubleshooting

### Tool does not appear in the launcher

1. Check `tool_manifest.json` exists in the tool's root directory
2. Check `tools.json` has an entry with the correct path
3. Verify `launch_pyqt6.py` is executable: `python3 src/my_tool/launch_pyqt6.py`
4. Check for import errors: `python3 -c "from my_tool.ui.pyqt6.main_window import MyToolWidget"`

### ImportError when launching

Most likely the bootstrap path is wrong. Check:

```python
# In launch_pyqt6.py, ensure _root points to the repo root:
_root = Path(__file__).resolve().parents[2]  # adjust depth!
# src/my_tool/launch_pyqt6.py → parents[0]=my_tool, parents[1]=src, parents[2]=repo root
```

### Tool launches but crashes

Use the shared logger (never `print()`):

```python
import logging
logger = logging.getLogger(__name__)
logger.exception("Tool crashed: %s", e)
```

Run with `PYTHONPATH` set to see full tracebacks:
```bash
PYTHONPATH=src:src/shared/python python3 src/my_tool/launch_pyqt6.py
```
