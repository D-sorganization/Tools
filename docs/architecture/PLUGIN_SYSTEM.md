# Plugin System Documentation

## Overview

The Tools repository supports two methods for tool registration:

1. **Centralized Registration** (`tools.json`) - Manual, explicit tool registration
2. **Automatic Discovery** (`tool_manifest.json`) - Automatic tool discovery via manifest files

## Automatic Tool Discovery

Tools can be automatically discovered by placing a `tool_manifest.json` file in their directory. This eliminates the need to manually edit `tools.json` when adding new tools.

### Creating a Tool Manifest

Create a `tool_manifest.json` file in your tool's directory:

```json
{
  "name": "My Awesome Tool",
  "path": "main.py",
  "type": "python",
  "description": "A tool that does amazing things",
  "category": "Development Tools"
}
```

### Manifest Fields

- **`name`** (required): Display name for the tool
- **`path`** (optional): Relative path to the tool's entry point. If omitted, the system will search for `*.py` files in the directory
- **`type`** (optional, default: `"python"`): Tool type (`python`, `matlab`, `web`, `browser`, `bat`)
- **`description`** (optional): Tool description shown in the launcher
- **`category`** (optional, default: `"Development Tools"`): Category for grouping tools

### Example: Folder Tool Manifest

```json
{
  "name": "Folder Packer Pro",
  "path": "folder_packer_pro.py",
  "type": "python",
  "description": "Project archiving and distribution tool",
  "category": "Development Tools"
}
```

## Using the Plugin Manager

The `PluginManager` class provides methods for tool discovery:

```python
from pathlib import Path
from python.src.core.plugin_manager import PluginManager

repo_root = Path(__file__).parent
manager = PluginManager(repo_root)

# Load tools with automatic discovery
tools = manager.load_tools_with_discovery()

# Or scan for tools only
discovered = manager.scan_for_tools()
```

## Migration Path

1. **Existing tools**: Continue using `tools.json` (fully supported)
2. **New tools**: Add `tool_manifest.json` for automatic discovery
3. **Gradual migration**: Both methods work together - discovered tools are merged with `tools.json` entries

## Benefits

- **No manual registration**: Tools are discovered automatically
- **Self-documenting**: Manifest files document tool metadata
- **Version control friendly**: Each tool manages its own manifest
- **Backward compatible**: Existing `tools.json` continues to work

## Future Enhancements

- Support for `tool_manifest.yaml` format
- Automatic icon discovery
- Dependency declaration in manifests
- Tool versioning and compatibility checks
