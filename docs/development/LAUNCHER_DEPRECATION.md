# Launcher Deprecation Notice

## Overview

The `launch_tools_main.py` launcher has been **deprecated** in favor of `UnifiedToolsLauncher.py`.

## Migration Guide

### Why the Change?

- **Modern Interface**: `UnifiedToolsLauncher.py` uses PyQt6, providing a more modern and responsive UI
- **Better Tool Discovery**: Uses the PluginManager system for automatic tool discovery
- **Improved Error Handling**: More robust error handling and user feedback
- **Consistency**: Aligns with repository standards and architecture
- **DRY Principle**: Uses shared utilities for path setup and configuration

### How to Migrate

1. **Update your scripts/aliases**:
   ```bash
   # Old (deprecated)
   python launch_tools_main.py
   
   # New (recommended)
   python UnifiedToolsLauncher.py
   ```

2. **Update documentation** that references `launch_tools_main.py`

3. **Update CI/CD workflows** if they use the old launcher

### Timeline

- **Current**: `launch_tools_main.py` is deprecated but still functional
- **Future**: `launch_tools_main.py` will be removed in a future release

### Backward Compatibility

The deprecated launcher will continue to work for backward compatibility, but:
- It shows a deprecation warning when used
- It uses shared utilities where possible to reduce duplication
- No new features will be added to it

## Questions?

If you have questions or concerns about this deprecation, please open an issue.
