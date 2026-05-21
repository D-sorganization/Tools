# Namespace Collision Report — Issue #2405

**Date:** 2026-04-30
**Focus:** Analysis of duplicate filenames across the Tools monorepo

---

## Summary

| Collision Type      | Count  | Risk Level | Status                     |
| ------------------- | ------ | ---------- | -------------------------- |
| launch_pyqt6.py     | 21     | LOW        | Safe (tool-scoped)         |
| gui_registration.py | 21     | LOW        | Safe (manifest replaces)   |
| core.py             | 5      | LOW        | Safe (package-scoped)      |
| models.py           | 3      | LOW        | Safe (package-scoped)      |
| main_window.py      | 19     | NONE       | Safe (deeply nested)       |
| **TOTAL**           | **69** | **LOW**    | **No breaking collisions** |

---

## Critical Duplicates

### 1. launch_pyqt6.py (21 occurrences)

**Locations:**

```
src/c3d_viewer/launch_pyqt6.py
src/data_processing/data_processor/launch_pyqt6.py
src/document_processing/pdf_renamer/launch_pyqt6.py
src/financial_calculator/launch_pyqt6.py
src/flow_rate_converter/launch_pyqt6.py
src/folder_packer_pro/launch_pyqt6.py
src/folder_tool/launch_pyqt6.py
src/function_generator/launch_pyqt6.py
src/humanoid_builder_gui/launch_pyqt6.py
src/inertia_calculator/launch_pyqt6.py
src/lower_body_model/launch_pyqt6.py
src/multi_param_analysis/launch_pyqt6.py
src/ode_solver/launch_pyqt6.py
src/optimizer_gui/launch_pyqt6.py
src/pid_generator/launch_pyqt6.py
src/pressure_drop_calculator/launch_pyqt6.py
src/rotation_converter/launch_pyqt6.py
src/signal_processing_studio/launch_pyqt6.py
src/steam_engine_calculator/launch_pyqt6.py
src/urdf_builder_gui/launch_pyqt6.py
src/vessel_drafter/launch_pyqt6.py
```

**Purpose:** PyQt6 entry point for each tool's GUI

**Current Usage Pattern:**

```python
# Example: src/financial_calculator/launch_pyqt6.py
# Always imported with tool prefix:
from financial_calculator.launch_pyqt6 import launch_app
# NOT: from launch_pyqt6 import launch_app (would be ambiguous)
```

**Collision Risk:** NONE

- No direct imports of `launch_pyqt6` without tool prefix
- Each file is tool-specific and only referenced within tool context
- Manifest routing prevents dynamic import collisions

**Why Safe:**

```
✓ Import path includes tool name: financial_calculator.launch_pyqt6
✓ Not imported with `from ... import *`
✓ Manifest provides single dispatch point
✓ No cross-tool dependencies on launch_pyqt6
```

**Deprecation Timeline:**

- Phase 2: Keep as-is
- Phase 3: Mark deprecated, delegate to gui_launcher
- Phase 4: Remove completely

**Recommended Action:** KEEP FOR NOW, Deprecate in Phase 3

---

### 2. gui_registration.py (21 occurrences)

**Locations:** (same as launch_pyqt6.py)

```
src/c3d_viewer/gui_registration.py
src/financial_calculator/gui_registration.py
... (19 more)
```

**Purpose:** GUI metadata dictionary (LEGACY)

**Typical Contents:**

```python
# Example: src/financial_calculator/gui_registration.py
GUI_INFO = {
    "name": "Financial Calculator",
    "description": "...",
    "category": "...",
    "icon": "...",
    # ... more metadata
}
```

**Historical Issue:** Issue #1863

- This metadata was duplicated across 20 files
- Maintenance burden: changes required in 20 places
- Resolution: tool_manifest.yaml centralizes this data

**Current Status:** DEPRECATED (superseded by tool_manifest.yaml)

**Collision Risk:** NONE

- Not imported across tools
- Manifest is now single source of truth
- Files are legacy, not actively used

**Why Safe:**

```
✓ Each file is local to its tool
✓ Not shared between tools
✓ Manifest provides centralized replacement
✓ Code already reads from manifest (not local files)
```

**Deprecation Timeline:**

- Phase 2: Mark deprecated in comments
- Phase 3: Add deprecation warnings in code
- Phase 4: Remove completely

**Recommended Action:** DEPRECATE NOW, Remove in Phase 4

**Migration Path:**

```python
# OLD: Direct import from local file
from financial_calculator.gui_registration import GUI_INFO

# NEW: Read from manifest
from gui_launcher import GuiLauncher
launcher = GuiLauncher()
gui_info = launcher.get_tool_info('financial_calculator')
```

---

## Secondary Duplicates

### 3. core.py (5 occurrences)

**Locations:**

```
src/signal_toolkit/core.py
src/rotation_converter/core.py
src/shared/python/upstream_drift_tools/calculators/conversion/core.py
src/shared/python/upstream_drift_tools/data_processing/core.py
src/document_processing/pdf_renamer/src/pdf_renamer/core.py
```

**Purpose:** Core functionality module (different purpose in each)

**Import Analysis:**

```python
# signal_toolkit.core
from signal_toolkit.core import FFT, Spectrogram

# rotation_converter.core
from rotation_converter.core import RotationMatrix

# upstream_drift_tools (nested)
from upstream_drift_tools.calculators.conversion.core import ConversionCalculator
from upstream_drift_tools.data_processing.core import DataProcessor

# pdf_renamer.core
from pdf_renamer.core import PDFRenamer
```

**Collision Risk:** VERY LOW

- Each `core.py` is scoped within its package hierarchy
- No package-level imports of bare `core` module
- Fully qualified names prevent ambiguity

**Why Safe:**

```
✓ No `import core` statements found
✓ All imports use package prefix
✓ Package hierarchy provides natural scoping
✓ No sys.path manipulation to cause shadowing
```

**Recommended Action:** NO ACTION NEEDED

- This is normal Python practice
- Scoping prevents collisions
- No refactoring required

---

### 4. models.py (3 occurrences)

**Locations:**

```
src/python/src/tile_launcher/models.py
src/shared/python/chat/models.py
src/shared/python/notes/models.py
```

**Purpose:** Data models (Django/Pydantic pattern)

**Import Patterns:**

```python
# Each imported with package prefix:
from tile_launcher.models import LauncherModel
from chat.models import Message, Conversation
from notes.models import Note
```

**Collision Risk:** VERY LOW

- Each in separate package
- No cross-imports between tile_launcher, chat, notes
- Standard naming convention across web frameworks

**Why Safe:**

```
✓ Separate packages (tile_launcher, chat, notes)
✓ No shared imports
✓ Common pattern in Django/FastAPI projects
✓ No ambiguity in imports
```

**Recommended Action:** NO ACTION NEEDED

- This is idiomatic Python
- No refactoring beneficial

---

### 5. main_window.py (19 occurrences)

**Locations:**

```
src/asteroid_jumper/main_window.py
src/c3d_viewer/python/c3d_viewer/ui/pyqt6/main_window.py
src/data_processing/data_processor/python/data_processor/ui/pyqt6/main_window.py
src/financial_calculator/python/financial_calculator/ui/pyqt6/main_window.py
src/flow_rate_converter/python/flow_rate_converter/ui/pyqt6/main_window.py
src/function_generator/python/function_generator/ui/pyqt6/main_window.py
src/humanoid_builder_gui/python/humanoid_builder_gui/ui/pyqt6/main_window.py
src/inertia_calculator/python/inertia_calculator/ui/pyqt6/main_window.py
src/multi_param_analysis/python/multi_param_analysis/ui/pyqt6/main_window.py
src/ode_solver/python/ode_solver/ui/pyqt6/main_window.py
src/optimizer_gui/python/optimizer_gui/ui/pyqt6/main_window.py
src/pendulum_simulator/src/double_pendulum_golf/gui/main_window.py
src/pid_generator/ui/pyqt6/main_window.py
src/pressure_drop_calculator/python/pressure_drop_calculator/ui/pyqt6/main_window.py
src/rotation_converter/ui/pyqt6/main_window.py
src/shared/python/upstream_drift_tools/process_calculators/psa_package/ui/main_window.py
src/signal_processing_studio/python/signal_processing_studio/main_window.py
src/steam_engine_calculator/python/steam_engine_calculator/ui/pyqt6/main_window.py
src/urdf_builder_gui/python/urdf_builder_gui/ui/pyqt6/main_window.py
```

**Purpose:** PyQt6 main window class for each tool

**Import Patterns:**

```python
# Each tool imports its own main_window:
from c3d_viewer.python.c3d_viewer.ui.pyqt6.main_window import C3DViewerWindow
from financial_calculator.python.financial_calculator.ui.pyqt6.main_window import FinancialCalculatorMainWindow

# OR (for tools with different paths):
from signal_processing_studio.python.signal_processing_studio.main_window import SignalProcessingStudioWindow
```

**Collision Risk:** NO RISK

- Each main_window.py is at a deeply nested path unique to its tool
- Never imported without full qualification
- Manifest specifies exact module path for each tool

**Why Safe:**

```
✓ Deeply nested in tool-specific paths
✓ Full paths are unique per tool
✓ No ambiguous imports
✓ Standard pattern across PyQt6 applications
✓ Manifest explicitly lists exact module path
```

**Example from manifest:**

```yaml
- tool_name: financial_calculator
  pyqt6:
    module: financial_calculator.ui.pyqt6.main_window
    class: FinancialCalculatorMainWindow
```

**Recommended Action:** NO ACTION NEEDED

- This is correct encapsulation
- No refactoring required
- Pattern is ideal for monorepo

---

## Impact Assessment

### Collision Severity Matrix

| File                | Count | Severity      | Risk | Action            |
| ------------------- | ----- | ------------- | ---- | ----------------- |
| launch_pyqt6.py     | 21    | INFORMATIONAL | LOW  | Deprecate Phase 3 |
| gui_registration.py | 21    | INFORMATIONAL | NONE | Deprecate Phase 2 |
| core.py             | 5     | NORMAL        | LOW  | No action         |
| models.py           | 3     | NORMAL        | LOW  | No action         |
| main_window.py      | 19    | NORMAL        | NONE | No action         |

### Breaking Change Risk: NONE

- No actual collisions preventing code execution
- All imports work correctly
- No refactoring needed to maintain functionality
- Manifest already centralized problematic data

### CI/CD Impact: NONE

- Tests pass without modification
- Linting passes without modification
- Type checking passes without modification
- No changes required to CI pipeline

---

## Recommendations

### Phase 2 (Immediate)

1. **No urgent action needed** — collisions are safe and scoped
2. **Document the structure** — already done (TOOL_STRUCTURE.md)
3. **Mark gui_registration.py deprecated** — add comment at top of files:
   ```python
   # DEPRECATED: This file is superseded by tool_manifest.yaml
   # See: src/shared/python/gui_launcher/tool_manifest.yaml
   # Phase 4: This file will be removed
   ```

### Phase 3 (Future)

4. **Standardize tool structure** — ensure all tools follow pattern:
   - launch_pyqt6.py at root
   - gui_registration.py at root (marked deprecated)
   - python/<tool_name>/ for implementation
5. **Add manifest validation to CI**:

   ```bash
   # Validate all manifest entries reference real modules
   python3 -c "
   import yaml
   from importlib import import_module

   with open('src/shared/python/gui_launcher/tool_manifest.yaml') as f:
       manifest = yaml.safe_load(f)

   for tool in manifest['tools']:
       if 'pyqt6' in tool:
           module_path = tool['pyqt6']['module']
           class_name = tool['pyqt6']['class']
           try:
               mod = import_module(module_path)
               assert hasattr(mod, class_name)
           except (ImportError, AssertionError) as e:
               print(f'ERROR: {tool[\"tool_name\"]}: {e}')
   "
   ```

### Phase 4 (Later)

6. **Remove deprecated files**:
   - Delete all gui_registration.py (after 6+ month deprecation period)
   - Delete launch_pyqt6.py (or refactor to delegate to manifest)
   - Simplify tool loading code

---

## Conclusion

**Overall Assessment:** No Breaking Collisions

The 69 duplicate filenames in the Tools monorepo do not constitute actual namespace collisions because:

1. **Tool-scoped imports:** All duplicates are imported with tool-specific prefixes
2. **Manifest centralization:** Metadata duplication (gui_registration.py) already resolved by tool_manifest.yaml
3. **Proper nesting:** Files like core.py and main_window.py are deeply scoped within package hierarchies
4. **No cross-tool dependencies:** No tool imports implementation files from other tools (only from shared/python/)

**Collision files are safe to keep** during Phase 2-3 refactoring. Deprecation and removal are optimization steps, not functional requirements.

**Next Steps:**

1. Proceed with Phase 2.2 (register lower_body_model)
2. Deploy deprecation notices in Phase 2.3
3. Plan structural improvements for Phase 3+

See TOOL_STRUCTURE.md and AUDIT_REPORT_2405.md for complete refactoring roadmap.
