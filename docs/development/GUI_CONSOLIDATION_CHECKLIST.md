# GUI Consolidation Checklist

This document tracks modules that have engines in the Tools shared folder but GUIs in other repositories, and need consolidation to enable single-repo development.

## Principle

**If a module's engine is in Tools/src/shared/, its GUI should also be in Tools.**

This enables:
- Single-repo development and troubleshooting
- Unified testing and CI/CD
- Consistent deployment
- Easier maintenance

## Migration Status

### 1. Electrode Advisor

| Component | Current Location | Target Location | Status |
|-----------|-----------------|-----------------|--------|
| Engine | `Tools/src/shared/python/upstream_drift_tools/calculators/electrical/` | N/A (already in place) | :white_check_mark: Complete |
| PyQt6 GUI | `Gasification_Model/src/integrated_process_simulator/electrode/` | `Tools/src/electrode_advisor/python/electrode_advisor/ui/pyqt6/` | :white_check_mark: Complete |
| React GUI | `Gasification_Model/frontend/src/features/calculators/calculators/ElectrodeAdvisorCalculator.tsx` | `Tools/src/electrode_advisor/web/` | :white_check_mark: Complete |
| Tests | Scattered | `Tools/src/electrode_advisor/tests/` | :white_check_mark: Complete |
| Launchers | N/A | `Tools/src/electrode_advisor/` | :white_check_mark: Complete |

**Files to migrate (PyQt6):**
- `refactored_electrode_advisor.py` - Main widget
- `controllers/main_controller.py` - Controller logic
- `models/` - Data models (config, calculation_results, input_data, etc.)
- `ui_components/` - UI panels and widgets
- `ui_builders/` - UI construction helpers
- `visualization/` - Chart and D3 rendering components
- `event_handlers/` - Input and interaction handling
- `calculation_engine/` - Data processing and calculations
- `configs/` - Color schemes and UI defaults

**Files to migrate (React):**
- `ElectrodeAdvisorCalculator.tsx` - Full calculator component

---

### 2. TRC Vessel Designer

| Component | Current Location | Target Location | Status |
|-----------|-----------------|-----------------|--------|
| Engine | `Tools/src/shared/python/upstream_drift_tools/calculators/mechanical/trc_geometry.py` | N/A (already in place) | :white_check_mark: Complete |
| PyQt6 GUI | Does not exist | `Tools/src/trc_vessel_designer/python/trc_vessel_designer/ui/pyqt6/` | :white_check_mark: Created |
| React GUI | `Gasification_Model/frontend/src/features/calculators/calculators/TRCVesselDesignerCalculator.tsx` | `Tools/src/trc_vessel_designer/web/` | :white_check_mark: Complete |
| Tests | N/A | `Tools/src/trc_vessel_designer/tests/` | :white_check_mark: Complete |
| Launchers | N/A | `Tools/src/trc_vessel_designer/` | :white_check_mark: Complete |

**Files to migrate (React):**
- `TRCVesselDesignerCalculator.tsx` - Full calculator component

**Files to create (PyQt6):**
- Main window with tabs
- Dimension input panel
- Refractory configuration panel
- Equipment configuration panel (burners, nozzles)
- Results display panel
- SVG/matplotlib vessel visualization

---

### 3. Syngas Compression Calculator

| Component | Current Location | Target Location | Status |
|-----------|-----------------|-----------------|--------|
| Engine | `Tools/src/shared/python/upstream_drift_tools/process_calculators/syngas_compression_calculator.py` | N/A (already in place) | :white_check_mark: Complete |
| PyQt6 GUI | `syngas_compression_calculator.py` (embedded) | `Tools/src/syngas_compression/launch_pyqt6.py` | :white_check_mark: Complete |
| React GUI | None | `Tools/src/syngas_compression/web/` | :white_check_mark: Complete |
| Tests | N/A | `Tools/src/syngas_compression/tests/` | :white_check_mark: Complete |
| Launchers | N/A | `Tools/src/syngas_compression/` | :white_check_mark: Complete |

**Features:**
- Multi-stage compression calculations
- Gas composition analysis with presets
- Water dropout calculations
- Temperature and pressure profile charts
- Process safety analysis and recommendations

---

### 4. Already Consolidated (No Action Needed)

These modules already have their GUIs co-located with engines in Tools:

| Module | Location | GUI Type |
|--------|----------|----------|
| Signal Toolkit | `Tools/src/shared/python/signal_toolkit/` | PyQt6 widgets |
| PSA Package | `Tools/src/shared/python/upstream_drift_tools/process_calculators/psa_package/` | PyQt6 GUI |
| Pressure Drop Calculator | `Tools/src/shared/python/upstream_drift_tools/process_calculators/pressure_drop_calculator/` | UI components |
| Data Processor | `Tools/src/data_processing/data_processor/` | PyQt6 + React |

---

## Migration Approach

### Phase 1: Directory Structure
1. Create module directories following existing patterns
2. Set up `__init__.py` files for Python imports
3. Create `package.json` for React apps

### Phase 2: Copy and Adapt
1. Copy source files maintaining original structure
2. Update import paths
3. Add backward-compatibility re-exports in original locations

### Phase 3: Testing
1. Write/migrate unit tests
2. Write integration tests
3. Verify launcher functionality

### Phase 4: Launchers
1. Create standalone launchers (launch_pyqt6.py, launch_web.py)
2. Register with GUI launcher system
3. Update tools.json

### Phase 5: Backward Compatibility
1. Create shim modules in original locations that re-export from Tools
2. Update documentation
3. Deprecation notices

---

## Reversibility Strategy

All migrations will maintain reversibility through:

1. **Re-export Shims**: Original import paths continue to work
2. **Version Tags**: Git tags before migration
3. **Feature Flags**: Environment variable to switch between old/new locations
4. **Documentation**: Clear rollback instructions

---

## Progress Tracking

- [x] Electrode Advisor PyQt6 GUI migration
- [x] Electrode Advisor React GUI migration
- [x] Electrode Advisor backward compatibility documentation
- [x] Electrode Advisor tests
- [x] TRC Vessel Designer React GUI migration
- [x] TRC Vessel Designer PyQt6 GUI creation
- [x] TRC Vessel Designer tests
- [x] tools.json updates
- [x] GUI registration updates
- [x] Final integration testing (pending PR/CI verification)
- [x] Syngas Compression Calculator PyQt6 launcher
- [x] Syngas Compression Calculator React GUI
- [x] Syngas Compression Calculator tests
