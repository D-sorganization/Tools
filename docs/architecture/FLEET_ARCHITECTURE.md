# Fleet-Wide Shared Tools Architecture

**Version**: 1.0
**Last Updated**: January 2026
**Status**: Production

---

## Overview

This document describes the shared tools architecture across the D-sorganization repository fleet. The architecture enables code reuse, maintains consistency, and reduces duplication across multiple projects.

## Repository Fleet

| Repository | Purpose | Shared Tools Integration |
|------------|---------|-------------------------|
| **Tools** | Central shared library | Source of truth |
| **Gasification_Model** | Chemical process simulation | Consumer via `src/tools/` |
| **UpstreamDrift** | Biomechanical golf swing analysis | Consumer via shared utils |
| **AffineDrift** | Financial drift analysis | Standalone |

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TOOLS REPOSITORY                                 │
│                    (Central Shared Library)                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   upstream_drift_tools/                                                  │
│   ├── unit_converter/         # Unit conversion utilities                │
│   │   ├── __init__.py         # Main converter service                  │
│   │   └── constants.py        # Physical constants database             │
│   ├── thermo/                 # Thermodynamic calculations              │
│   │   ├── core.py             # Basic thermodynamic engine              │
│   │   ├── optimized_core.py   # Performance-optimized engine            │
│   │   ├── species_database.py # Chemical species properties             │
│   │   ├── math_utils.py       # Mathematical utilities                  │
│   │   └── widget.py           # PyQt6 UI components                     │
│   ├── steam/                  # Steam cycle calculations                │
│   │   ├── iapws.py            # IAPWS-IF97 steam tables                 │
│   │   └── steam_tables.py     # Convenience wrappers                    │
│   └── visualization/          # Shared plotting utilities               │
│       └── plotters.py         # Common chart types                      │
│                                                                          │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    │ Git submodule / Copy
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        │                                                       │
        ▼                                                       ▼
┌───────────────────────────────────┐   ┌───────────────────────────────────┐
│      GASIFICATION_MODEL           │   │         UPSTREAM_DRIFT            │
│   (Chemical Process Simulator)    │   │   (Biomechanical Golf Analysis)   │
├───────────────────────────────────┤   ├───────────────────────────────────┤
│                                   │   │                                   │
│  src/                             │   │  shared/                          │
│  ├── tools/         ◄── SYNCED   │   │  ├── python/                      │
│  │   ├── unit_converter/         │   │  │   ├── physics_constants.py    │
│  │   ├── thermo/                 │   │  │   └── interfaces.py            │
│  │   └── steam/                  │   │  └── data/                        │
│  └── integrated_process_simulator │   │                                   │
│      └── utilities/              │   │  engines/                          │
│          ├── unit_conversion/    │   │  ├── mujoco/                      │
│          │   └── __init__.py     │   │  ├── drake/                       │
│          │       (backward compat)│   │  ├── pinocchio/                   │
│          └── unit_constants.py   │   │  └── opensim/                     │
│              (backward compat)   │   │                                   │
│                                   │   │                                   │
└───────────────────────────────────┘   └───────────────────────────────────┘
```

## Shared Tools Modules

### 1. Unit Converter (`unit_converter/`)

**Purpose**: Comprehensive unit conversion service for scientific computing.

**Key Components**:
- `UnitConversionService`: Main service class with all conversion methods
- `constants.py`: Physical constants (R, kB, NA, etc.) and conversion factors

**Usage**:
```python
from tools.unit_converter import UnitConversionService, PHYSICAL_CONSTANTS

converter = UnitConversionService()
temp_kelvin = converter.celsius_to_kelvin(25.0)
```

### 2. Thermodynamic Calculator (`thermo/`)

**Purpose**: High-performance thermodynamic property calculations for gas mixtures.

**Key Components**:
- `OptimizedThermodynamicCalculator`: Vectorized calculations with caching
- `GasStream`: Data model for gas stream properties
- `SpeciesDatabase`: Chemical species properties (Cp, Hf, S, etc.)
- `ThermodynamicPropertiesWidget`: PyQt6 UI component

**Usage**:
```python
from tools.thermo import (
    GasStream,
    FlowUnit,
    get_optimized_thermodynamic_calculator
)

calc = get_optimized_thermodynamic_calculator()
stream = GasStream(
    flow_rate=100.0,
    flow_unit=FlowUnit.MASS,
    temperature=500.0,
    pressure=101325.0,
    composition={"CO": 0.3, "H2": 0.5, "CO2": 0.2}
)
props = calc.calculate_stream_properties(stream)
```

### 3. Steam Engine (`steam/`)

**Purpose**: IAPWS-IF97 steam tables and Rankine cycle calculations.

**Key Components**:
- `iapws.py`: Core IAPWS-IF97 implementation
- `SteamTables`: High-level API for steam properties
- Rankine cycle analysis utilities

**Usage**:
```python
from tools.steam import SteamTables

steam = SteamTables()
props = steam.get_properties(temperature=400.0, pressure=1000.0)
```

## Backward Compatibility Architecture

When migrating modules to the shared tools library, we maintain backward compatibility through import shims:

```
Old Import Path                          New Import Path
─────────────────                        ─────────────────
integrated_process_simulator             tools.unit_converter
  .utilities.unit_conversion      ───►

integrated_process_simulator             tools.unit_converter.constants
  .utilities.unit_constants       ───►

integrated_process_simulator             tools.thermo
  .calculators.thermodynamic_properties
                                  ───►
```

**Shim Implementation Pattern**:

```python
# src/integrated_process_simulator/utilities/unit_conversion/__init__.py
"""Backward compatibility shim for unit_conversion module.

This module has been moved to tools.unit_converter.
This shim provides backward compatibility for existing imports.
"""

# Re-export all public symbols from the new location
from tools.unit_converter import *  # noqa: F401, F403
```

This pattern:
1. Preserves existing import statements in consumer code
2. Allows gradual migration without breaking changes
3. Enables deprecation warnings when needed
4. Maintains single source of truth in `tools/`

## Migration Status

### Completed Migrations

| Module | Old Location | New Location | Status |
|--------|-------------|--------------|--------|
| Unit Converter | `utilities.unit_conversion` | `tools.unit_converter` | ✅ Complete |
| Unit Constants | `utilities.unit_constants` | `tools.unit_converter.constants` | ✅ Complete |
| Thermodynamic Calculator | `calculators.thermodynamic_properties` | `tools.thermo` | ✅ Complete |
| Species Database | `calculators.thermodynamic_properties` | `tools.thermo.species_database` | ✅ Complete |

### Candidates for Future Migration

| Module | Current Location | Priority | Notes |
|--------|------------------|----------|-------|
| Steam Tables | `tools.steam` | Low | Already in tools |
| Plotting Utilities | Various | Medium | Needs consolidation |
| Data Validation | Various | Medium | Common patterns exist |

## Testing Strategy

### Shared Tools Tests

Located in `tests/` within Tools repository:
- Unit tests for each module
- Integration tests for cross-module functionality
- Performance benchmarks

### Consumer Tests

Each consuming repository maintains tests that:
1. Verify shim imports work correctly
2. Test integration with shared tools
3. Mock shared tools for isolated unit tests

**Example Mock Pattern**:
```python
# test_service_registry.py
unit_conversion = types.ModuleType("tools.unit_converter")
unit_conversion.UnitConversionService = _DummyClass
monkeypatch.setitem(sys.modules, "tools.unit_converter", unit_conversion)
```

## Development Workflow

### Making Changes to Shared Tools

1. **Develop in Tools repo**: Make changes in `upstream_drift_tools/`
2. **Test locally**: Run `pytest tests/`
3. **Create PR**: Submit changes for review
4. **Sync consumers**: After merge, update consuming repos

### Syncing Changes to Consumers

```bash
# In Gasification_Model repository
cd src/tools
git pull origin main  # If using submodule

# Or manually copy updated files
cp -r /path/to/Tools/upstream_drift_tools/* ./
```

## Best Practices

### For Shared Tools

1. **No application-specific logic**: Keep tools generic and reusable
2. **Comprehensive docstrings**: Document all public APIs
3. **Type hints**: Use typing throughout
4. **Backward compatibility**: Never break existing APIs without deprecation period
5. **Performance**: Optimize for common use cases

### For Consumers

1. **Use canonical imports**: Import from `tools.*` not from shims
2. **Don't modify shared tools locally**: Changes go through Tools repo
3. **Document dependencies**: List shared tools requirements
4. **Test with mocks**: Don't couple tests to shared tools implementation

## Architecture Decision Records

### ADR-001: Shared Tools in Tools Repository

**Decision**: Maintain shared tools in the Tools repository as the single source of truth.

**Rationale**:
- Centralized maintenance
- Clear ownership
- Easier versioning
- Consistent testing

### ADR-002: Backward Compatibility Shims

**Decision**: Use import shims during migration rather than immediate breaking changes.

**Rationale**:
- Allows gradual migration
- Reduces risk of breaking existing code
- Enables deprecation warnings
- Supports parallel development

### ADR-003: Direct Imports in Core Logic

**Decision**: Core calculation engines should import directly from `tools.*` not through shims.

**Rationale**:
- Cleaner dependency graph
- Better for testing/mocking
- Avoids double-indirection
- Makes true dependencies visible

## Monitoring and Maintenance

### Deprecation Timeline

1. **Phase 1**: Add shim with no warning (current)
2. **Phase 2**: Add deprecation warning (6 months)
3. **Phase 3**: Remove shim, breaking change (12 months)

### Health Checks

- CI runs tests across all consuming repos
- Dependency audits track shared tools versions
- Performance benchmarks detect regressions

---

## Related Documents

- [Tools Repository README](../../README.md)
- [Gasification Model Architecture](../../../Linux_Gasification_Model/Gasification_Model/docs/architecture/)
- [UpstreamDrift Architecture](../../../Linux_Golf_Modeling_Suite/Golf_Modeling_Suite/docs/architecture/)
