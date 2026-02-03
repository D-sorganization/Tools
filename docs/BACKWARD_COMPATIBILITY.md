# Backward Compatibility Guide for GUI Consolidation

This document describes how to maintain backward compatibility when migrating GUI components from Gasification_Model to Tools.

## Overview

When GUIs are consolidated into Tools, the original import paths in Gasification_Model should continue to work. This is achieved through re-export shims.

## Electrode Advisor

### New Canonical Location (Tools)
```python
# Import from Tools (preferred)
from electrode_advisor import ElectrodeAdvisorWidget
from electrode_advisor import ElectrodeConfig, GlassPropertiesInterface
```

### Backward Compatibility Shim for Gasification_Model

Add this to `Gasification_Model/src/integrated_process_simulator/electrode/__init__.py`:

```python
"""Electrode Package - Backward Compatibility Shim.

This package now re-exports from Tools/src/electrode_advisor/.
The canonical location is Tools, but these imports continue to work.
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from integrated_process_simulator.electrode is deprecated. "
    "Import from electrode_advisor instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from Tools
try:
    from electrode_advisor import (
        ElectrodeAdvisorWidget,
    )
    from shared.python.upstream_drift_tools.calculators.electrical import (
        ElectrodeConfig,
        GlassPropertiesInterface,
        ThreePhaseElectricalModelEnhanced,
    )
except ImportError:
    # Fallback to local if Tools not available
    from .refactored_electrode_advisor import (
        RefactoredElectrodeAdvisorWidget as ElectrodeAdvisorWidget,
    )
    from .models import (
        ElectrodeConfig,
        GlassPropertiesInterface,
        ThreePhaseElectricalModelEnhanced,
    )

__all__ = [
    "ElectrodeAdvisorWidget",
    "ElectrodeConfig",
    "GlassPropertiesInterface",
    "ThreePhaseElectricalModelEnhanced",
]
```

## TRC Vessel Designer

### New Canonical Location (Tools)
```python
# Import from Tools (preferred)
from trc_vessel_designer import TRCVesselDesignerWidget
from trc_vessel_designer import TRCGeometryEngine, VesselDimensions
```

### Note on TRC Vessel Designer
The TRC Vessel Designer PyQt6 GUI is a new creation in Tools. No backward compatibility shim is needed as there was no previous PyQt6 GUI in Gasification_Model.

For the React component, update imports in Gasification_Model to point to the shared component.

## Environment Setup

To use the consolidated GUIs, ensure Tools is in your Python path:

```python
import sys
sys.path.insert(0, "/path/to/Tools/src")
sys.path.insert(0, "/path/to/Tools/src/shared/python")
```

Or set `PYTHONPATH`:
```bash
export PYTHONPATH=/path/to/Tools/src:/path/to/Tools/src/shared/python:$PYTHONPATH
```

## Reversibility

All migrations are reversible:

1. **Git tags**: Tag the repository before migration
2. **Shim removal**: Remove the deprecation warning and change imports back
3. **Feature flag**: Use environment variable to switch:

```python
import os

if os.environ.get("USE_LEGACY_ELECTRODE_GUI", "false").lower() == "true":
    from .refactored_electrode_advisor import RefactoredElectrodeAdvisorWidget as ElectrodeAdvisorWidget
else:
    from electrode_advisor import ElectrodeAdvisorWidget
```

## Testing Compatibility

Run tests to verify backward compatibility:

```bash
# Test Tools imports
python -c "from electrode_advisor import ElectrodeAdvisorWidget; print('OK')"

# Test legacy imports (after shim is in place)
python -c "from integrated_process_simulator.electrode import ElectrodeAdvisorWidget; print('OK')"
```
