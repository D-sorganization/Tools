# Fleet-Wide Shared Tools Architecture

**Version**: 2.0
**Last Updated**: January 2026
**Status**: Production

---

## Overview

This document describes the shared tools architecture across the repository fleet. The architecture separates **core business logic** (which stays with each application) from **standalone utility tools** (which are shared).

## Architecture Principles

1. **Core logic stays local**: Thermodynamic engines, equilibrium solvers, and application-specific calculations remain in their respective repositories
2. **Standalone tools are shared**: Independent calculators that can work without application context are in the Tools repo
3. **Backward compatibility**: Shims allow gradual migration without breaking existing code

## Repository Fleet

| Repository             | Purpose                     | Relationship to Tools               |
| ---------------------- | --------------------------- | ----------------------------------- |
| **Tools**              | Shared utility library      | Source of standalone calculators    |
| **Gasification_Model** | Chemical process simulation | Core thermo + consumes shared tools |
| **UpstreamDrift**      | Biomechanical golf analysis | Consumes shared physics utilities   |

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TOOLS REPOSITORY                                   │
│                      (Standalone Utility Library)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   src/shared/python/upstream_drift_tools/                                    │
│   │                                                                          │
│   ├── process_calculators/     # STANDALONE PROCESS CALCULATORS              │
│   │   ├── acid_gas_dewpoint_calculator.py    # Acid gas dewpoint            │
│   │   ├── baghouse_calculator.py             # Baghouse filter sizing       │
│   │   ├── flare_calculator.py                # Flare system design          │
│   │   ├── scrubber_calculator.py             # Packed bed scrubber          │
│   │   ├── financial_calculator.py            # NPV/IRR analysis             │
│   │   ├── electrode_advancement_calculator.py # Electrode tracking          │
│   │   ├── ode_solver.py                      # Generic ODE solver           │
│   │   ├── thermal_profile_predictor.py       # Thermal analysis             │
│   │   ├── wgs_reactor_calculator.py          # Water-gas shift reactor      │
│   │   ├── syngas_water_calculator.py         # Syngas water content         │
│   │   ├── syngas_compression_calculator.py   # Compression analysis         │
│   │   ├── optimization.py                    # Optimization utilities       │
│   │   ├── multi_param_analysis.py            # Parameter sweeps             │
│   │   ├── constants.py                       # Physical constants           │
│   │   ├── pressure_drop_calculator/          # Pipe pressure drop           │
│   │   ├── psa_package/                       # PSA H2 separation            │
│   │   └── syngas_compression/                # Compression stages           │
│   │                                                                          │
│   ├── utils/                   # General utilities                           │
│   │   └── unit_constants.py    # NIST physical constants                    │
│   │                                                                          │
│   ├── signal_toolkit/          # SIGNAL PROCESSING LIBRARY                   │
│   │   ├── core.py              # Signal class and generator                 │
│   │   ├── fitting.py           # Curve fitting (sin, exp, poly, custom)     │
│   │   ├── filters.py           # Digital filters (Butterworth, etc.)        │
│   │   ├── calculus.py          # Differentiation and integration            │
│   │   ├── noise.py             # Noise generation (white, pink, brown)      │
│   │   ├── limits.py            # Saturation, rate limiting, deadband        │
│   │   ├── io.py                # Import/export (CSV, JSON, MAT, NPZ)        │
│   │   ├── widget.py            # PyQt6 interactive visualization            │
│   │   └── polynomial_generator.py  # Interactive polynomial generator       │
│   │                                                                          │
│   └── calculators/             # Legacy location (being migrated)            │
│                                                                              │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  │ Import as dependency
                                  │
┌─────────────────────────────────┴───────────────────────────────────────────┐
│                        GASIFICATION_MODEL                                    │
│                   (Chemical Process Simulator)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  src/integrated_process_simulator/                                           │
│  │                                                                           │
│  ├── core/                         # CORE CALCULATION ENGINES                │
│  │   ├── equilibrium_solver.py     # Gibbs minimization (STAYS HERE)        │
│  │   ├── energy_balance.py         # Energy balance engine (STAYS HERE)     │
│  │   ├── unified_gibbs_minimizer.py                                         │
│  │   └── calculation_engines/      # Enthalpy, etc.                         │
│  │                                                                           │
│  ├── calculators/                                                            │
│  │   ├── thermodynamic_properties/ # CORE THERMO ENGINE (STAYS HERE)        │
│  │   │   ├── core.py               # Main thermo calculator                 │
│  │   │   ├── optimized_core.py     # Performance-optimized version          │
│  │   │   ├── species_database.py   # Species property database              │
│  │   │   └── engines/              # Backend engines (Cantera, CoolProp)    │
│  │   │                                                                       │
│  │   ├── gasification_solver.py    # Main solver (STAYS HERE)               │
│  │   ├── quench_components/        # Quench calculator (STAYS HERE)         │
│  │   ├── heating_value_calculator.py                                        │
│  │   └── standalone_shims.py       # Backward compat for migrated calcs     │
│  │                                                                           │
│  └── tools/                        # Backward compatibility shims            │
│      ├── thermo/                   # Shim → thermodynamic_properties         │
│      ├── unit_converter/           # Shim → Tools repo or local             │
│      └── steam/                    # Steam calculations                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Shared Tools (Tools Repo)

### Process Calculators (`process_calculators/`)

These are standalone calculators that can work independently:

| Calculator                       | Purpose                              | Dependencies            |
| -------------------------------- | ------------------------------------ | ----------------------- |
| `AcidGasDewpointCalculator`      | HF, HCl, H2S dewpoint                | numpy, pandas           |
| `BaghouseCalculator`             | Filter sizing, drum fill time        | numpy (optional thermo) |
| `FlareCalculator`                | Flare system design, radiation zones | math only               |
| `ScrubberCalculator`             | Packed bed scrubber design           | numpy                   |
| `FinancialCalculator`            | NPV, IRR, payback analysis           | numpy                   |
| `ElectrodeAdvancementCalculator` | Arc furnace electrode tracking       | None                    |
| `ODESolver`                      | Generic ODE system solver            | scipy, sympy            |
| `ThermalProfilePredictor`        | Thermal transient analysis           | scipy                   |
| `WGSReactorCalculator`           | Water-gas shift equilibrium          | numpy, scipy            |
| `SyngasWaterCalculator`          | Water saturation in syngas           | numpy, scipy            |
| `SyngasCompressionCalculator`    | Multi-stage compression              | PyQt6 (optional)        |
| `PressureDropCalculator`         | Pipe flow pressure drop              | numpy                   |
| `PSAModel`                       | Pressure swing adsorption            | numpy                   |

**Usage:**

```python
from upstream_drift_tools.process_calculators import (
    FlareCalculator,
    ScrubberCalculator,
    FinancialCalculator,
)

# These work standalone without Gasification_Model
flare = FlareCalculator()
design = flare.calculate_flare_size(
    total_flow=1000,  # kg/hr
    gas_composition={"H2": 50, "CO": 30, "CH4": 20},
    temperature=500,  # K
    pressure=1.5,  # bar
)
```

### Signal Toolkit (`signal_toolkit/`)

A comprehensive signal processing library for control systems, simulation, and data analysis:

| Module                                   | Purpose                                                    | Dependencies      |
| ---------------------------------------- | ---------------------------------------------------------- | ----------------- |
| `Signal`, `SignalGenerator`              | Signal creation (13 types)                                 | numpy             |
| `FunctionFitter`                         | Curve fitting (sinusoid, exponential, polynomial, custom)  | numpy, scipy      |
| `FilterDesigner`                         | Digital filters (Butterworth, Chebyshev, Bessel, adaptive) | scipy             |
| `Differentiator`, `Integrator`           | Calculus operations                                        | numpy, scipy      |
| `NoiseGenerator`                         | Noise generation (white, pink, brown, blue, violet)        | numpy             |
| `apply_saturation`, `apply_rate_limiter` | Limits and constraints                                     | numpy             |
| `SignalImporter`, `SignalExporter`       | File I/O (CSV, JSON, MAT, NPZ)                             | numpy, scipy      |
| `PolynomialGeneratorWidget`              | Interactive polynomial fitting                             | PyQt6, sympy      |
| `SignalToolkitWidget`                    | Interactive signal visualization                           | PyQt6, matplotlib |

**Usage:**

```python
from signal_toolkit import Signal, SignalGenerator, FunctionFitter, apply_filter

# Generate a noisy sinusoid
import numpy as np
t = np.linspace(0, 10, 1000)
signal = SignalGenerator.sinusoid(t, amplitude=1.0, frequency=2.0)

# Fit a function to data
fitter = FunctionFitter()
result = fitter.auto_fit(signal)  # Tries multiple models, returns best
print(f"Best fit: {result.fit_type}, R-squared: {result.r_squared:.4f}")

# Apply a low-pass filter
from signal_toolkit import create_butterworth_filter
filter_spec = create_butterworth_filter('lowpass', cutoff=5, fs=100, order=4)
filtered = apply_filter(signal, filter_spec)
```

## Core Logic (Gasification_Model)

These components remain in Gasification_Model because they are core business logic:

| Module                        | Purpose                      | Why It Stays               |
| ----------------------------- | ---------------------------- | -------------------------- |
| `thermodynamic_properties/`   | Thermo property calculations | Core to equilibrium solver |
| `gasification_solver.py`      | Main Gibbs minimization      | Primary application logic  |
| `quench_components/`          | Quench system calculations   | Coupled to solver          |
| `heating_value_calculator.py` | HHV/LHV calculations         | Integrated with thermo     |
| `energy_balance.py`           | Energy balance engine        | Core calculation           |

## Backward Compatibility

### Shim Pattern

For migrated calculators, backward compatibility is maintained via shims:

```python
# src/integrated_process_simulator/calculators/standalone_shims.py

# This allows old imports to continue working:
from integrated_process_simulator.calculators.flare_calculator import FlareCalculator
# Actually imports from: upstream_drift_tools.process_calculators
```

### Thermodynamic Shims

The `tools/thermo/` directory in Gasification_Model contains shims pointing to the canonical location:

```python
# src/tools/thermo/__init__.py
"""Backward compatibility shim - actual code in calculators/thermodynamic_properties/"""
from integrated_process_simulator.calculators.thermodynamic_properties import *
```

## Migration Guide

### For New Code

Import directly from the canonical location:

```python
# Standalone tools
from upstream_drift_tools.process_calculators import FlareCalculator

# Core thermo (in Gasification_Model)
from integrated_process_simulator.calculators.thermodynamic_properties import (
    GasStream,
    get_optimized_thermodynamic_calculator,
)
```

### For Existing Code

Existing imports continue to work via shims, but will show deprecation warnings:

```python
# These still work but are deprecated:
from tools.thermo import ThermodynamicCalculator  # → shows warning
from integrated_process_simulator.calculators.flare_calculator import FlareCalculator  # → shows warning
```

## Best Practices

### For Shared Tools

1. **No application-specific dependencies**: Shared tools should not import from `integrated_process_simulator` core
2. **Optional heavy dependencies**: Make PyQt6, CoolProp, Cantera optional with fallbacks
3. **Comprehensive constants**: Use `constants.py` for all physical constants
4. **Self-contained**: Each calculator should work independently

### For Core Logic

1. **Stay in application repo**: Core business logic belongs with the application
2. **Use canonical imports**: Import thermo from `calculators/thermodynamic_properties/`
3. **Avoid circular dependencies**: Use lazy imports where needed

## Testing

### Shared Tools Tests

```bash
cd Tools
pytest src/shared/python/upstream_drift_tools/process_calculators/
```

### Gasification_Model Tests

```bash
cd Gasification_Model
pytest tests/
```

## Related Documents

- [Gasification_Model README](../../../Gasification_Model/README.md)
- [UpstreamDrift Architecture](../../../UpstreamDrift/docs/architecture/)
