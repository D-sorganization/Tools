# PSA Package - Pressure Swing Adsorption System Modeling

A comprehensive two-stage Pressure Swing Adsorption (PSA) system modeling tool for hydrogen purification and gas separation processes. This package provides bed sizing calculations, cycle optimization, mass balance analysis, and safety assessment capabilities.

## Purpose

The PSA Package enables process engineers to:

- Model two-stage PSA systems with recycle streams
- Calculate hydrogen recovery and purity for various operating conditions
- Perform sensitivity analysis on recycle fractions
- Assess oxygen concentration safety in tail gas streams
- Optimize cycle parameters for maximum efficiency

## Key Features

- **Algebraic Mass Balance Solution**: Eliminates circular references in recycle calculations
- **Multi-Component Tracking**: Supports H2, CO, CO2, H2O, N2, O2, and CH4
- **Sensitivity Analysis**: Evaluates performance across recycle fraction ranges
- **O2 Safety Analysis**: Monitors flammability limits in recycle streams
- **Multiple Interfaces**: PyQt6 desktop GUI, Streamlit web app, and standalone HTML calculator
- **Pre-calculated Plots**: Immediate visualization without computation delay

## Installation / Prerequisites

### Required Dependencies

```bash
pip install numpy matplotlib PyQt6
```

### Optional Dependencies

```bash
# For web application
pip install streamlit plotly pandas

# For Jupyter notebooks
pip install jupyter
```

### Python Version

- Python 3.10 or higher recommended

## Usage Instructions

### Desktop GUI Application

```bash
python launch_pyqt6.py
```

### Web Application

```bash
python launch_web.py
# Opens Streamlit app in default browser
```

### Programmatic Usage

```python
from upstream_drift_tools.process_calculators.psa_package import PSAModel

model = PSAModel(
    total_feed_scfm=1100.0,
    s2_tail_recycle_frac=1.0,
    product_recycle_frac=0.0,
)
results = model.calculate()

print(f"H2 Recovery: {results.h2_recovery_pct:.2f}%")
print(f"H2 Purity: {results.h2_purity_pct:.4f}%")
print(f"Net Product: {results.total_net_product_scfm:.1f} SCFM")
```

## Input Parameters

| Parameter              | Description                   | Range      | Units    | Default |
| ---------------------- | ----------------------------- | ---------- | -------- | ------- |
| `total_feed_scfm`      | Total fresh feed flow rate    | 0 - 10,000 | SCFM     | 1100.0  |
| `s2_tail_recycle_frac` | Stage 2 tail recycle fraction | 0.0 - 1.0  | fraction | 1.0     |
| `product_recycle_frac` | Product recycle fraction      | 0.0 - 1.0  | fraction | 0.0     |
| `feed_pct`             | Component feed composition    | 0 - 100    | %        | varies  |
| `stage1_removal_pct`   | Stage 1 removal efficiency    | 0 - 100    | %        | varies  |
| `stage2_removal_pct`   | Stage 2 removal efficiency    | 0 - 100    | %        | varies  |

### Default Component Data

| Component | Feed % | Stage 1 Removal % | Stage 2 Removal % |
| --------- | ------ | ----------------- | ----------------- |
| H2        | 32.08  | 18.0              | 15.0              |
| CO        | 38.22  | 98.0              | 99.99             |
| CO2       | 21.98  | 98.0              | 99.99             |
| H2O       | 4.85   | 99.0              | 99.99             |
| N2        | 0.50   | 95.0              | 99.99             |
| O2        | 0.50   | 81.0              | 99.99             |
| CH4       | 1.88   | 99.0              | 99.99             |

## Output Format

The `PSAResults` dataclass contains:

```python
@dataclass
class PSAResults:
    component_names: list[str]      # Component identifiers
    flows: StreamFlows              # Mass flows for all streams (SCFM)
    compositions: StreamCompositions # Compositions for all streams (%)
    h2_recovery_pct: float          # H2 recovery efficiency (%)
    h2_purity_pct: float            # H2 purity in net product (%)
    total_feed_scfm: float          # Total feed flow (SCFM)
    total_net_product_scfm: float   # Net product flow (SCFM)
    total_exhaust_scfm: float       # Exhaust flow (SCFM)
    total_s2_tail_vent_scfm: float  # S2 tail vent flow (SCFM)
    mass_balance_error: float       # Mass balance closure error
    s2_tail_h2_pct: float           # H2 in S2 tail (%)
    s2_tail_o2_pct: float           # O2 in S2 tail (%)
```

## Mathematical Models

### Key Mass Balance Equation

The algebraic solution for mixed feed flow eliminates circular references:

```
M_i = F_i / [1 - (1 - R1_i) * (R2_i * r_tail + (1 - R2_i) * r_prod)]
```

Where:

- `M_i` = Mixed feed flow for component i (SCFM)
- `F_i` = Fresh feed flow for component i (SCFM)
- `R1_i` = Stage 1 removal fraction for component i
- `R2_i` = Stage 2 removal fraction for component i
- `r_tail` = S2 tail recycle fraction
- `r_prod` = Product recycle fraction

### Stream Flow Calculations

```
Exhaust = Mixed_Feed * R1
Interstage = Mixed_Feed - Exhaust
S2_Tail = Interstage * R2
Gross_Product = Interstage - S2_Tail
Net_Product = Gross_Product * (1 - r_prod)
```

### Flammability Assessment

| Condition                   | Status           |
| --------------------------- | ---------------- |
| O2 < 0.1%                   | Safe - Low O2    |
| H2 < 4%                     | Safe - Below LFL |
| H2 > 4% AND O2 > 2%         | CRITICAL         |
| H2 > 75%                    | Caution - Rich   |
| 4% < H2 < 75% AND O2 > 0.1% | FLAMMABLE        |

## Example Usage

### Sensitivity Analysis

```python
from upstream_drift_tools.process_calculators.psa_package import calculate_sensitivity
import numpy as np

results = calculate_sensitivity(
    total_feed=1100.0,
    s2_tail_recycle_range=np.linspace(0, 1, 11),
    product_recycle_range=np.array([0.0, 0.1, 0.2]),
)

print(f"H2 Recovery Range: {results['h2_recovery'].min():.1f}% - {results['h2_recovery'].max():.1f}%")
```

### O2 Safety Analysis

```python
from upstream_drift_tools.process_calculators.psa_package import calculate_o2_safety_analysis

safety = calculate_o2_safety_analysis(
    inlet_o2_pcts=np.array([0.5, 1.0, 2.0, 5.0]),
    stage1_o2_removal_range=np.arange(50.0, 100.0, 5.0),
)
```

## Troubleshooting

| Issue                           | Cause                         | Solution                                   |
| ------------------------------- | ----------------------------- | ------------------------------------------ |
| "Missing required dependencies" | PyQt6/numpy not installed     | Run `pip install PyQt6 numpy matplotlib`   |
| "Engine not available"          | Package not in PYTHONPATH     | Ensure `upstream_drift_tools` is installed |
| Mass balance error > 1e-6       | Numerical precision issue     | Check for extreme removal fractions        |
| Zero net product flow           | High product recycle fraction | Reduce `product_recycle_frac`              |
| Web app password prompt         | Security feature              | Default password: "password"               |

## Related Tools

- **Steam Engine Calculator**: Thermodynamic property calculations for steam systems
- **Flow Rate Converter**: Unit conversions for gas flow rates (SCFM, ACFM, Nm3/hr)
- **ODE Solver**: Numerical integration for dynamic PSA bed models
- **Thermal Profile Predictor**: Temperature predictions for adsorption beds

## References

- Stream numbering follows the standard PSA Process Flow Diagram (PFD)
- Model validated against Excel reference within 1e-10 relative tolerance
- H2 flammability limits: LFL 4%, UFL 75%
