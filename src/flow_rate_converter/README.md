# Flow Rate Converter

A comprehensive flow rate unit conversion tool for process engineering applications. This calculator handles mass, molar, and volumetric flow rate conversions including SCFM/ACFM/Nm3/hr with temperature and pressure corrections.

## Purpose

The Flow Rate Converter enables process engineers to:

- Convert between mass flow rate units (kg/s, lb/h, etc.)
- Convert between molar flow rate units (mol/s, kmol/h, lbmol/h, etc.)
- Convert between volumetric flow rate units (m3/h, CFM, GPM, etc.)
- Apply temperature and pressure corrections for gas flows
- Handle standard and actual volumetric flow conversions

## Key Features

- **Three Conversion Categories**: Mass, molar, and volumetric flow rates
- **Comprehensive Unit Support**: Metric, imperial, and mixed units
- **Standard Conditions Support**: SCFM, ACFM, Nm3/hr conversions
- **High Precision**: 6 significant figures for accurate engineering calculations
- **Tabbed Interface**: Organized by flow rate type for easy navigation
- **Catppuccin Mocha Theme**: Modern dark interface design

## Installation / Prerequisites

### Required Dependencies

```bash
pip install PyQt6
```

### Optional Dependencies

```bash
# For backend calculation library
pip install upstream_drift_tools
```

### Python Version

- Python 3.10 or higher recommended

## Usage Instructions

### Launch Desktop Application

```bash
python launch_pyqt6.py
```

### Programmatic Usage

```python
from upstream_drift_tools.calculators.conversion.flow_rate_converter import (
    mass_to_mass,
    molar_to_molar,
    VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S
)

# Mass flow conversion
result = mass_to_mass(1000, "kg/h", "lb/h")
print(f"1000 kg/h = {result:.2f} lb/h")

# Molar flow conversion
result = molar_to_molar(100, "kmol/h", "lbmol/h")
print(f"100 kmol/h = {result:.2f} lbmol/h")

# Volumetric flow conversion
value_m3_s = 1000 * VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S["m3/h"]
result_cfm = value_m3_s / VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S["CFM"]
print(f"1000 m3/h = {result_cfm:.2f} CFM")
```

## Input Parameters

### Mass Flow Rate Units

| Unit     | Description          | Conversion to kg/s |
| -------- | -------------------- | ------------------ |
| `kg/s`   | Kilograms per second | 1.0                |
| `kg/h`   | Kilograms per hour   | 1/3600             |
| `kg/min` | Kilograms per minute | 1/60               |
| `g/s`    | Grams per second     | 0.001              |
| `g/h`    | Grams per hour       | 0.001/3600         |
| `lb/s`   | Pounds per second    | 0.453592           |
| `lb/h`   | Pounds per hour      | 0.453592/3600      |
| `lb/min` | Pounds per minute    | 0.453592/60        |
| `ton/h`  | Metric tons per hour | 1000/3600          |

### Molar Flow Rate Units

| Unit        | Description            | Conversion to mol/s |
| ----------- | ---------------------- | ------------------- |
| `mol/s`     | Moles per second       | 1.0                 |
| `mol/h`     | Moles per hour         | 1/3600              |
| `mol/min`   | Moles per minute       | 1/60                |
| `kmol/s`    | Kilomoles per second   | 1000                |
| `kmol/h`    | Kilomoles per hour     | 1000/3600           |
| `kmol/min`  | Kilomoles per minute   | 1000/60             |
| `lbmol/s`   | Pound-moles per second | 453.592             |
| `lbmol/h`   | Pound-moles per hour   | 453.592/3600        |
| `lbmol/min` | Pound-moles per minute | 453.592/60          |

### Volumetric Flow Rate Units

| Unit      | Description             | Conversion to m3/s |
| --------- | ----------------------- | ------------------ |
| `m3/s`    | Cubic meters per second | 1.0                |
| `m3/h`    | Cubic meters per hour   | 1/3600             |
| `m3/min`  | Cubic meters per minute | 1/60               |
| `L/s`     | Liters per second       | 0.001              |
| `L/min`   | Liters per minute       | 0.001/60           |
| `L/h`     | Liters per hour         | 0.001/3600         |
| `ft3/s`   | Cubic feet per second   | 0.0283168          |
| `ft3/min` | Cubic feet per minute   | 0.0283168/60       |
| `ft3/h`   | Cubic feet per hour     | 0.0283168/3600     |
| `CFM`     | Cubic feet per minute   | 0.0283168/60       |
| `GPM`     | US gallons per minute   | 0.0000630902       |

## Output Format

Results are displayed with:

- 6 significant figures for precision
- Target unit label appended
- Color-coded status (green for success, red for errors)

Example output:

```
2,204.62 lb/h
```

## Mathematical Models

### Mass Flow Conversion

```
m_target = m_source * (CF_source / CF_target)
```

Where:

- `m_source` = Source mass flow rate value
- `CF_source` = Conversion factor from source unit to kg/s
- `CF_target` = Conversion factor from target unit to kg/s

### Molar Flow Conversion

```
n_target = n_source * (CF_source / CF_target)
```

Where:

- `n_source` = Source molar flow rate value
- `CF_source` = Conversion factor from source unit to mol/s
- `CF_target` = Conversion factor from target unit to mol/s

### Volumetric Flow Conversion (Standard to Actual)

For gas flow at non-standard conditions:

```
Q_actual = Q_standard * (T_actual / T_standard) * (P_standard / P_actual)
```

Where:

- `Q_standard` = Standard volumetric flow (at 15C, 101.325 kPa for Nm3)
- `T_standard` = 288.15 K (15C) for normal conditions
- `T_actual` = Actual temperature in Kelvin
- `P_standard` = 101.325 kPa for normal conditions
- `P_actual` = Actual pressure in kPa

### SCFM to ACFM Conversion

Standard Cubic Feet per Minute (60F, 14.696 psia) to Actual CFM:

```
ACFM = SCFM * (T_actual / 520) * (14.696 / P_actual)
```

Where:

- `T_actual` = Actual temperature in Rankine (F + 459.67)
- `P_actual` = Actual pressure in psia

### Nm3/hr to SCFM Conversion

```
SCFM = Nm3/hr * 0.5886
```

Based on:

- Normal conditions: 0C (273.15 K), 101.325 kPa
- Standard conditions: 60F (288.71 K), 14.696 psia

## Example Usage

### Convert Process Stream Flow

```python
from upstream_drift_tools.calculators.conversion.flow_rate_converter import mass_to_mass

# Convert feed stream from metric to imperial
feed_kg_h = 5000
feed_lb_h = mass_to_mass(feed_kg_h, "kg/h", "lb/h")
print(f"Feed rate: {feed_lb_h:,.0f} lb/h")
# Output: Feed rate: 11,023 lb/h
```

### Reactor Molar Flow Balance

```python
from upstream_drift_tools.calculators.conversion.flow_rate_converter import molar_to_molar

# Convert reactor feed for stoichiometry calculations
methane_kmol_h = 100
methane_mol_s = molar_to_molar(methane_kmol_h, "kmol/h", "mol/s")
print(f"Methane feed: {methane_mol_s:.3f} mol/s")
# Output: Methane feed: 27.778 mol/s
```

### Compressor Flow Calculation

```python
# Convert compressor specification from SCFM to m3/h
scfm = 1000
m3_h = scfm * (0.0283168/60) / (1/3600)
print(f"Flow rate: {m3_h:.1f} m3/h")
# Output: Flow rate: 1,699.0 m3/h
```

## Troubleshooting

| Issue                   | Cause                   | Solution                                   |
| ----------------------- | ----------------------- | ------------------------------------------ |
| "Error: Invalid unit"   | Unsupported unit string | Check unit spelling matches supported list |
| Zero result             | Input value is zero     | Verify input value > 0                     |
| Very large/small result | Unit scale mismatch     | Verify correct source and target units     |
| Import error            | Missing package         | Install `upstream_drift_tools` package     |
| GUI not responding      | Large value calculation | Wait for computation to complete           |

### Common Conversion Pitfalls

1. **SCFM vs ACFM**: Remember SCFM is at standard conditions (60F, 14.696 psia)
2. **Nm3 vs Sm3**: Normal cubic meters (0C) differ from standard cubic meters (15C)
3. **GPM**: US gallons, not imperial gallons (1 US gal = 3.785 L)
4. **CFM**: Equivalent to ft3/min, commonly used for air flow

## Related Tools

- **PSA Package**: Gas separation calculations requiring flow rate inputs
- **Steam Engine Calculator**: Steam mass flow and energy calculations
- **Thermal Profile Predictor**: Flow rate inputs for heat transfer
- **ODE Solver**: Dynamic flow modeling for process simulations

## References

### Standard Reference Conditions

| Standard        | Temperature    | Pressure    |
| --------------- | -------------- | ----------- |
| Normal (Nm3)    | 0C (273.15 K)  | 101.325 kPa |
| Standard (SCFM) | 60F (288.71 K) | 14.696 psia |
| API Standard    | 60F            | 14.73 psia  |
| ISO Standard    | 15C            | 101.325 kPa |

### Conversion Constants

| Constant      | Value     | Description               |
| ------------- | --------- | ------------------------- |
| lb to kg      | 0.453592  | Pound to kilogram         |
| ft3 to m3     | 0.0283168 | Cubic foot to cubic meter |
| gal (US) to L | 3.78541   | US gallon to liter        |
| lbmol to mol  | 453.59237 | Pound-mole to mole        |
