# Steam Engine Calculator

A comprehensive steam thermodynamic property calculator implementing the IAPWS-IF97 standard for industrial steam applications. This tool provides accurate calculations of enthalpy, entropy, density, and transport properties across all steam phases.

## Purpose

The Steam Engine Calculator enables engineers and scientists to:
- Calculate steam thermodynamic properties from temperature and pressure inputs
- Determine saturated steam properties from temperature or pressure alone
- Identify phase states (liquid, vapor, two-phase, supercritical)
- Obtain transport properties for heat transfer and fluid flow calculations
- Support multiple calculation backends (CoolProp, Cantera, simplified correlations)

## Key Features

- **IAPWS-IF97 Compliance**: Industry-standard accuracy for steam properties
- **Multiple Calculation Modes**: Temperature-Pressure, Saturated from T, Saturated from P
- **Phase Detection**: Automatic identification of liquid, vapor, two-phase, and supercritical regions
- **Transport Properties**: Thermal conductivity, viscosity, speed of sound
- **Derived Properties**: Compressibility factor, Prandtl number, specific heat ratio
- **Unit Flexibility**: Support for K/C temperature and Pa/kPa/bar/MPa pressure units
- **Catppuccin Mocha Theme**: Modern dark interface for reduced eye strain

## Installation / Prerequisites

### Required Dependencies
```bash
pip install PyQt6 numpy
```

### Optional High-Accuracy Backends
```bash
# CoolProp backend (recommended for accuracy)
pip install CoolProp

# Cantera backend (alternative thermodynamic library)
pip install cantera
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
from upstream_drift_tools.calculators.thermo.steam_engine import SteamCalculationEngine

engine = SteamCalculationEngine()

# Calculate properties at given T and P
result = engine.calculate_properties(
    temperature=373.15,  # K
    pressure=101325,     # Pa
    engine="auto"
)

print(f"Enthalpy: {result.enthalpy / 1000:.2f} kJ/kg")
print(f"Entropy: {result.entropy / 1000:.4f} kJ/kg-K")
print(f"Phase: {result.phase}")
```

## Input Parameters

### Temperature Input

| Parameter | Description | Range | Units |
|-----------|-------------|-------|-------|
| Temperature | Steam temperature | 273.16 - 647.15 | K |
| Temperature | Steam temperature | 0.01 - 374.0 | C |

### Pressure Input

| Parameter | Description | Range | Units |
|-----------|-------------|-------|-------|
| Pressure | Steam pressure | > 0 - 100,000,000 | Pa |
| Pressure | Steam pressure | > 0 - 100,000 | kPa |
| Pressure | Steam pressure | > 0 - 1,000 | bar |
| Pressure | Steam pressure | > 0 - 100 | MPa |

### Calculation Modes

| Mode | Required Inputs | Description |
|------|-----------------|-------------|
| Temperature & Pressure | T, P | Calculate properties at specified state |
| Saturated (from Temperature) | T | Calculate saturated liquid/vapor properties |
| Saturated (from Pressure) | P | Calculate saturated liquid/vapor properties |

### Calculation Engines

| Engine | Description | Accuracy |
|--------|-------------|----------|
| Auto | Automatic selection of best available engine | Highest available |
| CoolProp | NIST reference implementation | Industrial grade |
| Cantera | Thermodynamic calculation library | Industrial grade |
| Simplified | Built-in correlations | Engineering estimates |

## Output Format

The `SteamProperties` result contains:

### Thermodynamic Properties

| Property | Description | Units |
|----------|-------------|-------|
| `temperature` | Steam temperature | K |
| `pressure` | Steam pressure | Pa |
| `density` | Steam density | kg/m3 |
| `specific_volume` | Specific volume | m3/kg |
| `enthalpy` | Specific enthalpy | J/kg |
| `entropy` | Specific entropy | J/kg-K |
| `internal_energy` | Specific internal energy | J/kg |
| `cp` | Isobaric specific heat | J/kg-K |
| `cv` | Isochoric specific heat | J/kg-K |

### Transport Properties

| Property | Description | Units |
|----------|-------------|-------|
| `speed_of_sound` | Sonic velocity | m/s |
| `thermal_conductivity` | Thermal conductivity | W/m-K |
| `dynamic_viscosity` | Dynamic viscosity | Pa-s |
| `kinematic_viscosity` | Kinematic viscosity | m2/s |

### Derived Properties

| Property | Description | Units |
|----------|-------------|-------|
| `phase` | Phase state | - |
| `quality` | Steam quality (two-phase only) | 0-1 |
| `compressibility_factor` | Compressibility factor Z | - |
| `prandtl_number` | Prandtl number | - |
| `specific_heat_ratio` | Cp/Cv ratio (k or gamma) | - |

## Mathematical Models

### IAPWS-IF97 Regions

The IAPWS-IF97 formulation divides the thermodynamic surface into five regions:

```
Region 1: Compressed liquid (T < T_sat at given P)
Region 2: Superheated vapor (T > T_sat at given P)
Region 3: Near-critical region
Region 4: Saturation line (two-phase)
Region 5: High-temperature steam (T > 1073.15 K)
```

### Saturation Pressure Equation

```
ln(P_sat/P_c) = (T_c/T) * sum(n_i * theta^a_i)
```

Where:
- `P_c` = Critical pressure (22.064 MPa)
- `T_c` = Critical temperature (647.096 K)
- `theta` = 1 - T/T_c
- `n_i`, `a_i` = IAPWS-IF97 coefficients

### Specific Enthalpy (Region 2 - Vapor)

```
h = R * T * gamma_tau

gamma = gamma_0 + gamma_r
gamma_tau = (d gamma / d tau) at constant pi
```

Where:
- `R` = Specific gas constant (461.526 J/kg-K)
- `tau` = T_star / T (dimensionless temperature)
- `pi` = P / P_star (dimensionless pressure)

### Transport Property Correlations

**Dynamic Viscosity:**
```
mu = mu_0(T) * mu_1(T, rho) * mu_2(T, rho)
```

**Thermal Conductivity:**
```
k = k_0(T) + k_1(T, rho) + k_2(T, rho)
```

## Example Usage

### Calculate Boiler Steam Properties

```python
# Boiler operating at 10 bar, 180 C
temp_k = 180 + 273.15  # Convert to Kelvin
pressure_pa = 10 * 100000  # Convert bar to Pa

result = engine.calculate_properties(
    temperature=temp_k,
    pressure=pressure_pa
)

print(f"Phase: {result.phase}")
print(f"Enthalpy: {result.enthalpy / 1000:.2f} kJ/kg")
print(f"Density: {result.density:.4f} kg/m3")
```

### Saturated Steam Tables

```python
# Get saturated properties at 100 C
result = engine.calculate_saturated_properties_from_temperature(
    temperature=373.15,
    engine="coolprop"
)

print(f"Saturation Pressure: {result.pressure / 1000:.2f} kPa")
print(f"h_fg: {(result.enthalpy) / 1000:.2f} kJ/kg")
```

### Turbine Expansion Calculations

```python
# Inlet conditions
h1 = result_inlet.enthalpy
s1 = result_inlet.entropy

# Isentropic expansion to lower pressure
result_exit_ideal = engine.calculate_properties_from_entropy(
    entropy=s1,
    pressure=exit_pressure
)
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "Temperature below triple point" | T < 273.16 K | Ensure temperature >= 0.01 C |
| "Temperature above critical point" | T > 647.15 K | Use Region 5 for high-T steam |
| "Pressure exceeds maximum" | P > 100 MPa | Reduce pressure input |
| "CoolProp: Not installed" | Missing backend | Install with `pip install CoolProp` |
| "Engine not available" | Missing dependencies | Check `upstream_drift_tools` installation |
| Phase shows "--" | Calculation error | Verify inputs are within valid ranges |

## Related Tools

- **PSA Package**: Gas separation system modeling using steam for regeneration
- **Flow Rate Converter**: Convert steam flow rates between mass and volumetric units
- **Thermal Profile Predictor**: Temperature profiles in steam-heated vessels
- **ODE Solver**: Dynamic simulation of steam system transients

## References

- IAPWS-IF97: Industrial Formulation for Thermodynamic Properties of Water and Steam
- IAPWS R7-97: Revised Release on the IAPWS Industrial Formulation 1997
- CoolProp: Open-source thermophysical property library
- Critical Point: T_c = 647.096 K, P_c = 22.064 MPa, rho_c = 322 kg/m3
