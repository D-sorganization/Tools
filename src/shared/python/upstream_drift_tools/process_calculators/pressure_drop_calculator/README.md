# Advanced Pressure Drop Calculator

Comprehensive pressure drop calculator for combustion and gasification gases with support for variable composition, multiple friction factor correlations, and extensive unit conversions.

## Features

- **Variable Gas Compositions**: Support for H₂, CO, CO₂, CH₄, N₂, H₂O, O₂, Ar, H₂S, NH₃, and more
- **Advanced Friction Factor Correlations**: Colebrook-White, Swamee-Jain, Churchill, Haaland
- **Standard and Custom Pipe Sizes**: ASME B36.10M pipe database with all common schedules
- **Comprehensive Unit Support**:
  - Mass flow: kg/s, kg/h, lb/hr
  - Molar flow: mol/s, kmol/h, lbmol/hr
  - Volumetric flow: SCFM, ACFM, Nm³/h, m³/s
- **Fitting and Valve Losses**: Extensive database of K-factors from Crane TP-410
- **Compressible Flow Corrections**: Real gas effects with Z-factor calculations
- **Erosional Velocity Checks**: API RP 14E correlations
- **Professional Accuracy**: All correlations sourced from established references

## Quick Start

### Basic Usage

```python
from calculators.pressure_drop_calculator import calculate_pressure_drop, print_results

# Calculate pressure drop for air in 4" Schedule 40 pipe
result = calculate_pressure_drop(
    pipe_size='4',
    pipe_schedule='40',
    pipe_length=100,  # meters
    flow_rate=1000,  # SCFM
    flow_unit='SCFM',
    pressure=5,  # bar
    temperature=400  # K
)

print_results(result)
```

### Custom Gas Composition

```python
from calculators.pressure_drop_calculator import calculate_pressure_drop_custom_gas

# Syngas composition
syngas = {
    'H2': 0.30,
    'CO': 0.40,
    'CO2': 0.15,
    'N2': 0.10,
    'CH4': 0.05
}

result = calculate_pressure_drop_custom_gas(
    pipe_diameter=0.1543,  # 6" Schedule 40 (meters)
    pipe_length=50,
    gas_composition=syngas,
    flow_rate=2000,
    flow_unit='kg/h',
    pressure=25,  # bar
    temperature=800  # K
)

print(f"Pressure drop: {result['pressure_drop_bar']:.4f} bar")
print(f"Outlet pressure: {result['outlet_pressure_bar']:.2f} bar")
print(f"Reynolds number: {result['reynolds_number']:.0f}")
print(f"Flow velocity: {result['flow_velocity_m_s']:.2f} m/s")
```

### With Fittings and Valves

```python
# Add fittings to calculation
fittings = [
    {'type': '90_elbow_std', 'quantity': 4},
    {'type': '45_elbow_std', 'quantity': 2},
    {'type': 'gate_valve_open', 'quantity': 2},
    {'type': 'tee_through_run', 'quantity': 1}
]

result = calculate_pressure_drop(
    pipe_size='6',
    pipe_schedule='40',
    pipe_length=100,
    flow_rate=5000,
    flow_unit='kg/h',
    pressure=20,
    temperature=700,
    gas_composition=syngas,
    fittings=fittings
)
```

## Advanced Usage

### Low-Level API (Direct Engine Access)

```python
from calculators.pressure_drop_calculator import (
    PressureDropCalculationEngine,
    PressureDropInputs,
    GasComposition,
    PipeFitting
)

# Define gas composition
composition = GasComposition(components={
    'H2': 0.30,
    'CO': 0.40,
    'CO2': 0.20,
    'N2': 0.10
})

# Define fittings
fittings = [
    PipeFitting('90_elbow_std', quantity=4, k_factor=30),
    PipeFitting('gate_valve_open', quantity=2, k_factor=8),
]

# Create inputs
inputs = PressureDropInputs(
    pipe_diameter=0.1023,  # 4" Schedule 40
    pipe_length=100.0,
    pipe_roughness=0.000045,  # Commercial steel
    elevation_change=0.0,
    mass_flow_rate=0.556,  # kg/s
    inlet_pressure=10e5,  # Pa
    inlet_temperature=700,  # K
    gas_composition=composition,
    fittings=fittings,
    compressibility_correction=True,
    friction_method='colebrook'
)

# Calculate
engine = PressureDropCalculationEngine()
results = engine.calculate(inputs)

# Access detailed results
print(f"Friction factor: {results.friction_factor:.6f}")
print(f"Flow regime: {results.flow_regime}")
print(f"Friction loss: {results.friction_pressure_drop/1e5:.4f} bar")
print(f"Fitting loss: {results.fitting_pressure_drop/1e5:.4f} bar")
print(f"Mach number: {results.flow_properties.mach_number:.4f}")
```

### Using Pipe Database

```python
from calculators.pressure_drop_calculator import (
    get_pipe_spec,
    list_available_sizes,
    list_schedules_for_size
)

# List all available pipe sizes
sizes = list_available_sizes()
print(f"Available sizes: {sizes}")

# List schedules for 6" pipe
schedules = list_schedules_for_size('6')
print(f"Schedules for 6\" pipe: {schedules}")

# Get specific pipe specification
spec = get_pipe_spec('6', '40')
print(f"6\" Schedule 40:")
print(f"  OD: {spec.outer_diameter} mm")
print(f"  Wall: {spec.wall_thickness} mm")
print(f"  ID: {spec.inner_diameter} mm")
```

### Material Roughness Values

```python
from calculators.pressure_drop_calculator import get_roughness, MATERIAL_ROUGHNESS

# Get roughness for a material
roughness_m = get_roughness('Commercial Steel', 'm')
roughness_mm = get_roughness('Stainless Steel', 'mm')
roughness_ft = get_roughness('Cast Iron', 'ft')

# List all available materials
for material, (rough_mm, rough_ft, desc) in MATERIAL_ROUGHNESS.items():
    print(f"{material:30s}: ε = {rough_mm:8.5f} mm - {desc}")
```

### Fitting K-Factors

```python
from calculators.pressure_drop_calculator import (
    get_fitting_k_factor,
    list_available_fittings
)

# Get K-factor for a fitting
k_elbow = get_fitting_k_factor('90_elbow_std')
k_valve = get_fitting_k_factor('gate_valve_open')

# List all available fittings
fittings_db = list_available_fittings()
for fitting_name, k_value in sorted(fittings_db.items()):
    print(f"{fitting_name:40s}: K = {k_value:6.0f}")
```

### Flow Rate Unit Conversions

```python
from calculators.pressure_drop_calculator import (
    mass_to_mass,
    molar_to_mass,
    scfm_to_acfm,
    STANDARD_CONDITIONS
)

# Convert mass flow units
flow_lbhr = mass_to_mass(1000, 'kg/h', 'lb/hr')
print(f"1000 kg/h = {flow_lbhr:.1f} lb/hr")

# Convert molar to mass flow
MW_air = 29.0  # kg/kmol
mass_flow = molar_to_mass(10, 'kmol/h', MW_air, 'kg/h')
print(f"10 kmol/h of air = {mass_flow:.1f} kg/h")

# SCFM to ACFM conversion
T_actual = 800  # K
P_actual = 5e5  # Pa
acfm = scfm_to_acfm(1000, T_actual, P_actual, 'SCFM')
print(f"1000 SCFM @ {T_actual}K, {P_actual/1e5:.0f} bar = {acfm:.0f} ACFM")

# Available standard conditions
for name, (T, P, desc) in STANDARD_CONDITIONS.items():
    print(f"{name:10s}: {desc}")
```

## Gas Component Database

Supported gas components with thermophysical properties:

| Component | Name | MW (kg/kmol) | Tc (K) | Pc (bar) |
|-----------|------|--------------|--------|----------|
| H₂ | Hydrogen | 2.016 | 33.2 | 12.96 |
| CO | Carbon Monoxide | 28.010 | 132.9 | 34.94 |
| CO₂ | Carbon Dioxide | 44.010 | 304.2 | 73.82 |
| CH₄ | Methane | 16.043 | 190.6 | 45.99 |
| C₂H₆ | Ethane | 30.070 | 305.4 | 48.80 |
| C₂H₄ | Ethylene | 28.054 | 282.4 | 50.42 |
| N₂ | Nitrogen | 28.014 | 126.2 | 33.94 |
| O₂ | Oxygen | 31.999 | 154.6 | 50.43 |
| H₂O | Water Vapor | 18.015 | 647.1 | 220.64 |
| Ar | Argon | 39.948 | 150.9 | 48.98 |
| H₂S | Hydrogen Sulfide | 34.082 | 373.5 | 90.00 |
| NH₃ | Ammonia | 17.031 | 405.7 | 113.57 |
| Air | Air (pseudo) | 28.97 | 132.5 | 37.74 |

## Friction Factor Methods

Available friction factor correlations:

1. **Colebrook-White** (`'colebrook'`): Most accurate, requires iteration
   - Reference: Colebrook (1939)
   - Recommended for final calculations

2. **Swamee-Jain** (`'swamee-jain'`): Explicit approximation
   - Accurate within 1% of Colebrook
   - Fast, no iteration required
   - Valid: 5000 < Re < 10⁸, 10⁻⁶ < ε/D < 10⁻²

3. **Churchill** (`'churchill'`): Valid for all flow regimes
   - Reference: Churchill (1977)
   - Works for laminar, transitional, and turbulent flow
   - Single equation for all Re

4. **Haaland** (`'haaland'`): Simple explicit formula
   - Accurate within 1.5%
   - Very fast computation

## Output Variables

The calculator returns a comprehensive dictionary with:

### Pressure Drops
- `pressure_drop_pa`, `pressure_drop_bar`, `pressure_drop_psi`, `pressure_drop_kpa`
- `friction_loss_pa`, `friction_loss_bar`
- `fitting_loss_pa`, `fitting_loss_bar`
- `elevation_loss_pa`
- `pressure_drop_per_100ft_pa`

### Outlet Conditions
- `outlet_pressure_pa`, `outlet_pressure_bar`, `outlet_pressure_psi`

### Flow Characteristics
- `friction_factor`: Darcy friction factor
- `reynolds_number`: Reynolds number
- `flow_velocity_m_s`, `flow_velocity_ft_s`
- `mach_number`: Mach number
- `flow_regime`: 'laminar', 'transitional', or 'turbulent'

### Gas Properties
- `density_kg_m3`: Gas density
- `viscosity_pa_s`: Dynamic viscosity
- `compressibility_factor`: Z-factor
- `molecular_weight`: Mixture molecular weight

### Performance Metrics
- `erosional_velocity_m_s`: Erosional velocity limit (API RP 14E)
- `erosion_ratio`, `erosion_ratio_percent`: Actual/erosional velocity ratio
- `velocity_pressure_pa`: Dynamic pressure

### Warnings
- `warnings`: List of warning messages

## Example Applications

### 1. Gasifier Syngas Line Design

```python
# Design syngas transfer line from gasifier to quench
syngas = {'H2': 0.28, 'CO': 0.42, 'CO2': 0.18, 'N2': 0.08, 'CH4': 0.04}

result = calculate_pressure_drop(
    pipe_size='8',
    pipe_schedule='80',  # Heavy wall for high temperature
    pipe_length=25,
    pipe_material='Stainless Steel 316',
    flow_rate=8000,  # kg/h
    flow_unit='kg/h',
    pressure=30,  # bar
    temperature=1000,  # K (~727°C)
    gas_composition=syngas,
    elevation_change=5,  # 5 m rise
    friction_method='colebrook'
)

print_results(result, "Gasifier Syngas Line")
```

### 2. Combustion Air Blower Sizing

```python
# Size blower for combustion air supply
result = calculate_pressure_drop(
    pipe_size='12',
    pipe_schedule='40',
    pipe_length=150,
    flow_rate=50000,  # SCFM
    flow_unit='SCFM',
    pressure=1.5,  # bar
    temperature=300,  # K (ambient)
    gas_composition={'Air': 1.0},
    fittings=[
        {'type': '90_elbow_long', 'quantity': 6},
        {'type': 'butterfly_valve_open', 'quantity': 1},
    ]
)

# Blower must overcome this pressure drop
required_dp = result['pressure_drop_bar']
print(f"Required blower head: {required_dp:.3f} bar")
```

### 3. Product Gas Cooling Line

```python
# Cooled product gas from gasifier
cooled_gas = {'H2': 0.30, 'CO': 0.35, 'CO2': 0.25, 'N2': 0.10}

result = calculate_pressure_drop(
    pipe_size='6',
    pipe_schedule='40',
    pipe_length=50,
    flow_rate=150,  # kmol/h
    flow_unit='kmol/h',
    pressure=20,  # bar
    temperature=400,  # K (cooled)
    gas_composition=cooled_gas,
    fittings=[
        {'type': '90_elbow_std', 'quantity': 3},
        {'type': 'gate_valve_open', 'quantity': 1},
    ]
)

print_results(result)
```

## Validation and Accuracy

### Verified Against

- Crane TP-410 example problems
- Perry's Chemical Engineers' Handbook calculations
- GPSA Engineering Data Book examples
- Published literature data for gas flow

### Typical Accuracy

- Friction factor: ±1% (Colebrook-White)
- Pressure drop: ±3-5% for turbulent flow
- Gas properties: ±2-3% at typical conditions

### Limitations

- Assumes single-phase gas flow (no liquids or solids)
- Isothermal flow (temperature constant along pipe)
- Fully developed flow (entrance effects neglected)
- Steady-state conditions

## Technical References

### Primary Sources

1. **Crane Technical Paper No. 410** (TP-410)
   - "Flow of Fluids Through Valves, Fittings, and Pipe"
   - Fitting K-factors, resistance coefficients

2. **Perry's Chemical Engineers' Handbook**, 9th Edition
   - Chapter 6: Fluid and Particle Dynamics
   - Friction factor correlations, compressible flow

3. **API RP 14E**
   - "Recommended Practice for Design and Installation of Offshore Production Platform Piping Systems"
   - Erosional velocity correlations

4. **ASME B36.10M-2015**
   - "Welded and Seamless Wrought Steel Pipe"
   - Pipe dimensions and specifications

5. **Reid, Prausnitz, Poling** (2001)
   - "The Properties of Gases and Liquids", 5th Edition
   - Gas property correlations, viscosity, compressibility

### Key Publications

- Colebrook, C.F. (1939): "Turbulent Flow in Pipes", J. Inst. Civil Engineers
- Swamee, P.K., Jain, A.K. (1976): "Explicit Equations for Pipe-Flow Problems", ASCE
- Churchill, S.W. (1977): "Friction Factor Equation Spans All Fluid Flow Regimes", Chem. Eng.
- Moody, L.F. (1944): "Friction factors for pipe flow", Trans. ASME
- Wilke, C.R. (1950): "A Viscosity Equation for Gas Mixtures", J. Chem. Phys.

## Testing

Run the example calculations:

```bash
python -m calculators.pressure_drop_calculator.pressure_drop_interface
```

Run unit tests:

```bash
pytest tests/test_pressure_drop_calculator.py -v
```

## Support and Contributing

For issues, feature requests, or contributions, please contact the development team.

## License

Professional use authorized for gasification and combustion system design.

---

**Version**: 1.0.0
**Last Updated**: 2024
**Status**: Production Ready
