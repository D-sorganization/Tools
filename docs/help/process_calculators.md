# Process Engineering Calculators

The Process Engineering category contains 24 specialized calculators for industrial process design and analysis. All calculators share:

- PyQt6-based desktop GUI with theme support
- Optional web interface (React/Tauri)
- Shared calculation engines for consistency
- Input validation and unit conversion

## Calculator Categories

### Thermodynamic Calculators

#### Acid Gas Dewpoint Calculator

Calculate dewpoints for acid gases in syngas systems.

**Purpose**: Determine safe operating temperatures to prevent acid gas condensation and corrosion.

**Inputs**:

- Operating temperature (C)
- Operating pressure (bar)
- Gas composition:
  - H2O (water vapor)
  - HF (hydrogen fluoride)
  - HCl (hydrogen chloride)
  - H2S (hydrogen sulfide)

**Outputs**:

- Individual component dewpoints
- Overall acid dewpoint
- Safety margin calculation
- Condensation warnings

**Calculation Methods**:

- Antoine equation
- Extended Antoine equation

**Presets**:

- Typical Syngas
- Coal Gasification
- Biomass Gasification

---

#### Syngas Water Calculator

Calculate water content and dewpoint in syngas systems.

**Purpose**: Determine water vapor saturation conditions for process design.

**Inputs**:

- Temperature (C)
- Pressure (bar)
- Gas composition preset

**Outputs**:

- Water content in multiple units:
  - mg/Nm3
  - ppmv
  - g/m3
  - lb/MMscf
- Vapor pressure
- Dewpoint temperature
- Condensation risk assessment

**Calculation Methods**:
| Method | Best For |
|--------|----------|
| Auto | General use |
| Antoine | Standard conditions |
| Buck | Meteorological |
| IAPWS-IF97 | High accuracy |
| Magnus | Quick estimates |

---

#### Steam Engine Calculator

Calculate steam thermodynamic properties using multiple engines.

**Modes**:

- Temperature & Pressure
- Saturated (from Temperature)
- Saturated (from Pressure)

**Outputs**:

- Phase state and quality
- Density
- Enthalpy
- Entropy
- Internal energy
- Specific heats (Cp, Cv)
- Speed of sound
- Thermal conductivity
- Viscosity
- Compressibility factor

**Calculation Engines**:

- CoolProp (recommended)
- Cantera
- Simplified correlations

---

### Equipment Sizing Calculators

#### Baghouse Calculator

Design baghouse filter systems for particulate removal.

**Inputs**:

- Gas stream:
  - Flow rate (kg/s)
  - Inlet temperature (C)
  - Pressure (bar)
- Solids:
  - Carbon input rate (kg/h)
  - Ash input rate (kg/h)
- Removal efficiencies:
  - Carbon removal (%)
  - Ash removal (%)
- Equipment:
  - Heat loss (kW)
  - Drum volume (m3)
  - Bag filter area (m2)

**Outputs**:

- Solids removal rates
- Drum fill time (hours/days)
- Air-to-cloth ratio
- Outlet temperature
- Pressure drop estimate

---

#### Flare Calculator

Size flare systems and determine safety zones.

**Inputs**:

- Total flow rate (kg/hr)
- Gas composition (8 components):
  - H2, CO, CH4, CO2
  - N2, H2O, H2S, Other
- Temperature (K)
- Pressure (bar)

**Outputs**:

- Flare dimensions:
  - Stack height
  - Tip diameter
- Performance:
  - Exit velocity
  - Heat release rate
  - Flame length
- Safety zones:
  - Lethal zone
  - Damage zone
  - Safe zone
  - Comfort zone

**Standards Reference**: API 521, API 537

---

#### Scrubber Calculator

Design wet scrubber systems for gas cleaning.

**Inputs**:

- Gas flow rate
- Contaminant concentrations
- Scrubbing liquid properties
- Column dimensions

**Outputs**:

- Removal efficiency
- Liquid flow requirements
- Pressure drop
- Mass transfer coefficients

---

#### Pressure Drop Calculator

Calculate pressure drops in piping systems.

**Pipe Parameters**:

- Nominal sizes: 0.5" to 24"
- Schedules: 5, 10, 20, 40, 80, STD, XS, XXS
- Materials: Carbon Steel, Stainless, Copper, PVC, HDPE, Concrete

**Flow Conditions**:

- Flow rate (various units)
- Inlet pressure
- Temperature
- Gas composition

**Friction Methods**:
| Method | Description |
|--------|-------------|
| Colebrook | Implicit, most accurate |
| Swamee-Jain | Explicit approximation |
| Churchill | Single equation all regimes |
| Haaland | Simple explicit |

**Outputs**:

- Total pressure drop (Pa)
- Friction factor
- Reynolds number
- Flow velocity
- Mach number
- Erosional velocity warning

---

#### TRC Vessel Designer

Design thermal reaction chamber vessels with refractory lining.

**Geometry Configuration**:

- Cylinder section:
  - Height
  - Inner diameter
- Cone section:
  - Height
  - Bottom diameter

**Refractory Presets**:

- Standard (3-layer)
- High Temperature (4-layer)
- Economy (2-layer)

**Outputs**:

- Net internal volume
- Total refractory mass
- Layer-by-layer breakdown
- Residence time (with flow rate)
- Outside surface area
- Shell dimensions

---

### Process Unit Calculators

#### WGS Reactor Calculator

Design and analyze water-gas shift reactors.

**Inputs**:

- Feed composition
- Inlet temperature
- Pressure
- Catalyst parameters

**Outputs**:

- CO conversion
- Product composition
- Heat duty
- Equilibrium approach

---

#### PSA Package

Design Pressure Swing Adsorption systems.

**Inputs**:

- Feed composition
- Product purity requirements
- Operating pressures

**Outputs**:

- Bed sizing
- Cycle timing
- Product recovery
- Adsorbent requirements

---

#### Syngas Compression Calculator

Size syngas compression systems.

**Inputs**:

- Inlet conditions
- Outlet pressure requirement
- Gas composition

**Outputs**:

- Number of stages
- Compression ratios
- Power requirements
- Intercooler duties

---

### Electrical Systems

#### Electrode Advisor

Analyze 3-phase electrical systems for electrode heating.

**Inputs**:

- Phase measurements:
  - Current (A) x 3
  - Voltage (V) x 3
- Electrode configuration:
  - Depths (m) x 3
- Physical parameters:
  - Bath diameter
  - Electrode tip diameter
  - Bath temperature

**Outputs**:

- Total power (kW)
- Phase powers
- Phase resistances
- System balance indicators

---

### Unit Conversion

#### Flow Rate Converter

Convert between flow rate units.

**Mass Flow Units**:
kg/s, kg/h, kg/min, g/s, g/h, lb/s, lb/h, lb/min, ton/h

**Molar Flow Units**:
mol/s, mol/h, kmol/s, kmol/h, lbmol/s, lbmol/h

**Volumetric Flow Units**:
m3/s, m3/h, L/s, L/min, ft3/s, CFM, GPM

---

## Additional Calculators

| Calculator          | Purpose                                       |
| ------------------- | --------------------------------------------- |
| Heat Exchanger      | LMTD, effectiveness-NTU calculations          |
| Pump Sizing         | Head, power, NPSH calculations                |
| Tank Volume         | Storage tank capacity                         |
| Relief Valve Sizer  | Safety valve sizing per API 520               |
| Cooling Tower       | Performance and approach                      |
| Distillation Column | McCabe-Thiele, stage calculations             |
| Reactor Sizing      | CSTR, PFR, batch design                       |
| Catalyst Bed        | Volume, pressure drop                         |
| Combustion          | Air requirements, adiabatic flame temperature |
| Mass Balance        | Process mass balance                          |
| Energy Balance      | Process energy balance                        |

---

## Common Features

### Input Validation

All calculators validate inputs:

- Range checking
- Unit consistency
- Physical constraints (e.g., positive values for flow)

### Preset Compositions

Quick-start with typical compositions:

- Syngas (various sources)
- Natural gas
- Air
- Flue gas

### Export Options

- Copy results to clipboard
- Export to CSV
- Generate reports

### Shared Engines

Calculators use shared libraries from `upstream_drift_tools`:

- Consistent calculations across tools
- Validated against literature
- Unit tested

---

## Tips for Process Engineers

1. **Always verify units** - Check that input and output units match your requirements

2. **Use presets as starting points** - Modify preset compositions for your specific case

3. **Enable Debug Mode** - See detailed calculation steps for troubleshooting

4. **Cross-check results** - Compare with hand calculations or other tools

5. **Document assumptions** - Note any deviations from default parameters

---

For detailed documentation on each calculator, see the individual tool help or the [User Manual](../USER_MANUAL.md).
