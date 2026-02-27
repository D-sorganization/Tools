# TRC Vessel Designer

A comprehensive PyQt6 GUI application for Thermal Reaction Chamber (TRC) vessel design, including refractory layer configuration, volume calculations, and residence time analysis. Uses the shared TRCGeometryEngine for accurate geometric calculations.

## Purpose

The TRC Vessel Designer provides engineering calculations for designing refractory-lined thermal processing vessels. Key applications include:

- High-temperature reactor vessel sizing
- Refractory lining design and material selection
- Thermal insulation optimization
- Residence time calculations for chemical processes
- Mass estimation for structural support design

## Key Features

- **Cylinder-Cone Geometry**: Combined cylindrical body with conical outlet section
- **Multi-Layer Refractory**: Support for 2-4 refractory layers with presets
- **Volume Calculations**: Net internal, gross external, and refractory volumes
- **Mass Estimation**: Layer-by-layer mass calculations
- **Residence Time Analysis**: Based on operating conditions and flow rate
- **Preset Configurations**: Standard, high-temperature, and economy options
- **Real-Time Updates**: Interactive parameter adjustment with instant recalculation

## Installation / Prerequisites

### System Requirements

- Python 3.10 or higher
- Windows, macOS, or Linux

### Dependencies

```bash
pip install PyQt6
pip install upstream-drift-tools  # Shared TRC geometry engine
```

### Running the Application

```bash
# From the Tools repository root
python -m src.trc_vessel_designer.launch_pyqt6

# Or via web interface
python src/trc_vessel_designer/launch_web.py
```

## Usage Instructions

1. **Set Vessel Dimensions**: Cylinder height/diameter, cone height/bottom diameter
2. **Select Refractory Preset**: Standard, High Temperature, or Economy configuration
3. **Set Operating Conditions**: Temperature, pressure, gas flow rate
4. **View Results**: Volume summary, mass summary, residence time, layer details
5. **Export/Save**: State persistence for later retrieval

## Input Parameters

### Vessel Dimensions

| Parameter            | Range  | Units  | Default | Description                     |
| -------------------- | ------ | ------ | ------- | ------------------------------- |
| Cylinder Height      | 12-300 | inches | 72      | Main body height (6 ft default) |
| Cylinder Diameter    | 6-120  | inches | 24      | Inside diameter (2 ft default)  |
| Cone Height          | 0-100  | inches | 24      | Conical section height          |
| Cone Bottom Diameter | 1-24   | inches | 6       | Outlet diameter                 |

### Operating Conditions

| Parameter             | Range      | Units | Default | Description          |
| --------------------- | ---------- | ----- | ------- | -------------------- |
| Operating Temperature | 500-2000   | C     | 1400    | Process temperature  |
| Operating Pressure    | 50-500     | kPa   | 101.325 | System pressure      |
| Gas Flow Rate         | 100-50,000 | m3/hr | 2000    | Volumetric flow rate |

### Refractory Presets

**Standard (3-layer)**
| Layer | Material | Thickness | Density |
|-------|----------|-----------|---------|
| 1 | High-Alumina Working Lining | 6.0" | 150 lb/ft3 |
| 2 | Insulating Firebrick | 4.5" | 60 lb/ft3 |
| 3 | Microporous Insulation | 1.0" | 20 lb/ft3 |

**High Temperature (4-layer)**
| Layer | Material | Thickness | Density |
|-------|----------|-----------|---------|
| 1 | Chrome-Alumina Working | 8.0" | 200 lb/ft3 |
| 2 | High-Alumina Backup | 4.0" | 150 lb/ft3 |
| 3 | Insulating Firebrick | 4.5" | 60 lb/ft3 |
| 4 | Calcium Silicate Board | 2.0" | 30 lb/ft3 |

**Economy (2-layer)**
| Layer | Material | Thickness | Density |
|-------|----------|-----------|---------|
| 1 | Castable Refractory | 8.0" | 130 lb/ft3 |
| 2 | Ceramic Fiber Blanket | 2.0" | 12 lb/ft3 |

## Output Format

### Volume Summary

- **Net Internal Volume**: Usable process volume (ft3)
- **Gross Volume**: Total vessel external volume (ft3)
- **Refractory Volume**: Lining material volume (ft3)

### Mass Summary

- **Total Refractory Mass**: Combined weight of all layers (lb)
- **Outside Surface Area**: External vessel surface (ft2)

### Residence Time

- **Residence Time**: Gas retention time (seconds)
- **Void Diameter**: Internal working diameter (inches)

### Layer Details

Per-layer breakdown showing:

- Layer name and material
- Volume (ft3)
- Mass (lb)

## Mathematical Models

### Cylindrical Section Volume

```
V_cylinder = pi * r^2 * h
```

Where:

- r = Internal radius (after refractory)
- h = Cylinder height

### Conical Section Volume

```
V_cone = (1/3) * pi * h * (R1^2 + R1*R2 + R2^2)
```

Where:

- h = Cone height
- R1 = Top radius (cylinder diameter)
- R2 = Bottom radius (outlet)

### Total Internal Volume

```
V_internal = V_cylinder + V_cone
```

### Refractory Layer Volume

For each layer:

```
V_layer = V_outer - V_inner

V_outer_cyl = pi * (r + t)^2 * h
V_inner_cyl = pi * r^2 * h
```

Where t = layer thickness

### Layer Mass

```
Mass = V_layer * density
```

### Residence Time

```
tau = V_internal / Q_actual

Q_actual = Q_std * (T_op / T_std) * (P_std / P_op)
```

Where:

- tau = Residence time (s)
- V_internal = Internal volume (m3)
- Q = Volumetric flow rate (m3/s)
- Temperature and pressure corrections for actual conditions

### Surface Area

```
A_cylinder = 2 * pi * R_outer * H
A_cone = pi * (R1 + R2) * sqrt((R1-R2)^2 + h^2)
A_top = pi * R_outer^2
A_total = A_cylinder + A_cone + A_top
```

## Example Usage

**Scenario**: Design a thermal oxidizer vessel

**Input**:

- Cylinder: 72" height x 24" diameter
- Cone: 24" height, 6" bottom diameter
- Refractory: Standard (3-layer, 11.5" total)
- Operating: 1400C, 101.325 kPa, 2000 m3/hr

**Expected Results**:

- Net Internal Volume: ~2.5 ft3
- Gross Volume: ~15 ft3
- Total Mass: ~1,500 lb
- Residence Time: ~1.5 seconds

## Troubleshooting

### Common Issues

| Issue                    | Cause                          | Solution                               |
| ------------------------ | ------------------------------ | -------------------------------------- |
| Negative internal volume | Refractory thicker than vessel | Increase vessel diameter               |
| Zero residence time      | Flow rate too high             | Increase vessel size or reduce flow    |
| High mass estimate       | Dense refractory selected      | Use lighter insulation layers          |
| Calculation error        | Invalid dimensions             | Ensure cone bottom < cylinder diameter |

### Design Guidelines

| Temperature Range | Recommended Preset | Notes                             |
| ----------------- | ------------------ | --------------------------------- |
| < 1200C           | Economy            | Basic insulation sufficient       |
| 1200-1500C        | Standard           | Good balance of cost/performance  |
| > 1500C           | High Temperature   | Chrome-alumina for severe service |

### Refractory Selection Criteria

| Factor               | Consideration                                    |
| -------------------- | ------------------------------------------------ |
| Hot Face Temperature | Must exceed max operating temp                   |
| Thermal Shock        | Cyclic operations need shock-resistant materials |
| Chemical Attack      | Match lining to process chemistry                |
| Mechanical Wear      | Dense materials for abrasive service             |

## Related Tools

- **[Electrode Advisor](../electrode_advisor/README.md)**: Electric heating system design
- **[Thermal Profile Predictor](../thermal_profile_predictor/README.md)**: Temperature distribution modeling
- **[Scrubber Calculator](../scrubber_calculator/README.md)**: Off-gas treatment design
- **[Flare Calculator](../flare_calculator/README.md)**: Combustion system sizing

## References

- Carniglia, S.C. & Barna, G.L. "Handbook of Industrial Refractories Technology"
- Routschka, G. & Wuthnow, H. "Refractory Materials"
- ASTM C401: Classification of Alumina and Alumina-Silicate Castable Refractories
- API Standard 560: Fired Heaters for General Refinery Service
- NFPA 86: Standard for Ovens and Furnaces

## Current Features

- Purpose: Thermal Reaction Chamber vessel design tool
- Category: Process Simulation
- Python files in tool path: 11
- Surface support: PyQt6=implemented, Web manifest=yes, Web implementation=present
- Test visibility: 0 name-matched test files under tests/

## Implementation State

- PyQt6 launcher: Implemented
- Web surface declared in manifest: Yes
- Web surface implementation: Implemented
- README last reviewed: 2026-02-27

## Implementation Gaps

- No name-matched tests detected in repository-level tests/.
