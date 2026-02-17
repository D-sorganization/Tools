# Syngas Water Calculator

A comprehensive PyQt6 GUI application for calculating water content, dew point, and condensation risk in syngas systems. This tool is essential for process engineers working with synthesis gas production and handling.

## Purpose

The Syngas Water Calculator determines the moisture content in syngas streams and assesses condensation risks. Understanding water content is critical for:

- Preventing corrosion in downstream equipment
- Avoiding hydrate formation in pipelines
- Optimizing gas treatment processes
- Meeting pipeline and process specifications

## Key Features

- **Multiple Water Content Units**: Results in mole fraction, mg/Nm3, ppmv, g/m3, and lb/MMscf
- **Dew Point Calculation**: Determines the temperature at which water condensation begins
- **Condensation Risk Assessment**: Real-time risk level evaluation with color-coded indicators
- **Multiple Vapor Pressure Methods**: Antoine, Buck, IAPWS-IF97, Magnus, and Auto selection
- **Preset Gas Compositions**: Typical syngas, biomass gasification, coal gasification, natural gas reforming
- **Temperature Margin Analysis**: Calculates safety margin above dew point

## Installation / Prerequisites

### System Requirements

- Python 3.10 or higher
- Windows, macOS, or Linux

### Dependencies

```bash
pip install PyQt6
pip install upstream-drift-tools  # Core calculation engine
```

### Running the Application

```bash
# From the Tools repository root
python -m src.syngas_water_calculator.launch_pyqt6

# Or directly
python src/syngas_water_calculator/launch_pyqt6.py
```

## Usage Instructions

1. **Set Temperature**: Enter the operating temperature in degrees Celsius (-50 to 400 C)
2. **Set Pressure**: Enter the system pressure in bar (0.1 to 500 bar)
3. **Select Composition**: Choose a preset gas composition from the dropdown
4. **Select Method**: Choose vapor pressure calculation method (Auto recommended)
5. **Calculate**: Click "Calculate Water Content" to generate results

## Input Parameters

| Parameter          | Range      | Units | Description                |
| ------------------ | ---------- | ----- | -------------------------- |
| Temperature        | -50 to 400 | C     | Operating gas temperature  |
| Pressure           | 0.1 to 500 | bar   | System absolute pressure   |
| Gas Composition    | Preset     | -     | Syngas composition profile |
| Calculation Method | Selection  | -     | Vapor pressure correlation |

### Gas Composition Presets

| Preset                | CO  | H2  | CO2 | CH4 | N2  | Description          |
| --------------------- | --- | --- | --- | --- | --- | -------------------- |
| Typical Syngas        | 25% | 20% | 15% | 5%  | 35% | General purpose      |
| Biomass Gasification  | 18% | 15% | 12% | 4%  | 51% | Wood/biomass derived |
| Coal Gasification     | 30% | 25% | 8%  | 3%  | 34% | Coal-derived syngas  |
| Natural Gas Reforming | 12% | 50% | 18% | 2%  | 18% | SMR product gas      |

## Output Format

### Water Content Results

- **Mole Fraction**: Dimensionless water mole fraction (6 decimal places)
- **mg/Nm3**: Milligrams water per normal cubic meter
- **ppmv**: Parts per million by volume
- **g/m3**: Grams per actual cubic meter
- **lb/MMscf**: Pounds per million standard cubic feet

### Risk Assessment

- **Temperature Margin**: Degrees above dew point (C)
- **Risk Level**: Low / Medium / High / Critical (color-coded)
- **Recommended Min Temp**: Suggested operating temperature

## Mathematical Models

### Vapor Pressure Correlations

**Antoine Equation** (default for 0-100C):

```
log10(P) = A - B / (C + T)
```

Where P is in mmHg, T in Celsius. For water: A=8.07131, B=1730.63, C=233.426

**Buck Equation** (high accuracy):

```
P = 0.61121 * exp((18.678 - T/234.5) * (T/(257.14 + T)))
```

Where P is in kPa, T in Celsius.

**IAPWS-IF97** (industrial standard):
Uses the International Association for the Properties of Water and Steam formulation for high-precision calculations across wide temperature ranges.

**Magnus Equation**:

```
P = 6.1094 * exp(17.625 * T / (T + 243.04))
```

Where P is in hPa, T in Celsius.

### Water Content Calculation

```
y_water = P_sat / P_total
```

Where y_water is the mole fraction, P_sat is saturation pressure, P_total is system pressure.

### Dew Point Estimation

Iteratively solves for temperature where:

```
P_sat(T_dew) = y_water * P_total
```

## Example Usage

**Scenario**: Syngas from a biomass gasifier at 40C and 30 bar

1. Input: Temperature = 40C, Pressure = 30 bar
2. Select: "Biomass Gasification" composition
3. Method: "Auto (Recommended)"
4. Click Calculate

**Expected Results**:

- Water Content: ~2,450 mg/Nm3
- Dew Point: ~25C
- Temperature Margin: +15C
- Risk Level: Low (green)

## Troubleshooting

### Common Issues

| Issue              | Cause                         | Solution                                          |
| ------------------ | ----------------------------- | ------------------------------------------------- |
| Import Error       | Missing upstream_drift_tools  | Install: `pip install upstream-drift-tools`       |
| Negative dew point | Very dry gas at high pressure | Normal for low water content scenarios            |
| High risk warning  | Operating near dew point      | Increase operating temperature or reduce pressure |
| Results show "--"  | Calculation exception         | Check input ranges are valid                      |

### Error Messages

- **"Error: ..."**: Check that temperature and pressure are within valid ranges
- **Critical Risk**: Condensation is imminent or occurring; increase temperature margin

## Related Tools

- **[Syngas Compression Calculator](../syngas_compression/README.md)**: Multi-stage compression with water dropout analysis
- **[WGS Reactor Calculator](../wgs_reactor/README.md)**: Water-gas shift equilibrium calculations
- **[Acid Gas Dewpoint Calculator](../acid_gas_dewpoint/README.md)**: H2S and CO2 dewpoint analysis
- **[Scrubber Calculator](../scrubber_calculator/README.md)**: Gas cleaning and water removal

## References

- IAPWS-IF97: Industrial Formulation for Water and Steam Properties
- Perry's Chemical Engineers' Handbook, 9th Edition
- GPSA Engineering Data Book, 14th Edition
