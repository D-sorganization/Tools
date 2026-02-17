# Syngas Compression Calculator

A comprehensive PyQt6 GUI application for multi-stage syngas compression analysis, including power requirements, temperature rise, water dropout, and process safety evaluation.

## Purpose

The Syngas Compression Calculator provides detailed analysis for compressing synthesis gas from atmospheric or low pressure to process conditions. Key applications include:

- Designing compression trains for gasification plants
- Evaluating intercooling requirements
- Predicting water condensation during compression
- Estimating power consumption and operating costs
- Identifying process safety concerns

## Key Features

- **Multi-Stage Compression**: Up to 4 compression stages with individual parameters
- **Three Compression Models**: Isentropic, polytropic, and isothermal calculations
- **Water Dropout Analysis**: Predicts condensation at each stage
- **Intercooling Support**: Optional cooling between stages
- **Process Monitoring**: Safety warnings and recommendations
- **Visualization**: Temperature, pressure, and power plots
- **Configurable Efficiency**: Stage-by-stage efficiency settings

## Installation / Prerequisites

### System Requirements

- Python 3.10 or higher
- Windows, macOS, or Linux

### Dependencies

```bash
pip install PyQt6
pip install matplotlib
pip install numpy
pip install upstream-drift-tools  # Core calculation engine
```

### Running the Application

```bash
# From the Tools repository root
python -m src.syngas_compression.launch_pyqt6

# Or via web interface
python src/syngas_compression/launch_web.py
```

## Usage Instructions

1. **Define Gas Composition**: Enter syngas composition in mol% (H2, CO, CO2, CH4, N2, H2O, Ar)
2. **Set Process Conditions**: Flow rate (kmol/h), inlet temperature (C), inlet pressure (bar)
3. **Configure Stages**: Set inlet/outlet pressures and efficiency for each active stage
4. **Select Compression Type**: Isentropic, polytropic, or isothermal
5. **Enable/Disable Intercooling**: Cooling to 40C between stages
6. **Calculate**: View results in Results, Analysis, and Plots tabs

## Input Parameters

### Gas Composition

| Component | Range  | Default | Description      |
| --------- | ------ | ------- | ---------------- |
| H2        | 0-100% | 20.0%   | Hydrogen content |
| CO        | 0-100% | 25.0%   | Carbon monoxide  |
| CO2       | 0-100% | 15.0%   | Carbon dioxide   |
| CH4       | 0-100% | 5.0%    | Methane          |
| N2        | 0-100% | 30.0%   | Nitrogen         |
| H2O       | 0-100% | 5.0%    | Water vapor      |
| Ar        | 0-100% | 0.0%    | Argon            |

### Process Conditions

| Parameter         | Range      | Units  | Default |
| ----------------- | ---------- | ------ | ------- |
| Flow Rate         | 0-10,000   | kmol/h | 100     |
| Inlet Temperature | -50 to 500 | C      | 40      |
| Inlet Pressure    | 0.1-1000   | bar    | 1.0     |

### Compression Stages

| Parameter       | Range    | Units | Description                  |
| --------------- | -------- | ----- | ---------------------------- |
| Inlet Pressure  | 0.1-1000 | bar   | Stage inlet pressure         |
| Outlet Pressure | 0.1-1000 | bar   | Stage outlet pressure        |
| Efficiency      | 50-100   | %     | Isentropic efficiency        |
| Active          | Checkbox | -     | Include stage in calculation |

## Output Format

### Stage-by-Stage Results

- **Inlet/Outlet Temperature**: K and C
- **Heat Rise**: Temperature increase (K)
- **Pressure Ratio**: Outlet/inlet pressure
- **Power Required**: HP (horsepower)
- **Water Dropout**: Condensed water (mol%)

### Summary Results

- **Total Power Required**: Sum of all stages (HP)
- **Final Temperature**: Discharge temperature (K, C)
- **Final Pressure**: Discharge pressure (bar)
- **Total Water Dropout**: Cumulative condensation (mol%)
- **Average Efficiency**: Overall compression efficiency (%)

### Process Analysis

- **Critical Warnings**: Temperature or pressure limit violations
- **Concerns**: Equipment or process issues
- **Recommendations**: Suggested improvements

## Mathematical Models

### Isentropic Compression

```
T2_isen = T1 * (P2/P1)^((gamma-1)/gamma)

W_isen = (gamma/(gamma-1)) * R * T1 * ((P2/P1)^((gamma-1)/gamma) - 1)

W_actual = W_isen / eta_isen
```

Where:

- T1, T2 = Inlet, outlet temperatures (K)
- P1, P2 = Inlet, outlet pressures (bar)
- gamma = Heat capacity ratio (Cp/Cv)
- R = Gas constant (8.314 J/mol-K)
- eta_isen = Isentropic efficiency

### Polytropic Compression

```
T2 = T1 * (P2/P1)^((n-1)/n)

W_poly = (n/(n-1)) * R * T1 * ((P2/P1)^((n-1)/n) - 1) / eta_poly
```

Where n = polytropic exponent (typically equals gamma for ideal gases)

### Isothermal Compression

```
W_iso = R * T * ln(P2/P1) / eta_iso
T2 = T1  (constant temperature)
```

### Power Calculation

```
Power (HP) = (Flow_rate * 1000 / 3600) * W_actual / 745.7
```

Where flow rate is in kmol/h, work in J/mol, and 745.7 W = 1 HP

### Water Dropout

Water condenses when partial pressure exceeds saturation:

```
If (y_H2O * P_total) > P_sat(T):
    Water_dropout = y_H2O - P_sat(T)/P_total
```

## Example Usage

**Scenario**: Compress 100 kmol/h syngas from 1 bar to 81 bar in 4 stages

**Input**:

- Composition: H2=20%, CO=25%, CO2=15%, CH4=5%, N2=30%, H2O=5%
- Inlet: 40C, 1 bar, 100 kmol/h
- Stages: 1->3, 3->9, 9->27, 27->81 bar (85% efficiency each)
- Intercooling: Enabled

**Expected Results**:

- Total Power: ~180-220 HP
- Final Temperature: ~150-180C (with intercooling)
- Water Dropout: Significant in stages 2-4

## Troubleshooting

### Common Issues

| Issue                    | Cause                     | Solution                          |
| ------------------------ | ------------------------- | --------------------------------- |
| High temperature warning | Insufficient intercooling | Enable intercooling or add stages |
| Negative pressure ratio  | Outlet < Inlet pressure   | Verify stage pressures            |
| Low efficiency warning   | Efficiency < 70%          | Check compressor maintenance      |
| No active stages         | All checkboxes unchecked  | Enable at least one stage         |

### Warning Thresholds

- **Temperature**: Warning at 200C, Critical at 250C
- **Pressure**: Warning at 100 bar
- **Power**: Warning at 1000 HP total
- **Efficiency**: Warning below 70%

## Related Tools

- **[Syngas Water Calculator](../syngas_water_calculator/README.md)**: Detailed dew point analysis
- **[WGS Reactor Calculator](../wgs_reactor/README.md)**: Shift reactor design
- **[Pressure Drop Calculator](../pressure_drop_calculator/README.md)**: Piping pressure losses
- **[Flare Calculator](../flare_calculator/README.md)**: Emergency relief sizing

## References

- Walas, S.M. "Chemical Process Equipment: Selection and Design"
- Ludwig, E.E. "Applied Process Design for Chemical and Petrochemical Plants"
- GPSA Engineering Data Book, 14th Edition
- API Standard 617: Axial and Centrifugal Compressors
