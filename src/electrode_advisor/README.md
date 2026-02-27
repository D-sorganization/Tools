# Electrode Advisor

A comprehensive PyQt6 GUI application for electrode advancement calculations, wear rate analysis, and replacement scheduling in electric glass melting furnaces. Uses the shared ThreePhaseElectricalModelEnhanced engine for accurate electrical system modeling.

## Purpose

The Electrode Advisor provides real-time monitoring and optimization tools for molybdenum electrode systems in glass melting furnaces. Key applications include:

- Monitoring electrode wear and advancement rates
- Calculating power distribution across three-phase systems
- Predicting electrode replacement schedules
- Optimizing electrical efficiency and power balance
- Preventing electrode failures and production interruptions

## Key Features

- **3-Phase Electrical Modeling**: Complete three-phase power analysis with phase-to-phase calculations
- **Real-Time Resistance Calculation**: Glass bath resistance modeling based on geometry and temperature
- **Electrode Depth Management**: Individual depth control for 3 electrodes
- **Power Summary**: Total power, average resistance, and phase-by-phase breakdown
- **Physical Parameter Integration**: Bath diameter, tip diameter, metal depth, temperature
- **Glass Properties Interface**: Connection to external glass property calculators
- **State Persistence**: Save and restore calculation states

## Installation / Prerequisites

### System Requirements

- Python 3.10 or higher
- Windows, macOS, or Linux

### Dependencies

```bash
pip install PyQt6
pip install numpy
pip install matplotlib
pip install upstream-drift-tools  # Shared electrical model engine
```

### Running the Application

```bash
# From the Tools repository root
python -m src.electrode_advisor.launch_pyqt6

# Or via web interface
python src/electrode_advisor/launch_web.py
```

## Usage Instructions

1. **Enter Electrical Measurements**: Current (A) and voltage (V) for phases 1-2, 2-3, 3-1
2. **Set Electrode Depths**: Individual depth in inches for each of 3 electrodes
3. **Configure Physical Parameters**: Bath diameter, tip diameter, metal depth, temperature
4. **View Results**: Power summary, phase resistances, and calculated currents
5. **Connect External Calculators**: Optionally link to glass property calculators

## Input Parameters

### 3-Phase Electrical Measurements

| Parameter           | Range    | Units | Default | Description       |
| ------------------- | -------- | ----- | ------- | ----------------- |
| Current (Phase 1-2) | 0-10,000 | A     | 300     | Phase 1-2 current |
| Current (Phase 2-3) | 0-10,000 | A     | 300     | Phase 2-3 current |
| Current (Phase 3-1) | 0-10,000 | A     | 300     | Phase 3-1 current |
| Voltage (Phase 1-2) | 0-1,000  | V     | 100     | Phase 1-2 voltage |
| Voltage (Phase 2-3) | 0-1,000  | V     | 100     | Phase 2-3 voltage |
| Voltage (Phase 3-1) | 0-1,000  | V     | 100     | Phase 3-1 voltage |

### Electrode Depths

| Electrode   | Range | Units  | Default | Description               |
| ----------- | ----- | ------ | ------- | ------------------------- |
| Electrode 1 | 0-50  | inches | 12.0    | Depth below glass surface |
| Electrode 2 | 0-50  | inches | 12.0    | Depth below glass surface |
| Electrode 3 | 0-50  | inches | 12.0    | Depth below glass surface |

### Physical Parameters

| Parameter        | Range    | Units  | Default | Description              |
| ---------------- | -------- | ------ | ------- | ------------------------ |
| Bath Diameter    | 10-500   | inches | 120.0   | Glass bath diameter      |
| Tip Diameter     | 1-100    | inches | 24.0    | Electrode tip diameter   |
| Metal Depth      | 0-20     | inches | 2.0     | Molten metal layer depth |
| Bath Temperature | 500-2000 | C      | 1350    | Glass melt temperature   |

## Output Format

### Power Summary

- **Total Power**: Combined three-phase power (kW)
- **Avg Resistance**: Mean phase-to-phase resistance (Ohms)

### Phase Results Table

| Phase | Resistance (Ohm) | Current (A) | Power (kW) |
| ----- | ---------------- | ----------- | ---------- |
| 1-2   | Calculated       | Calculated  | Calculated |
| 2-3   | Calculated       | Calculated  | Calculated |
| 3-1   | Calculated       | Calculated  | Calculated |

### System Information

- Engine: ThreePhaseElectricalModelEnhanced
- Source: upstream_drift_tools
- Configuration display with current parameters

## Mathematical Models

### Power Calculation

```
P_phase = V_phase * I_phase / 1000  (kW)
P_total = P_12 + P_23 + P_31
```

### Resistance Calculation

The shared ThreePhaseElectricalModelEnhanced uses:

```
R_total = R_glass + R_tip + R_electrode

R_glass = rho * L / A
```

Where:

- rho = Glass resistivity (temperature-dependent, Ohm-cm)
- L = Current path length (cm)
- A = Effective cross-sectional area (cm2)

### Glass Resistivity

Temperature-dependent resistivity model:

```
rho(T) = rho_0 * exp(-E_a / (k * T))
```

Where:

- rho_0 = Reference resistivity
- E_a = Activation energy
- k = Boltzmann constant
- T = Temperature (K)

### Electrode Wear Rate

```
Wear_rate = f(current_density, temperature, glass_composition)

Advancement_rate = Wear_rate + thermal_expansion
```

### Replacement Scheduling

```
Remaining_life = (Total_length - Current_position) / Wear_rate

Replacement_date = Current_date + Remaining_life
```

## Example Usage

**Scenario**: Monitor a 3-phase electric glass melter

**Input**:

- Currents: 300A, 300A, 300A (balanced)
- Voltages: 100V, 100V, 100V
- Electrode depths: 12", 12", 12"
- Bath: 120" diameter, 1350C

**Expected Results**:

- Total Power: ~90 kW
- Phase Resistances: ~0.33 Ohm each
- Individual Power: ~30 kW per phase

**Unbalanced Scenario**:

- Currents: 350A, 280A, 320A
- Results show power imbalance and potential wear rate differences

## Troubleshooting

### Common Issues

| Issue                    | Cause                         | Solution                       |
| ------------------------ | ----------------------------- | ------------------------------ |
| Zero resistance          | Electrode depth too deep      | Check depth settings           |
| High power imbalance     | Electrode wear or positioning | Balance electrode depths       |
| Calculation error        | Invalid input parameters      | Verify all inputs are in range |
| Glass properties missing | No glass interface connected  | Connect glass calculator       |

### Status Messages

- **System Ready**: Application initialized
- **Calculating...**: Computation in progress
- **Calculation complete**: Results updated
- **Error: ...**: Check input parameters

### Electrode Health Indicators

| Indicator                        | Status   | Action               |
| -------------------------------- | -------- | -------------------- |
| Balanced power (< 10% deviation) | Normal   | Monitor              |
| Moderate imbalance (10-20%)      | Caution  | Plan adjustment      |
| High imbalance (> 20%)           | Warning  | Immediate attention  |
| Rapid wear rate increase         | Critical | Schedule replacement |

## Related Tools

- **[TRC Vessel Designer](../trc_vessel_designer/README.md)**: Thermal vessel design with refractory
- **[Thermal Profile Predictor](../thermal_profile_predictor/README.md)**: Temperature distribution modeling
- **[Inertia Calculator](../inertia_calculator/README.md)**: Mechanical system analysis

## References

- Trier, W. "Glass Furnaces: Design, Construction and Operation"
- Shelby, J.E. "Introduction to Glass Science and Technology"
- Tooley, F.V. "Handbook of Glass Manufacture"
- IEEE Standard 519: Harmonic Control in Electrical Power Systems
- NFPA 86: Standard for Ovens and Furnaces

## Current Features

- Purpose: AC Electrode Advancement Module for electrode system analysis
- Category: Process Simulation
- Python files in tool path: 44
- Surface support: PyQt6=implemented, Web manifest=yes, Web implementation=present
- Test visibility: 0 name-matched test files under tests/

## Implementation State

- PyQt6 launcher: Implemented
- Web surface declared in manifest: Yes
- Web surface implementation: Implemented
- README last reviewed: 2026-02-27

## Implementation Gaps

- No name-matched tests detected in repository-level tests/.
