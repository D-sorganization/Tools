# Water-Gas Shift (WGS) Reactor Calculator

A comprehensive PyQt6 GUI application for water-gas shift reactor equilibrium calculations, kinetics analysis, and catalyst bed sizing. Essential for hydrogen production and syngas conditioning processes.

## Purpose

The WGS Reactor Calculator models the water-gas shift reaction for process design and optimization. The WGS reaction is fundamental to:

- Hydrogen production from syngas
- Carbon monoxide removal for fuel cell applications
- Adjusting H2/CO ratios for Fischer-Tropsch synthesis
- Ammonia plant feed gas preparation
- Integrated gasification combined cycle (IGCC) systems

## Key Features

- **Equilibrium Composition**: Calculates outlet gas composition at thermodynamic equilibrium
- **Temperature-Dependent K_eq**: Van't Hoff equation for equilibrium constant
- **Multiple Shift Configurations**: High Temperature Shift (HTS), Low Temperature Shift (LTS), Two-Stage
- **Reactor Sizing**: Catalyst volume, reactor dimensions, GHSV calculations
- **Heat Duty Analysis**: Reaction heat release calculations
- **H2/CO Ratio Optimization**: Target ratio achievement analysis
- **Real-Time Results**: Interactive composition comparison tables

## Installation / Prerequisites

### System Requirements

- Python 3.10 or higher
- Windows, macOS, or Linux

### Dependencies

```bash
pip install PyQt6
pip install numpy
```

### Running the Application

```bash
# From the Tools repository root
python -m src.wgs_reactor.launch_pyqt6

# Or via web interface
python src/wgs_reactor/launch_web.py
```

## Usage Instructions

1. **Set Reactor Configuration**: Temperature (C), pressure (bar), steam/CO ratio, feed rate (kmol/h)
2. **Enter Feed Composition**: CO, H2, CO2, H2O, N2 in mol%
3. **Select Shift Type**: HTS (350-450C), LTS (180-250C), or Two-Stage
4. **Calculate**: Click "Calculate WGS Performance"
5. **Review Results**: Examine conversion, composition, sizing, and heat duty

## Input Parameters

### Reactor Configuration

| Parameter      | Range     | Units   | Default | Description                   |
| -------------- | --------- | ------- | ------- | ----------------------------- |
| Temperature    | 200-800   | C       | 400     | Reactor operating temperature |
| Pressure       | 1-100     | bar     | 25      | Operating pressure            |
| Steam/CO Ratio | 0.5-10    | mol/mol | 2.0     | Excess steam ratio            |
| Feed Rate      | 1-100,000 | kmol/h  | 100     | Total molar feed rate         |

### Feed Composition (mol%)

| Component | Range  | Default | Description                            |
| --------- | ------ | ------- | -------------------------------------- |
| CO        | 0-100% | 25%     | Carbon monoxide (reactant)             |
| H2        | 0-100% | 20%     | Hydrogen (product)                     |
| CO2       | 0-100% | 10%     | Carbon dioxide (product)               |
| H2O       | 0-100% | 5%      | Steam (reactant, additional via ratio) |
| N2        | 0-100% | 40%     | Nitrogen (inert)                       |

### Shift Type Configurations

| Type      | Temperature  | Catalyst               | Typical Conversion |
| --------- | ------------ | ---------------------- | ------------------ |
| HTS       | 350-450C     | Fe-Cr (iron-chromium)  | 70-90%             |
| LTS       | 180-250C     | Cu-Zn-Al (copper-zinc) | 85-95%             |
| Two-Stage | HTS then LTS | Both                   | 95-99%             |

## Output Format

### Summary Cards

- **CO Conversion**: Percentage of CO converted (color-coded)
- **H2/CO Ratio**: Outlet hydrogen to CO ratio
- **Heat Duty**: Reactor heat release (kW)
- **Equilibrium K**: Temperature-dependent equilibrium constant

### Composition Comparison Table

| Species | Inlet (mol%) | Outlet (mol%) |
| ------- | ------------ | ------------- |
| CO      | Input value  | Calculated    |
| H2      | Input value  | Calculated    |
| CO2     | Input value  | Calculated    |
| H2O     | Input value  | Calculated    |

### Reactor Sizing Table

- **Reactor Volume**: Total vessel volume (m3)
- **Catalyst Volume**: Active catalyst volume (m3)
- **Diameter**: Vessel diameter (m)
- **Length**: Vessel length (m)
- **GHSV**: Gas hourly space velocity (h-1)
- **Heat Released**: Per mole CO converted (kJ/mol)

## Mathematical Models

### Water-Gas Shift Reaction

```
CO + H2O <-> CO2 + H2     deltaH = -41.2 kJ/mol
```

The reaction is exothermic and equilibrium-limited. Lower temperatures favor conversion but slower kinetics.

### Equilibrium Constant (Van't Hoff Equation)

```
ln(K_eq) = -deltaH/(R*T) + deltaS/R

K_eq = exp(-deltaH/(R*T) + deltaS/R)
```

Where:

- deltaH = -41,200 J/mol (reaction enthalpy)
- deltaS = -42.1 J/(mol-K) (reaction entropy)
- R = 8.314 J/(mol-K)
- T = Temperature (K)

### Equilibrium Composition

At equilibrium:

```
K_eq = (n_CO2 * n_H2) / (n_CO * n_H2O)

K_eq = ((n_CO2_0 + x) * (n_H2_0 + x)) / ((n_CO_0 - x) * (n_H2O_0 - x))
```

Where x = extent of reaction (moles converted)

Solving the quadratic equation:

```
(K-1)*x^2 + [K*(n_CO_0 + n_H2O_0) + n_CO2_0 + n_H2_0]*x + K*n_CO_0*n_H2O_0 - n_CO2_0*n_H2_0 = 0
```

### CO Conversion

```
X_CO = x / n_CO_0 * 100%
```

### Reactor Sizing

```
Reactor Volume = Feed Rate / GHSV
Catalyst Volume = Reactor Volume * 0.8
Diameter = (4 * V / (pi * L/D))^(1/3)
Length = Diameter * L/D  (L/D = 3 typical)
Heat Duty = Feed Rate * X_CO * 41.2 / 3.6  (kW)
```

## Example Usage

**Scenario**: High Temperature Shift reactor for hydrogen production

**Input**:

- Temperature: 400C (673 K)
- Pressure: 25 bar
- Steam/CO Ratio: 2.0
- Feed Rate: 100 kmol/h
- Feed: CO=25%, H2=20%, CO2=10%, H2O=5%, N2=40%

**Expected Results**:

- CO Conversion: ~85-90%
- H2/CO Ratio: >10
- Equilibrium K: ~10-15
- Reactor Volume: ~0.03 m3
- Heat Duty: ~300 kW

## Troubleshooting

### Common Issues

| Issue                | Cause                  | Solution                               |
| -------------------- | ---------------------- | -------------------------------------- |
| Low conversion       | Temperature too high   | Reduce temperature or use two-stage    |
| H2/CO ratio infinite | Complete CO conversion | Normal for high conversion cases       |
| Zero heat duty       | No CO in feed          | Verify feed composition                |
| Large reactor size   | High feed rate         | Increase GHSV or use multiple reactors |

### Temperature Guidelines

- **HTS (Fe-Cr catalyst)**: 350-450C, avoid <300C (low activity) or >500C (sintering)
- **LTS (Cu-Zn catalyst)**: 180-250C, avoid >280C (catalyst deactivation)
- **Equilibrium favors conversion at lower T**, but kinetics require minimum temperature

### Catalyst Considerations

| Catalyst | Operating Range | Poison Sensitivity         | Life      |
| -------- | --------------- | -------------------------- | --------- |
| Fe-Cr    | 350-450C        | Sulfur tolerant            | 3-5 years |
| Cu-Zn-Al | 180-250C        | Sulfur, chloride sensitive | 2-4 years |

## Related Tools

- **[Syngas Compression Calculator](../syngas_compression/README.md)**: Post-WGS compression
- **[Syngas Water Calculator](../syngas_water_calculator/README.md)**: Water content analysis
- **[Steam Engine Calculator](../steam_engine_calculator/README.md)**: Steam generation sizing
- **[PSA Package](../psa_package/README.md)**: Hydrogen purification

## References

- Twigg, M.V. "Catalyst Handbook", 2nd Edition
- Newsome, D.S. "The Water-Gas Shift Reaction", Catalysis Reviews
- Rase, H.F. "Chemical Reactor Design for Process Plants"
- Smith, R.J.B. et al. "A Review of the Water Gas Shift Reaction Kinetics"
