# Chapter 11 — Physical Constants and Unit Conversions

**Parent Document:** [Tools User Manual](./TOOLS_USER_MANUAL.md)
**Source:** `src/shared/python/upstream_drift_tools/process_calculators/constants.py`

---

## 11.1 Fundamental Physical Constants

All constants follow NIST CODATA 2018 recommended values.

| Constant | Symbol | Value | Unit | Variable Name |
|----------|--------|-------|------|---------------|
| Universal Gas Constant | $R$ | $8.314462618$ | $\text{J/(mol·K)}$ | `R_GAS_J_MOL_K` |
| Universal Gas Constant | $R$ | $8314.46$ | $\text{J/(kmol·K)}$ | `R_GAS_J_KMOL_K` |
| Universal Gas Constant | $R$ | $0.08206$ | $\text{L·atm/(mol·K)}$ | `R_GAS_L_ATM_MOL_K` |
| Standard Gravity | $g$ | $9.80665$ | $\text{m/s}^2$ | `STD_GRAVITY` |
| Avogadro Number | $N_A$ | $6.02214076 \times 10^{23}$ | $\text{mol}^{-1}$ | `AVOGADRO` |
| Boltzmann Constant | $k_B$ | $1.380649 \times 10^{-23}$ | $\text{J/K}$ | `BOLTZMANN` |
| Stefan-Boltzmann | $\sigma$ | $5.670374419 \times 10^{-8}$ | $\text{W/(m}^2\text{·K}^4\text{)}$ | `STEFAN_BOLTZMANN` |
| Speed of Light | $c$ | $2.99792458 \times 10^8$ | $\text{m/s}$ | `SPEED_OF_LIGHT` |

---

## 11.2 Standard Conditions

| Condition | Symbol | Value | Variable Name |
|-----------|--------|-------|---------------|
| Standard Temperature | $T_{STP}$ | $273.15\ \text{K}$ (0°C) | `STP_TEMP_K` |
| Standard Pressure | $P_{STP}$ | $101325\ \text{Pa}$ | `STD_ATM_PA` |
| Standard Pressure | $P_{STP}$ | $1.01325\ \text{bar}$ | `STD_ATM_BAR` |
| Normal Temperature | $T_{NTP}$ | $293.15\ \text{K}$ (20°C) | `NTP_TEMP_K` |
| Water Freezing Point | — | $273.15\ \text{K}$ | `WATER_FREEZE_K` |
| Water Boiling Point | — | $373.15\ \text{K}$ | `WATER_BOIL_K` |
| Absolute Zero Tolerance | — | $10^{-9}$ | `ATOL_ZERO` |

---

## 11.3 Molecular Weights

From NIST Standard Reference Database.

| Species | Formula | $M$ (g/mol) | $M$ (kg/mol) |
|---------|---------|-------------|---------------|
| Hydrogen | H₂ | 2.016 | 0.002016 |
| Carbon Monoxide | CO | 28.010 | 0.02801 |
| Carbon Dioxide | CO₂ | 44.010 | 0.04401 |
| Water | H₂O | 18.015 | 0.01802 |
| Methane | CH₄ | 16.043 | 0.01604 |
| Nitrogen | N₂ | 28.014 | 0.02801 |
| Oxygen | O₂ | 31.998 | 0.03200 |
| Hydrogen Sulfide | H₂S | 34.081 | 0.03408 |
| Hydrogen Fluoride | HF | 20.006 | 0.02001 |
| Hydrogen Chloride | HCl | 36.461 | 0.03646 |
| Sulfur Dioxide | SO₂ | 64.066 | 0.06407 |
| Argon | Ar | 39.948 | 0.03995 |
| Ethane | C₂H₆ | 30.069 | 0.03007 |
| Propane | C₃H₈ | 44.096 | 0.04410 |

---

## 11.4 Conversion Functions

### 11.4.1 Temperature Conversions

$$T_K = T_C + 273.15$$

$$T_C = (T_F - 32) \times \frac{5}{9}$$

$$T_R = T_F + 459.67$$

### 11.4.2 Pressure Conversions

| From | To Pa | Multiplier |
|------|-------|------------|
| atm | Pa | $\times\ 101325$ |
| bar | Pa | $\times\ 100000$ |
| psi | Pa | $\times\ 6894.757$ |
| mmHg | Pa | $\times\ 133.322$ |
| inH₂O | Pa | $\times\ 249.089$ |

### 11.4.3 Flow Rate Conversions

**Actual to Standard:**

$$\dot{V}_{std} = \dot{V}_{act} \times \frac{T_{std}}{T_{act}} \times \frac{P_{act}}{P_{std}}$$

**Volume to Mass:**

$$\dot{m} = \dot{V} \times \frac{P \cdot M}{R \cdot T}$$

### 11.4.4 Energy Conversions

| From | To J | Multiplier |
|------|------|------------|
| kJ | J | $\times\ 1000$ |
| BTU | J | $\times\ 1055.06$ |
| cal | J | $\times\ 4.184$ |
| kWh | J | $\times\ 3.6 \times 10^6$ |
| hp·hr | J | $\times\ 2.685 \times 10^6$ |

### 11.4.5 Length Conversions

| From | To m | Multiplier |
|------|------|------------|
| ft | m | $\times\ 0.3048$ |
| in | m | $\times\ 0.0254$ |
| cm | m | $\times\ 0.01$ |
| mm | m | $\times\ 0.001$ |
| mi | m | $\times\ 1609.344$ |

---

*[← Development Tools](./10_development_tools.md) | [Back to Manual](./TOOLS_USER_MANUAL.md) | [Next: Implementation Gaps →](./12_implementation_gaps.md)*
