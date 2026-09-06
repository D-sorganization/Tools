# Chapter 3 — Process Engineering Calculators

**Parent Document:** [Tools User Manual](./TOOLS_USER_MANUAL.md)

This chapter documents all process engineering calculators in the Tools monorepo. Each calculator's core engine resides in `src/shared/python/upstream_drift_tools/process_calculators/` and can be used independently of any GUI.

---

## 3.1 Acid Gas Dewpoint Calculator

**Source:** `process_calculators/acid_gas_dewpoint_calculator.py`
**GUI:** `src/acid_gas_dewpoint/` (PyQt6 + Web)
**Status:** ✅ Fully Implemented

### 3.1.1 Purpose

Predicts dewpoint temperatures of acid gases (HF, HCl, H₂S) in syngas/water vapor mixtures. Used to determine safe operating temperatures to prevent corrosive condensation in gas processing equipment.

### 3.1.2 Mathematical Model

**Antoine Equation** for vapor pressure calculation:

$$\log_{10}(P) = A - \frac{B}{C + T}$$

where:

- $P$ = vapor pressure (mmHg)
- $T$ = temperature (°C)
- $A$, $B$, $C$ = component-specific Antoine constants

**Antoine Constants (Perry's Chemical Engineers' Handbook, 8th Ed.):**

| Component | $A$     | $B$     | $C$     |
| --------- | ------- | ------- | ------- |
| H₂O       | 8.07131 | 1730.63 | 233.426 |
| HF        | 7.158   | 1111.0  | 235.0   |
| HCl       | 7.960   | 1118.0  | 240.0   |
| H₂S       | 6.987   | 884.0   | 240.0   |

**Inverse Antoine Equation** for dewpoint calculation:

$$T_{dew} = \frac{B}{A - \log_{10}(P_{partial}/133.322)} - C$$

where $P_{partial}$ is the partial pressure of the component in Pa.

**Partial Pressure Calculation:**

$$P_{partial,i} = y_i \cdot P_{total}$$

where $y_i$ is the mole fraction of component $i$.

**Overall Dewpoint:**

$$T_{dew,overall} = \max(T_{dew,\text{H}_2\text{O}},\ T_{dew,\text{HF}},\ T_{dew,\text{HCl}},\ T_{dew,\text{H}_2\text{S}})$$

**Dewpoint Margin:**

$$\Delta T_{margin} = T_{operating} - T_{dew,overall}$$

**Condensation Risk Assessment:**

| Margin ($\Delta T$)              | Risk Level                       |
| -------------------------------- | -------------------------------- |
| $\Delta T < 0$                   | HIGH — condensation occurring    |
| $0 \leq \Delta T < 10°\text{C}$  | MEDIUM — within 10°C of dewpoint |
| $10 \leq \Delta T < 30°\text{C}$ | LOW — safe margin                |
| $\Delta T \geq 30°\text{C}$      | VERY LOW — large safety margin   |

### 3.1.3 Vapor Pressure Methods

The calculator supports four vapor pressure calculation methods:

1. **Antoine** (default): Standard Antoine equation with literature constants
2. **Extended Antoine**: Dual-range constants for H₂O (below/above 100°C)
3. **Thermo**: Uses the `thermo` Python library (Peng-Robinson EOS)
4. **CoolProp**: Uses the CoolProp library (IAPWS-IF97 for water)

### 3.1.4 Inputs and Outputs

**Inputs:**

| Parameter         | Type   | Units | Range                                               |
| ----------------- | ------ | ----- | --------------------------------------------------- |
| Temperature       | float  | °C    | -100 to 400                                         |
| Pressure          | float  | bar   | 0.1 to 300                                          |
| H₂O mole fraction | float  | —     | 0 to 1                                              |
| HF mole fraction  | float  | —     | 0 to 1                                              |
| HCl mole fraction | float  | —     | 0 to 1                                              |
| H₂S mole fraction | float  | —     | 0 to 1                                              |
| Method            | string | —     | 'antoine', 'extended_antoine', 'thermo', 'coolprop' |

**Outputs (`DewpointResult`):**

| Field                | Type  | Description                       |
| -------------------- | ----- | --------------------------------- |
| `overall_dewpoint_c` | float | Overall dewpoint temperature (°C) |
| `limiting_component` | str   | Component with highest dewpoint   |
| `dewpoint_margin_c`  | float | Safety margin (°C)                |
| `condensation_risk`  | str   | Risk assessment string            |
| `h2o_dewpoint_c`     | float | Water dewpoint (°C)               |
| `hf_dewpoint_c`      | float | HF dewpoint (°C)                  |
| `hcl_dewpoint_c`     | float | HCl dewpoint (°C)                 |
| `h2s_dewpoint_c`     | float | H₂S dewpoint (°C)                 |

### 3.1.5 Predefined Compositions

| Preset               | H₂O  | HF     | HCl    | H₂S   |
| -------------------- | ---- | ------ | ------ | ----- |
| Typical Syngas       | 0.15 | 0.0001 | 0.0002 | 0.001 |
| High Acid Content    | 0.20 | 0.001  | 0.002  | 0.005 |
| Coal Gasification    | 0.12 | 0.0005 | 0.001  | 0.003 |
| Biomass Gasification | 0.18 | 0.0002 | 0.0005 | 0.002 |

### 3.1.6 Literature Sources

- Perry's Chemical Engineers' Handbook, 8th Ed.
- NIST Chemistry WebBook
- CRC Handbook of Chemistry and Physics
- IAPWS-IF97 Formulation
- Journal of Chemical & Engineering Data (2001, 2003)

---

## 3.2 Baghouse Calculator

**Source:** `process_calculators/baghouse_calculator.py`
**GUI:** `src/baghouse_calculator/` (PyQt6 + Web)
**Status:** ✅ Fully Implemented

### 3.2.1 Purpose

Calculates baghouse filter performance including solid (carbon + ash) removal rates, collection drum fill times, outlet temperature after heat loss, and air-to-cloth ratios.

### 3.2.2 Mathematical Model

**Temperature Drop from Heat Loss:**

$$\Delta T = \frac{Q_{loss}}{\dot{m}_{gas} \cdot C_p}$$

where:

- $Q_{loss}$ = heat loss rate (W)
- $\dot{m}_{gas}$ = gas mass flow rate (kg/s)
- $C_p$ = specific heat capacity of gas mixture (J/(kg·K))

**Mixture Heat Capacity (Ideal Gas Approximation):**

$$C_{p,mix} = \frac{\sum_i y_i \cdot C_{p,i}^{mol}}{\sum_i y_i \cdot M_i}$$

where $C_{p,i}^{mol}$ is the molar heat capacity (J/(mol·K)) and $M_i$ is the molecular weight (kg/mol).

**Approximate $C_p^{mol}$ values at ~500K:**

| Species | $C_p^{mol}$ (J/(mol·K)) | $M$ (kg/mol) |
| ------- | ----------------------- | ------------ |
| H₂      | 29.1                    | 0.002        |
| CO      | 29.2                    | 0.028        |
| CO₂     | 41.3                    | 0.044        |
| H₂O     | 35.5                    | 0.018        |
| N₂      | 29.5                    | 0.028        |
| CH₄     | 44.5                    | 0.016        |

**Volumetric Flow Rate (Ideal Gas Law):**

$$\dot{V}_{actual} = \frac{\dot{n} \cdot R \cdot T}{P}$$

$$\dot{V}_{std} = \frac{\dot{n} \cdot R \cdot T_{STP}}{P_{atm}}$$

where $T_{STP} = 273.15\ \text{K}$ and $P_{atm} = 101325\ \text{Pa}$.

**Solids Removal:**

$$\dot{m}_{carbon,out} = \dot{m}_{carbon,in} \cdot \eta_{carbon}$$

$$\dot{m}_{ash,out} = \dot{m}_{ash,in} \cdot \eta_{ash}$$

$$\dot{m}_{total} = \dot{m}_{carbon,out} + \dot{m}_{ash,out}$$

**Drum Fill Time:**

$$t_{fill} = \frac{\rho_{solid} \cdot V_{drum}}{\dot{m}_{total}}$$

**Air-to-Cloth Ratio:**

$$ACR = \frac{\dot{V}_{ACFM}}{A_{bag}}$$

where $A_{bag}$ is the total bag filter area (ft²) and the result is in ft/min.

### 3.2.3 Inputs and Outputs

**Inputs:**

| Parameter                   | Units | Description                      |
| --------------------------- | ----- | -------------------------------- |
| `gas_flow_kg_s`             | kg/s  | Gas mass flow rate               |
| `inlet_temp_k`              | K     | Inlet temperature                |
| `pressure_pa`               | Pa    | System pressure                  |
| `composition`               | dict  | Gas composition (mole fractions) |
| `solid_carbon_in_kg_hr`     | kg/hr | Carbon input rate                |
| `ash_in_kg_hr`              | kg/hr | Ash input rate                   |
| `carbon_removal_efficiency` | 0–1   | Carbon removal efficiency        |
| `ash_removal_efficiency`    | 0–1   | Ash removal efficiency           |
| `heat_loss_w`               | W     | Heat loss rate                   |
| `drum_volume_m3`            | m³    | Collection drum volume           |
| `solid_density_kg_m3`       | kg/m³ | Density of collected solids      |
| `bag_area_ft2`              | ft²   | Total bag filter area            |

**Outputs (`BaghouseResult`):**

| Field                       | Units  | Description                    |
| --------------------------- | ------ | ------------------------------ |
| `carbon_removed_rate`       | kg/hr  | Carbon removal rate            |
| `ash_removed_rate`          | kg/hr  | Ash removal rate               |
| `total_solids_removed_rate` | kg/hr  | Total solids removal rate      |
| `drum_fill_time_hours`      | hr     | Time to fill drum              |
| `drum_fill_time_days`       | days   | Time to fill drum              |
| `flow_acfm`                 | cfm    | Actual cubic feet per minute   |
| `flow_scfm`                 | cfm    | Standard cubic feet per minute |
| `air_to_cloth_ratio`        | ft/min | Air-to-cloth ratio             |
| `outlet_temperature_c`      | °C     | Outlet temperature             |

---

## 3.3 Flare Calculator

**Source:** `process_calculators/flare_calculator.py`
**GUI:** `src/flare_calculator/` (PyQt6 + Web)
**Status:** ✅ Fully Implemented

### 3.3.1 Purpose

Designs flare systems for syngas/waste gas disposal. Calculates flare height, diameter, heat release, radiation zones, and combustion efficiency using simplified API 521 methods.

### 3.3.2 Mathematical Model

**Mixture Molecular Weight:**

$$M_{mix} = \sum_i y_i \cdot M_i$$

**Mixture Heating Value:**

$$HV_{mix} = \sum_i y_i \cdot HV_i$$

**Heat Release Rate:**

$$\dot{Q} = \frac{\dot{m}_{total} \cdot HV_{mix}}{3600} \quad [\text{kW}]$$

where $\dot{m}_{total}$ is in kg/hr and $HV_{mix}$ is in kJ/kg.

**Gas Density (Ideal Gas Law):**

$$\rho = \frac{P \cdot M_{mix}}{R_{specific} \cdot T} = \frac{P}{\frac{R}{M_{mix}} \cdot T}$$

**Flare Diameter (API 521 Simplified):**

$$A = \frac{\dot{m}_{total}/3600}{\rho \cdot u_{target}}$$

$$D = \sqrt{\frac{4A}{\pi}}$$

where $u_{target} = 170\ \text{m/s}$ (smokeless operation).

**Flare Height (Point Source Radiation Model):**

$$H = \sqrt{\frac{\varepsilon \cdot \dot{Q}}{4\pi \cdot I_{target}}}$$

where:

- $\varepsilon = 0.3$ (flame emissivity)
- $I_{target} = 1.6\ \text{kW/m}^2$ (safe ground-level radiation)
- Minimum height: $H_{min} = 10\ \text{m}$

**Radiation Zone Distances:**

$$D_{zone} = \sqrt{\frac{\varepsilon \cdot \dot{Q}}{4\pi \cdot I_{zone}}}$$

| Zone    | Radiation Level ($I_{zone}$) | Description                      |
| ------- | ---------------------------- | -------------------------------- |
| Lethal  | 37.5 kW/m²                   | Immediate fatality risk          |
| Damage  | 12.5 kW/m²                   | Equipment/structure damage       |
| Safe    | 1.6 kW/m²                    | Safe for continuous access       |
| Comfort | 0.5 kW/m²                    | Comfortable for extended periods |

### 3.3.3 Gas Properties Database

| Gas   | $M$ (g/mol) | $HV$ (kJ/kg) | $C_p$ (kJ/(kg·K)) |
| ----- | ----------- | ------------ | ----------------- |
| H₂    | 2.016       | 119,930      | 14.3              |
| CO    | 28.01       | 10,100       | 1.04              |
| CH₄   | 16.04       | 50,010       | 2.22              |
| C₂H₆  | 30.07       | 47,520       | 1.75              |
| C₃H₈  | 44.10       | 46,360       | 1.67              |
| C₄H₁₀ | 58.12       | 45,720       | 1.66              |
| H₂S   | 34.08       | 16,500       | 1.05              |

---

## 3.4 Scrubber Calculator

**Source:** `process_calculators/scrubber_calculator.py`
**GUI:** `src/scrubber_calculator/` (PyQt6 + Web)
**Status:** ✅ Fully Implemented

### 3.4.1 Purpose

Designs countercurrent packed bed scrubbers for syngas acid gas removal. Implements industry-standard methods from Perry's, Treybal, and Eckert.

### 3.4.2 Mathematical Model

**Gas Density (Ideal Gas Law):**

$$\rho_G = \frac{P \cdot M}{R \cdot T}$$

where $R = 8314.46\ \text{J/(kmol·K)}$ and $M$ is in kg/kmol.

**Gas Viscosity (Sutherland's Formula):**

$$\mu = \mu_{ref} \cdot \left(\frac{T}{T_{ref}}\right)^{3/2} \cdot \frac{T_{ref} + S}{T + S} \cdot \left(\frac{M}{29}\right)^{0.25}$$

where $\mu_{ref} = 1.8 \times 10^{-5}\ \text{Pa·s}$, $T_{ref} = 300\ \text{K}$, $S = 110.4\ \text{K}$.

**Flooding Velocity (Eckert's Correlation, Perry's 9th Ed.):**

Flow parameter:

$$X = \frac{L}{G} \cdot \sqrt{\frac{\rho_G}{\rho_L}}$$

Capacity parameter at flooding:

$$Y_{flood} = C_{flood} \cdot \exp(-1.5 \cdot X^{0.5})$$

Flooding gas mass flux:

$$G'_{flood} = \sqrt{\frac{Y_{flood} \cdot \rho_G \cdot \rho_L \cdot g}{F_p \cdot (\mu_L/\mu_{water})^{0.1}}}$$

Flooding velocity:

$$u_{flood} = \frac{G'_{flood}}{\rho_G}$$

**Pressure Drop (Eckert's Generalized Correlation):**

$$Y = \frac{G'^2 \cdot F_p \cdot (\mu_L/\mu_{water})^{0.1}}{\rho_G \cdot \rho_L \cdot g}$$

$$\frac{\Delta P}{Z} = \alpha \cdot Y^\beta \cdot (1 + \gamma \cdot X)$$

where $\alpha = 85\ \text{Pa/m}$, $\beta = 1.1$, $\gamma = 3.5$.

**Number of Transfer Units (NTU):**

For irreversible absorption (chemical scrubbing with NaOH):

$$NTU = \ln\left(\frac{y_{in}}{y_{out}}\right)$$

**Height of Transfer Unit (HTU):**

$$HTU = \frac{C_H}{k_La \cdot a_s \cdot (L/G)^n}$$

where $C_H$, $n$ are packing-specific constants.

**Required Packed Height:**

$$Z = NTU \times HTU \times SF$$

where $SF$ is the safety factor (typically 1.2).

**Column Diameter:**

$$A = \frac{\dot{V}_G}{u_{design}}$$

$$D = \sqrt{\frac{4A}{\pi}}$$

where $u_{design} = u_{flood} \times (\% flood / 100)$.

**Caustic (NaOH) Requirement:**

| Reaction                   | Stoichiometry | NaOH/mol |
| -------------------------- | ------------- | -------- |
| HCl + NaOH → NaCl + H₂O    | 1:1           | 1.0      |
| SO₂ + 2NaOH → Na₂SO₃ + H₂O | 1:2           | 2.0      |
| H₂S + 2NaOH → Na₂S + 2H₂O  | 1:2           | 2.0      |
| HF + NaOH → NaF + H₂O      | 1:1           | 1.0      |
| CO₂ + 2NaOH → Na₂CO₃ + H₂O | 1:2           | 2.0      |

With 15% excess factor applied.

**Heat Transfer Duty:**

$$Q_{total} = Q_{sensible} + Q_{latent}$$

$$Q_{sensible} = \dot{m}_{gas} \cdot C_p \cdot \Delta T$$

$$Q_{latent} = \dot{m}_{condensed} \cdot h_{fg}$$

**Henry's Law Constants:**

$$H(T) = H_{ref} \cdot \exp\left(-\frac{\Delta H_{soln}}{R} \cdot \left(\frac{1}{T} - \frac{1}{T_{ref}}\right)\right)$$

| Gas | $H_{ref}$ (Pa)     | $\Delta H_{soln}$ (J/mol) |
| --- | ------------------ | ------------------------- |
| HCl | $2.04 \times 10^6$ | -17,600                   |
| SO₂ | $4.39 \times 10^4$ | -26,700                   |
| H₂S | $5.68 \times 10^5$ | -19,300                   |
| HF  | $1.27 \times 10^7$ | -15,200                   |
| CO₂ | $1.64 \times 10^8$ | -20,100                   |

### 3.4.3 Packing Database

| Packing Type          | Material | Size (mm) | $a_s$ (m²/m³) | $\varepsilon$ | $F_p$ (1/m) | $C_{flood}$ |
| --------------------- | -------- | --------- | ------------- | ------------- | ----------- | ----------- |
| Ceramic Raschig Rings | Ceramic  | 50        | 95            | 0.74          | 155         | 0.082       |
| Metal Pall Rings      | SS       | 50        | 112           | 0.95          | 66          | 0.11        |
| Plastic Cascade Rings | PP       | 50        | 105           | 0.92          | 72          | 0.10        |
| Structured 250Y       | SS       | —         | 250           | 0.98          | 33          | 0.15        |

---

## 3.5 Pressure Drop Calculator

**Source:** `process_calculators/pressure_drop_calculator/`
**GUI:** `src/pressure_drop_calculator/` (PyQt6 + Web)
**Status:** ✅ Fully Implemented (modular sub-package)

### 3.5.1 Purpose

Calculates pressure drop in piping systems including straight pipe, fittings, and equipment using the Darcy-Weisbach equation.

### 3.5.2 Mathematical Model

**Darcy-Weisbach Equation:**

$$\Delta P = f \cdot \frac{L}{D} \cdot \frac{\rho \cdot v^2}{2}$$

where:

- $f$ = Darcy friction factor
- $L$ = pipe length (m)
- $D$ = pipe inner diameter (m)
- $\rho$ = fluid density (kg/m³)
- $v$ = flow velocity (m/s)

**Reynolds Number:**

$$Re = \frac{\rho \cdot v \cdot D}{\mu}$$

**Friction Factor (Colebrook-White Equation):**

$$\frac{1}{\sqrt{f}} = -2 \log_{10}\left(\frac{\varepsilon/D}{3.7} + \frac{2.51}{Re \cdot \sqrt{f}}\right)$$

Solved iteratively. For laminar flow ($Re < 2300$):

$$f = \frac{64}{Re}$$

**Fitting Losses:**

$$\Delta P_{fitting} = K \cdot \frac{\rho \cdot v^2}{2}$$

where $K$ is the loss coefficient from the fitting database.

### 3.5.3 Sub-modules

| Module                                       | Description                  |
| -------------------------------------------- | ---------------------------- |
| `engine/pressure_drop_calculation_engine.py` | Core calculation engine      |
| `models/pressure_drop_data_models.py`        | Data models and result types |
| `utils/fitting_loss_coefficients.py`         | Fitting K-value database     |
| `utils/flow_rate_converter.py`               | Flow unit conversions        |
| `utils/gas_properties.py`                    | Gas property calculations    |
| `utils/pipe_database.py`                     | Standard pipe size database  |

---

## 3.6 Flow Rate Converter

**Source:** `src/flow_rate_converter/`
**GUI:** `src/flow_rate_converter/` (PyQt6 + Web)
**Status:** ✅ Fully Implemented

### 3.6.1 Purpose

Converts between volumetric and mass flow rate units for gases, accounting for temperature and pressure conditions.

### 3.6.2 Mathematical Model

**Ideal Gas Law Conversion:**

$$\dot{V}_2 = \dot{V}_1 \cdot \frac{T_2}{T_1} \cdot \frac{P_1}{P_2}$$

**Mass to Volume:**

$$\dot{V} = \frac{\dot{m}}{\rho} = \frac{\dot{m} \cdot R \cdot T}{P \cdot M}$$

**Standard Flow (SCFM at 60°F, 1 atm):**

$$\dot{V}_{SCFM} = \dot{V}_{actual} \cdot \frac{T_{std}}{T_{actual}} \cdot \frac{P_{actual}}{P_{std}}$$

---

## 3.7 Syngas Water Calculator

**Source:** `process_calculators/syngas_water_calculator.py`
**GUI:** `src/syngas_water_calculator/` (PyQt6 + Web)
**Status:** ✅ Fully Implemented

### 3.7.1 Purpose

Calculates water vapor content, saturation conditions, and condensation behavior in syngas streams.

### 3.7.2 Mathematical Model

**Modified Buck Equation (Buck, 1981):**

$$P_{vap} = A \cdot \exp\left(\frac{(B - T/C) \cdot T}{D + T}\right)$$

where $A = 0.61115\ \text{kPa}$, $B = 23.036$, $C = 279.82\ \text{K}$, $D = 333.7\ \text{K}$, and $T$ is in °C.

**Antoine Equation for Water:**

$$\log_{10}(P_{mmHg}) = 8.07131 - \frac{1730.63}{233.426 + T_C}$$

**IAPWS-IF97:** When available via CoolProp, uses the industrial formulation for water/steam properties.

---

## 3.8 Syngas Compression Calculator

**Source:** `process_calculators/syngas_compression_calculator.py`
**GUI:** `src/syngas_compression/` (PyQt6)
**Status:** ✅ Fully Implemented

### 3.8.1 Purpose

Comprehensive syngas compression analysis including multistage compression, water dropout, horsepower requirements, and heat rise analysis.

### 3.8.2 Mathematical Model

**Isentropic Compression:**

Outlet temperature:

$$T_{out,s} = T_{in} \cdot \left(\frac{P_{out}}{P_{in}}\right)^{(\gamma - 1)/\gamma}$$

Isentropic work:

$$W_s = \frac{\gamma}{\gamma - 1} \cdot R \cdot T_{in} \cdot \left[\left(\frac{P_{out}}{P_{in}}\right)^{(\gamma-1)/\gamma} - 1\right]$$

Actual work:

$$W_{actual} = \frac{W_s}{\eta_s}$$

**Polytropic Compression:**

$$T_{out} = T_{in} \cdot \left(\frac{P_{out}}{P_{in}}\right)^{(n-1)/n}$$

$$W_{poly} = \frac{n}{n - 1} \cdot R \cdot T_{in} \cdot \left[\left(\frac{P_{out}}{P_{in}}\right)^{(n-1)/n} - 1\right] / \eta_{poly}$$

**Isothermal Compression:**

$$W_{iso} = \frac{R \cdot T \cdot \ln(P_{out}/P_{in})}{\eta}$$

**Power Requirement:**

$$HP = \frac{\dot{n} \cdot 1000 / 3600 \cdot W_{actual}}{745.7}$$

where $\dot{n}$ is in kmol/h.

**Heat Capacity Ratio:**

$$\gamma_{mix} = \sum_i y_i \cdot \gamma_i$$

| Species | $\gamma$ |
| ------- | -------- |
| H₂      | 1.41     |
| CO      | 1.40     |
| CO₂     | 1.30     |
| CH₄     | 1.32     |
| N₂      | 1.40     |
| H₂O     | 1.33     |

**Water Dropout Calculation:**

$$RH = \frac{y_{H_2O} \cdot P_{total}}{P_{vap}(T)}$$

If $RH > 1.0$: water condenses. Maximum vapor content:

$$y_{max} = \frac{P_{vap}(T)}{P_{total}}$$

---

## 3.9 WGS Reactor Calculator

**Source:** `process_calculators/wgs_reactor_calculator.py`
**GUI:** `src/wgs_reactor/` (PyQt6)
**Status:** ✅ Fully Implemented

### 3.9.1 Purpose

Water-Gas Shift (WGS) reactor design and analysis for H₂/CO ratio adjustment.

### 3.9.2 Chemical Reaction

$$\text{CO} + \text{H}_2\text{O} \rightleftharpoons \text{CO}_2 + \text{H}_2 \qquad \Delta H° = -41.2\ \text{kJ/mol}$$

### 3.9.3 Mathematical Model

**Equilibrium Constant (Van't Hoff Equation):**

$$\ln K_{eq} = -\frac{\Delta H°}{R \cdot T} + \frac{\Delta S°}{R}$$

where $\Delta H° = -41200\ \text{J/mol}$ and $\Delta S° = -42.1\ \text{J/(mol·K)}$.

**Gibbs Free Energy Minimization:**

The equilibrium composition is found by minimizing total Gibbs free energy:

$$G_{total} = \sum_i n_i \cdot \left[G_{f,i}° + R \cdot T \cdot \ln\left(\frac{p_i}{P°}\right)\right]$$

subject to the constraint:

$$n_i = n_{i,0} + \nu_i \cdot \xi$$

where $\xi$ is the extent of reaction and $\nu_i$ is the stoichiometric coefficient.

The minimization is solved using `scipy.optimize.minimize` with bounds $0 \leq \xi \leq \min(n_{CO,0}, n_{H_2O,0})$.

**Reactor Sizing:**

$$V_{reactor} = \frac{\dot{F}}{GHSV}$$

where $GHSV = 3000\ \text{h}^{-1}$ (typical).

Dimensions with $L/D = 3$:

$$D = \left(\frac{4V}{\pi \cdot 3}\right)^{1/3}$$

$$L = 3D$$

**Heat Duty:**

$$\dot{Q} = \dot{F} \cdot X_{CO}/100 \cdot 41.2 / 3.6 \quad [\text{kW}]$$

### 3.9.4 Thermodynamic Data

| Species | $\Delta H_f°$ (kJ/mol) | $S°$ (J/(mol·K)) |
| ------- | ---------------------- | ---------------- |
| CO      | -110.525               | 197.66           |
| CO₂     | -393.509               | 213.74           |
| H₂      | 0.0                    | 130.68           |
| H₂O (g) | -241.826               | 188.83           |

---

## 3.10 Thermal Profile Predictor

**Source:** `process_calculators/thermal_profile_predictor.py`
**Status:** ✅ Fully Implemented

### 3.10.1 Purpose

Predicts temperature profiles for heated vessels using ODE integration and can fit thermal parameters to observed data.

### 3.10.2 Mathematical Model

**Heating ODE:**

$$\frac{dT}{dt} = \frac{Q_{in}(t) - h \cdot (T - T_{amb})}{C_{th}}$$

where:

- $C_{th}$ = thermal mass (J/K)
- $h$ = heat loss coefficient (W/K)
- $T_{amb}$ = ambient temperature (K)
- $Q_{in}(t)$ = power input function (W)

Solved using `scipy.integrate.solve_ivp`.

**Parameter Fitting:**

Uses `scipy.optimize.curve_fit` to determine $C_{th}$ and $h$ from observed temperature data.

---

## 3.11 ODE Solver

**Source:** `process_calculators/ode_solver.py`
**Status:** ✅ Fully Implemented

### 3.11.1 Purpose

General-purpose ODE solver for systems of differential equations defined symbolically.

### 3.11.2 Mathematical Model

Solves systems of the form:

$$\frac{dy_i}{dt} = f_i(t, y_1, y_2, \ldots, y_n, p_1, p_2, \ldots, p_m)$$

where expressions $f_i$ are defined as strings and parsed using SymPy. The system is solved using `scipy.integrate.solve_ivp`.

**Example:**

```python
derivs = {"T": "k*(T_env - T)"}
params = {"k": 0.3, "T_env": 350.0}
solver = ODESolver(derivs, params)
sol = solver.solve((0.0, 20.0), [300.0])
```

---

## 3.12 Electrode Advancement Calculator

**Source:** `process_calculators/electrode_advancement_calculator.py`
**Status:** ⚠️ Stub Implementation

### 3.12.1 Purpose

Calculates electrode consumption and slip rates for arc furnaces.

### 3.12.2 Current Implementation

$$C = r \cdot I \cdot t$$

where:

- $C$ = consumption (inches)
- $r$ = consumption rate (inches per kAh) — currently hardcoded at 0.5
- $I$ = current (kA)
- $t$ = time (hours)

### 3.12.3 Implementation Gaps

- [ ] Configurable consumption rate based on electrode material
- [ ] Temperature-dependent consumption models
- [ ] Multiple electrode type support (graphite, Söderberg)
- [ ] Slip rate calculation
- [ ] Wear profile modeling
- [ ] Integration with thermal models

---

## 3.13 Financial Calculator

**Source:** `process_calculators/financial_calculator.py`
**GUI:** `src/financial_calculator/` (PyQt6)
**Status:** ✅ Fully Implemented

### 3.13.1 Purpose

Comprehensive financial modeling for plant operations including revenue projections, operating costs, and return metrics.

### 3.13.2 Mathematical Model

**Annual Volumes:**

$$V_{feedstock} = C_{plant} \cdot D_{operating} \cdot U$$

$$V_{product} = V_{feedstock} \cdot Y_{product}$$

where $Y_{product} = 0.85$ (85% yield).

**Revenue:**

$$R_{total} = V_{product} \cdot P_{product} + V_{byproduct} \cdot P_{byproduct}$$

**EBITDA:**

$$EBITDA = R_{total} - C_{variable} - C_{fixed}$$

**Net Income:**

$$EBIT = EBITDA - D$$

$$EBT = EBIT - I_{expense}$$

$$\text{Net Income} = EBT - \text{Taxes}$$

where $D = \text{TCI}/\text{years}$ (straight-line depreciation) and $I_{expense} = \text{TCI} \cdot d_{ratio} \cdot r$.

**Return Metrics:**

$$ROE = \frac{\text{Net Income}}{\text{Equity}}$$

$$ROA = \frac{\text{Net Income}}{\text{TCI}}$$

$$\text{Payback} = \frac{\text{TCI}}{\text{Net Income} + D}$$

**Multi-Year Projections:**

- Revenue escalation: 2% annually
- Cost inflation: 3% annually (2.5% for utilities)

---

## 3.14 PSA Package

**Source:** `process_calculators/psa_package/`
**GUI:** `src/psa_package/` (PyQt6 + Web)
**Status:** ✅ Fully Implemented (modular sub-package)

### 3.14.1 Purpose

Pressure Swing Adsorption (PSA) modeling for gas separation, particularly H₂ purification from syngas.

### 3.14.2 Sub-modules

| Module          | Description               |
| --------------- | ------------------------- |
| `psa_model.py`  | Core PSA cycle simulation |
| `psa_gui.py`    | PyQt6 GUI interface       |
| `psa_webapp.py` | Web interface             |

---

## 3.15 Steam Engine Calculator

**Source:** `src/steam_engine_calculator/`
**GUI:** `src/steam_engine_calculator/` (PyQt6)
**Status:** ✅ Implemented

### 3.15.1 Purpose

Steam engine thermodynamic analysis and performance calculations.

---

## 3.16 TRC Vessel Designer

**Source:** `src/trc_vessel_designer/`
**GUI:** `src/trc_vessel_designer/` (PyQt6)
**Status:** ✅ Implemented

### 3.16.1 Purpose

Thermal reactor/converter vessel design tool for sizing and mechanical design of pressure vessels.

---

## 3.17 Optimizer GUI (legacy shim)

**Source:** `process_calculators/optimization.py`
**GUI:** `src/optimizer_gui/launch_pyqt6.py` (compatibility launcher)
**Status:** ⚠️ Legacy shim — retired standalone GUI (Tools #3983)

The standalone PyQt6 optimizer GUI that used to live in `src/optimizer_gui/`
was consolidated into `src/movement_optimizer/`, and its drifted vendored copy
of the swing/chain models was deleted. The directory now only carries a hidden
catalog registration and a compatibility launcher that starts the canonical
Movement Optimizer application. For the maintained optimization surface, see
the Movement Optimizer documentation.

---

## 3.18 Multi-Parameter Analysis

**Source:** `process_calculators/multi_param_analysis.py`
**GUI:** `src/multi_param_analysis/` (PyQt6)
**Status:** ✅ Implemented

### 3.18.1 Purpose

Systematic exploration of multi-dimensional parameter spaces for process optimization, generating contour plots and sensitivity analyses.

---

## 3.19 Inertia Calculator

**Source:** `src/inertia_calculator/`
**GUI:** `src/inertia_calculator/` (PyQt6)
**Status:** ✅ Implemented

### 3.19.1 Purpose

Calculates moments of inertia for standard geometric shapes, useful for mechanical design and robotics applications.

### 3.19.2 Mathematical Model

**Solid Cylinder (about central axis):**

$$I = \frac{1}{2} m r^2$$

**Solid Sphere:**

$$I = \frac{2}{5} m r^2$$

**Rectangular Prism (about center):**

$$I_{xx} = \frac{1}{12} m (b^2 + c^2)$$

**Parallel Axis Theorem:**

$$I = I_{cm} + m \cdot d^2$$

---

_[← Back to Main Manual](./TOOLS_USER_MANUAL.md) | [Next: Signal Processing Toolkit →](./04_signal_toolkit.md)_
