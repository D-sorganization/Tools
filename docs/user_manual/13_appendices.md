# Chapter 13 — Appendices

**Parent Document:** [Tools User Manual](./TOOLS_USER_MANUAL.md)

---

## Appendix A: Mathematical Reference

### A.1 Thermodynamic Equations

**Ideal Gas Law:**

$$PV = nRT$$

$$\rho = \frac{PM}{RT}$$

**Antoine Equation:**

$$\log_{10}(P^{sat}) = A - \frac{B}{C + T}$$

**Van't Hoff Equation:**

$$\ln K = -\frac{\Delta H°}{RT} + \frac{\Delta S°}{R}$$

$$\frac{d \ln K}{dT} = \frac{\Delta H°}{RT^2}$$

**Clausius-Clapeyron Equation:**

$$\frac{dP}{dT} = \frac{\Delta H_{vap}}{T \cdot \Delta V}$$

**Gibbs Free Energy:**

$$G = H - TS$$

$$\Delta G° = -RT \ln K$$

$$\Delta G = \Delta G° + RT \ln Q$$

### A.2 Fluid Mechanics Equations

**Darcy-Weisbach:**

$$h_f = f \cdot \frac{L}{D} \cdot \frac{v^2}{2g}$$

**Reynolds Number:**

$$Re = \frac{\rho v D}{\mu} = \frac{vD}{\nu}$$

**Colebrook-White:**

$$\frac{1}{\sqrt{f}} = -2 \log_{10}\left(\frac{\varepsilon/D}{3.7} + \frac{2.51}{Re\sqrt{f}}\right)$$

**Hagen-Poiseuille (Laminar):**

$$\Delta P = \frac{128 \mu L Q}{\pi D^4}$$

**Bernoulli Equation:**

$$P_1 + \frac{1}{2}\rho v_1^2 + \rho g z_1 = P_2 + \frac{1}{2}\rho v_2^2 + \rho g z_2$$

### A.3 Heat Transfer Equations

**Fourier's Law:**

$$q = -k \frac{dT}{dx}$$

**Newton's Law of Cooling:**

$$Q = hA(T_s - T_\infty)$$

**Stefan-Boltzmann Law:**

$$q = \varepsilon \sigma T^4$$

**Log Mean Temperature Difference:**

$$\Delta T_{LMTD} = \frac{\Delta T_1 - \Delta T_2}{\ln(\Delta T_1/\Delta T_2)}$$

**Overall Heat Transfer Coefficient:**

$$\frac{1}{U} = \frac{1}{h_i} + \frac{r_i \ln(r_o/r_i)}{k} + \frac{r_i}{r_o \cdot h_o}$$

### A.4 Mass Transfer Equations

**Fick's First Law:**

$$J = -D \frac{dC}{dx}$$

**Henry's Law:**

$$p_i = H_i \cdot x_i$$

**NTU-HTU Method:**

$$Z = NTU \times HTU$$

$$NTU = \int_{y_{out}}^{y_{in}} \frac{dy}{y - y^*}$$

### A.5 Compression Equations

**Isentropic Compression:**

$$T_2 = T_1 \left(\frac{P_2}{P_1}\right)^{(\gamma-1)/\gamma}$$

$$W_s = \frac{\gamma}{\gamma-1} RT_1 \left[\left(\frac{P_2}{P_1}\right)^{(\gamma-1)/\gamma} - 1\right]$$

**Polytropic Compression:**

$$PV^n = \text{const}$$

$$W_p = \frac{n}{n-1} RT_1 \left[\left(\frac{P_2}{P_1}\right)^{(n-1)/n} - 1\right]$$

**Isothermal Compression:**

$$W_T = RT \ln\left(\frac{P_2}{P_1}\right)$$

### A.6 Optimization Equations

**Adam Optimizer (Kingma & Ba, 2014):**

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

$$\theta_{t+1} = \theta_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\varepsilon}$$

**Newton-Raphson:**

$$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$$

**Gradient Descent:**

$$\theta_{t+1} = \theta_t - \alpha \nabla f(\theta_t)$$

### A.7 Signal Processing Equations

**Discrete Fourier Transform:**

$$X[k] = \sum_{n=0}^{N-1} x[n] e^{-j2\pi kn/N}$$

**Inverse DFT:**

$$x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] e^{j2\pi kn/N}$$

**Butterworth Filter:**

$$|H(j\omega)|^2 = \frac{1}{1+(\omega/\omega_c)^{2n}}$$

**Chebyshev Type I:**

$$|H(j\omega)|^2 = \frac{1}{1+\varepsilon^2 T_n^2(\omega/\omega_c)}$$

**LMS Adaptive Filter:**

$$\mathbf{w}(n+1) = \mathbf{w}(n) + \mu \cdot e(n) \cdot \mathbf{x}(n)$$

### A.8 Kinematics and Dynamics

**Homogeneous Transformation:**

$$T = \begin{bmatrix} R_{3\times3} & \mathbf{d}_{3\times1} \\ \mathbf{0}_{1\times3} & 1 \end{bmatrix}$$

**Forward Kinematics:**

$$T_n^0 = \prod_{i=1}^{n} T_i^{i-1}$$

**Moment of Inertia (Parallel Axis Theorem):**

$$I = I_{cm} + md^2$$

**Rotational Kinetic Energy:**

$$E_{rot} = \frac{1}{2} I \omega^2$$

---

## Appendix B: API Quick Reference

### B.1 Process Calculator API

```python
# Acid Gas Dewpoint
from upstream_drift_tools.process_calculators import AcidGasDewpointCalculator
calc = AcidGasDewpointCalculator()
result = calc.calculate_dewpoint(
    temperature_c=150.0,
    pressure_bar=10.0,
    composition={'h2o': 0.15, 'hf': 0.0001, 'hcl': 0.0002, 'h2s': 0.001}
)

# Baghouse Calculator
from upstream_drift_tools.process_calculators import BaghouseCalculator
calc = BaghouseCalculator()
result = calc.calculate(
    gas_flow_kg_s=5.0,
    inlet_temp_k=473.15,
    pressure_pa=101325.0,
    composition={'H2': 0.20, 'CO': 0.25, 'CO2': 0.15, 'N2': 0.30, 'H2O': 0.10},
    solid_carbon_in_kg_hr=50.0,
    ash_in_kg_hr=20.0,
)

# Flare Calculator
from upstream_drift_tools.process_calculators import FlareCalculator
calc = FlareCalculator()
result = calc.calculate_flare(
    composition={'H2': 20, 'CO': 25, 'CH4': 5, 'N2': 50},
    mass_flow_kg_hr=1000.0,
    temperature_k=573.15,
    pressure_bar=1.5,
)

# Scrubber Calculator
from upstream_drift_tools.process_calculators import ScrubberCalculator
calc = ScrubberCalculator()
result = calc.design_scrubber(
    gas_flow_rate=5.0,
    inlet_temperature=200.0,
    pressure=101325.0,
    acid_gas_concentrations={'HCl': 100, 'SO2': 50, 'H2S': 30},
)

# ODE Solver
from upstream_drift_tools.process_calculators import ODESolver
solver = ODESolver(
    derivatives={"T": "k*(T_env - T)"},
    parameters={"k": 0.3, "T_env": 350.0}
)
sol = solver.solve((0, 20), [300.0])

# WGS Reactor
from upstream_drift_tools.process_calculators import WGSReactorEngine
engine = WGSReactorEngine()
result = engine.calculate_equilibrium_composition(
    inlet_composition={'CO': 25, 'H2': 20, 'CO2': 10, 'H2O': 5},
    temperature=673.15,
    pressure=25.0,
    steam_ratio=2.0,
)
```

### B.2 Signal Toolkit API

```python
import numpy as np
from signal_toolkit import (
    Signal, SignalGenerator, FunctionFitter,
    create_butterworth_filter, apply_filter,
    compute_derivative, compute_integral,
    NoiseGenerator, NoiseType,
    exp_series, sin_series,
)

# Create signals
t = np.linspace(0, 10, 1000)
sig = SignalGenerator.sinusoid(t, amplitude=2.0, frequency=5.0)
chirp = SignalGenerator.chirp(t, f0=1.0, f1=20.0)
combined = sig + chirp

# Fit functions
fitter = FunctionFitter()
result = fitter.fit_sinusoid(sig)
print(f"R²: {result.r_squared:.4f}")

# Filter
spec = create_butterworth_filter('lowpass', cutoff=10, fs=100, order=4)
filtered = apply_filter(sig, spec)

# Calculus
deriv = compute_derivative(sig)
integral = compute_integral(sig)

# Noise
noisy = sig + NoiseGenerator.generate(t, NoiseType.WHITE, amplitude=0.1)

# Series
exp_approx = exp_series(t, terms=10)
```

---

## Appendix C: Configuration Reference

### C.1 Environment Variables

| Variable          | Default    | Description                                |
| ----------------- | ---------- | ------------------------------------------ |
| `HEADLESS`        | `false`    | Set `true` for headless (no-GUI) operation |
| `TOOLS_DATA_DIR`  | `~/.tools` | User data directory                        |
| `TOOLS_LOG_LEVEL` | `WARNING`  | Logging level                              |

### C.2 PyQt6 Requirements

- Python >= 3.10
- PyQt6 >= 6.6.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- matplotlib >= 3.7.0
- sympy >= 1.12

### C.3 Optional Dependencies

| Package   | Used By                         | Purpose               |
| --------- | ------------------------------- | --------------------- |
| CoolProp  | Acid Gas Dewpoint, Syngas Water | IAPWS-IF97 properties |
| thermo    | Acid Gas Dewpoint               | Peng-Robinson EOS     |
| Flask     | Web applications                | Web server            |
| Streamlit | Web dashboards                  | Interactive web apps  |
| Open3D    | C3D Viewer, URDF Builder        | 3D visualization      |
| trimesh   | Humanoid Builder                | Mesh generation       |

---

## Appendix D: Glossary

| Term                 | Definition                                                                                |
| -------------------- | ----------------------------------------------------------------------------------------- |
| **ACFM**             | Actual Cubic Feet per Minute                                                              |
| **ACR**              | Air-to-Cloth Ratio                                                                        |
| **Antoine Equation** | Empirical relation between vapor pressure and temperature                                 |
| **API 521**          | American Petroleum Institute standard for pressure-relieving systems                      |
| **ASME**             | American Society of Mechanical Engineers                                                  |
| **C3D**              | Coordinate 3D — motion capture file format                                                |
| **EBITDA**           | Earnings Before Interest, Taxes, Depreciation, and Amortization                           |
| **EOS**              | Equation of State                                                                         |
| **GHSV**             | Gas Hourly Space Velocity                                                                 |
| **HTU**              | Height of Transfer Unit                                                                   |
| **IAPWS-IF97**       | International Association for Properties of Water and Steam — Industrial Formulation 1997 |
| **LMS**              | Least Mean Squares (adaptive filter algorithm)                                            |
| **LMTD**             | Log Mean Temperature Difference                                                           |
| **NTU**              | Number of Transfer Units                                                                  |
| **ODE**              | Ordinary Differential Equation                                                            |
| **PSA**              | Pressure Swing Adsorption                                                                 |
| **RLS**              | Recursive Least Squares (adaptive filter algorithm)                                       |
| **ROA**              | Return on Assets                                                                          |
| **ROE**              | Return on Equity                                                                          |
| **RRT**              | Rapidly-exploring Random Tree                                                             |
| **SCFM**             | Standard Cubic Feet per Minute                                                            |
| **STP**              | Standard Temperature and Pressure (0°C, 1 atm)                                            |
| **TCI**              | Total Capital Investment                                                                  |
| **TRC**              | Thermal Reactor/Converter                                                                 |
| **URDF**             | Unified Robot Description Format                                                          |
| **WCAG**             | Web Content Accessibility Guidelines                                                      |
| **WGS**              | Water-Gas Shift                                                                           |

---

## Appendix E: File Count Summary

| Directory                             | Python Files | Test Files | Total Lines (est.) |
| ------------------------------------- | ------------ | ---------- | ------------------ |
| `shared/python/signal_toolkit/`       | 10           | 6+         | ~4,000             |
| `shared/python/upstream_drift_tools/` | 28+          | 5+         | ~8,000             |
| `shared/python/gui_launcher/`         | 3            | —          | ~500               |
| `shared/python/theme/`                | 4            | —          | ~600               |
| Process calculator GUIs (20 tools)    | 60+          | 20+        | ~15,000            |
| Scientific modeling                   | 10+          | 2+         | ~2,000             |
| Web applications                      | 15+          | 5+         | ~3,000             |
| Media processing                      | 20+          | 5+         | ~4,000             |
| Development tools                     | 10+          | —          | ~1,500             |
| **Total**                             | **~679**     | **~50+**   | **~40,000**        |

---

_[← Implementation Gaps](./12_implementation_gaps.md) | [Back to Manual](./TOOLS_USER_MANUAL.md)_

---

**Document Version History:**

| Version | Date          | Description                  |
| ------- | ------------- | ---------------------------- |
| 1.0.0   | February 2026 | Initial comprehensive manual |

**License:** This documentation is part of the Tools repository and follows the same license terms.

**Contributing:** To update this manual, edit files in `docs/user_manual/` and submit a pull request. See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.
