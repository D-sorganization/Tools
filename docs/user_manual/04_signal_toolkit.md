# Chapter 4 — Signal Processing Toolkit

**Parent Document:** [Tools User Manual](./TOOLS_USER_MANUAL.md)
**Source:** `src/shared/python/signal_toolkit/`
**Version:** 2.1.0
**Status:** ✅ Fully Implemented

---

## Overview

The Signal Processing Toolkit is a production-ready library for generating, fitting, filtering, and analyzing signals. It is designed for use in control systems, simulation, robotics, and data analysis.

**Modules:**

| Module | File | Description |
|--------|------|-------------|
| Core | `core.py` | Signal class and SignalGenerator |
| Fitting | `fitting.py` | Function fitting (sinusoid, exponential, polynomial) |
| Filters | `filters.py` | Digital filters (Butterworth, Chebyshev, Bessel, adaptive) |
| Calculus | `calculus.py` | Differentiation, integration, curvature |
| Series | `series.py` | Taylor/Maclaurin series expansions |
| Noise | `noise.py` | Noise generation (white, pink, brown, etc.) |
| Limits | `limits.py` | Saturation, rate limiting, deadband, hysteresis |
| I/O | `io.py` | CSV, JSON, NPZ, MAT file support |
| Widget | `widget.py` | PyQt6 interactive analysis widget |
| Polynomial Generator | `polynomial_generator.py` | PyQt6 polynomial visualization |

---

## 4.1 Signal Core Classes

### `Signal` Dataclass

The fundamental data structure for time-domain signals.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `time` | `np.ndarray` | Time array (1D) |
| `values` | `np.ndarray` | Signal values (1D or 2D) |
| `name` | `str` | Signal name |
| `units` | `str` | Units string (e.g., 'N·m', 'rad/s') |
| `metadata` | `dict` | Additional metadata |

**Computed Properties:**

| Property | Formula | Description |
|----------|---------|-------------|
| `fs` | $f_s = 1 / \overline{\Delta t}$ | Sampling frequency (Hz) |
| `dt` | $\Delta t = \overline{t_{i+1} - t_i}$ | Time step (s) |
| `duration` | $D = t_{N-1} - t_0$ | Total duration (s) |
| `n_samples` | $N = \|t\|$ | Number of samples |

**Operations:**

- `copy()`: Deep copy
- `slice(t_start, t_end)`: Extract time window
- `resample(new_fs)`: Linear interpolation resampling
- `+`, `*`, `-` (unary): Arithmetic operations between signals or with scalars

---

## 4.2 Signal Generation

The `SignalGenerator` class provides static factory methods for 13 signal types:

### 4.2.1 Sinusoid

$$y(t) = A \sin(2\pi f t + \phi) + C$$

**Parameters:** amplitude $A$, frequency $f$ (Hz), phase $\phi$ (rad), offset $C$

### 4.2.2 Cosine

$$y(t) = A \cos(2\pi f t + \phi) + C$$

### 4.2.3 Exponential

$$y(t) = A \cdot e^{-\lambda t} + C$$

**Parameters:** amplitude $A$, decay rate $\lambda$, offset $C$

### 4.2.4 Linear (Ramp)

$$y(t) = m \cdot t + b$$

### 4.2.5 Polynomial

$$y(t) = c_0 + c_1 t + c_2 t^2 + \cdots + c_n t^n$$

### 4.2.6 Step

$$y(t) = \begin{cases} y_0 & t < t_{step} \\ y_1 & t \geq t_{step} \end{cases}$$

### 4.2.7 Pulse

$$y(t) = \begin{cases} A & t_0 \leq t < t_0 + \Delta t \\ B & \text{otherwise} \end{cases}$$

### 4.2.8 Chirp (Frequency Sweep)

**Linear chirp:**

$$y(t) = A \sin\left(2\pi \left(f_0 t + \frac{k t^2}{2}\right)\right)$$

where $k = (f_1 - f_0) / T$.

**Exponential chirp:**

$$y(t) = A \sin\left(\frac{2\pi f_0 (r^t - 1)}{\ln r}\right)$$

where $r = (f_1/f_0)^{1/T}$.

### 4.2.9 Sawtooth

$$y(t) = A \left(2 \cdot \frac{t \bmod T}{T} - 1\right) + C$$

### 4.2.10 Triangle

$$y(t) = A \left(4 \left|\frac{t \bmod T}{T} - 0.5\right| - 1\right) + C$$

### 4.2.11 Square Wave

$$y(t) = \begin{cases} +A + C & (t \bmod T)/T < d \\ -A + C & \text{otherwise} \end{cases}$$

where $d$ is the duty cycle.

### 4.2.12 Custom Function

$$y(t) = f(t)$$

where $f$ is a user-supplied callable.

### 4.2.13 Superposition

$$y(t) = \sum_{k=1}^{N} y_k(t)$$

---

## 4.3 Function Fitting

**Source:** `fitting.py`

Provides curve fitting with goodness-of-fit metrics for various function types.

### `FitResult` Dataclass

| Field | Description |
|-------|-------------|
| `parameters` | Fitted parameter values |
| `r_squared` | Coefficient of determination ($R^2$) |
| `rmse` | Root mean square error |
| `fitted_signal` | Signal with fitted values |

### 4.3.1 Sinusoid Fitting

Fits $y = A \sin(2\pi f t + \phi) + C$ using:
1. FFT-based frequency estimation for initial guess
2. Nonlinear least squares (`scipy.optimize.curve_fit`)

### 4.3.2 Exponential Fitting

Fits $y = A \cdot e^{-\lambda t} + C$ with log-transform initial estimate.

### 4.3.3 Linear Fitting

Fits $y = mt + b$ using `np.polyfit(t, y, 1)`.

### 4.3.4 Polynomial Fitting

Fits $y = \sum_{k=0}^{n} c_k t^k$ using `np.polyfit` of degree $n$.

### 4.3.5 Cosine Fitting

Fits $y = A \cos(2\pi f t + \phi) + C$.

### 4.3.6 Custom Function Fitting

Fits any user-defined function $y = f(t; \theta)$ via `scipy.optimize.curve_fit`.

**Goodness of Fit:**

$$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

$$RMSE = \sqrt{\frac{1}{N} \sum_i (y_i - \hat{y}_i)^2}$$

---

## 4.4 Digital Filters

**Source:** `filters.py`

### 4.4.1 Filter Types

| Type | Enum Value | Description |
|------|------------|-------------|
| Lowpass | `FilterType.LOWPASS` | Pass frequencies below cutoff |
| Highpass | `FilterType.HIGHPASS` | Pass frequencies above cutoff |
| Bandpass | `FilterType.BANDPASS` | Pass frequencies within band |
| Bandstop | `FilterType.BANDSTOP` | Reject frequencies within band |

### 4.4.2 Filter Designs

| Design | Enum Value | Transfer Function Characteristics |
|--------|------------|-----------------------------------|
| Butterworth | `FilterDesign.BUTTERWORTH` | Maximally flat passband |
| Chebyshev Type I | `FilterDesign.CHEBYSHEV1` | Equiripple passband |
| Chebyshev Type II | `FilterDesign.CHEBYSHEV2` | Equiripple stopband |
| Bessel | `FilterDesign.BESSEL` | Linear phase response |

### 4.4.3 Butterworth Filter

**Transfer Function:**

$$|H(j\omega)|^2 = \frac{1}{1 + \left(\omega / \omega_c\right)^{2n}}$$

where $n$ is the filter order and $\omega_c$ is the cutoff frequency.

**Usage:**

```python
from signal_toolkit import create_butterworth_filter, apply_filter

spec = create_butterworth_filter('lowpass', cutoff=5, fs=100, order=4)
filtered = apply_filter(signal, spec)
```

### 4.4.4 Chebyshev Filter

**Type I Transfer Function:**

$$|H(j\omega)|^2 = \frac{1}{1 + \varepsilon^2 T_n^2(\omega/\omega_c)}$$

where $T_n$ is the Chebyshev polynomial of order $n$ and $\varepsilon$ controls ripple.

### 4.4.5 Additional Smoothing Methods

| Method | Function | Description |
|--------|----------|-------------|
| Moving Average | `apply_moving_average(signal, window)` | Simple windowed average |
| Savitzky-Golay | `apply_savgol(signal, window, polyorder)` | Polynomial smoothing |
| Median Filter | `apply_median_filter(signal, kernel)` | Nonlinear noise removal |
| Exponential Smoothing | `apply_exponential_smoothing(signal, alpha)` | $y_k = \alpha x_k + (1-\alpha) y_{k-1}$ |
| Gaussian Smoothing | `apply_gaussian_smoothing(signal, sigma)` | Gaussian kernel convolution |
| Bilateral Filter | `apply_bilateral_filter(signal, sigma_s, sigma_r)` | Edge-preserving smoothing |

### 4.4.6 Adaptive Filters

**Least Mean Squares (LMS):**

$$w(n+1) = w(n) + \mu \cdot e(n) \cdot x(n)$$

where $\mu$ is the step size and $e(n) = d(n) - w^T(n)x(n)$.

**Recursive Least Squares (RLS):**

$$k(n) = \frac{P(n-1)x(n)}{\lambda + x^T(n)P(n-1)x(n)}$$

$$w(n) = w(n-1) + k(n) \cdot e(n)$$

$$P(n) = \frac{1}{\lambda}\left[P(n-1) - k(n)x^T(n)P(n-1)\right]$$

where $\lambda$ is the forgetting factor.

---

## 4.5 Calculus Operations

**Source:** `calculus.py`

### 4.5.1 Differentiation Methods

| Method | Enum | Formula |
|--------|------|---------|
| Forward Difference | `FORWARD` | $f'(x) \approx \frac{f(x+h) - f(x)}{h}$ |
| Backward Difference | `BACKWARD` | $f'(x) \approx \frac{f(x) - f(x-h)}{h}$ |
| Central Difference | `CENTRAL` | $f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}$ |
| Gradient | `GRADIENT` | `np.gradient(y, x)` |
| Savitzky-Golay | `SAVGOL` | Polynomial derivative via SG filter |

### 4.5.2 Integration Methods

| Method | Enum | Formula |
|--------|------|---------|
| Trapezoidal | `TRAPEZOID` | $\int f \, dx \approx \sum \frac{(f_i + f_{i+1}) \cdot \Delta x_i}{2}$ |
| Simpson's Rule | `SIMPSON` | `scipy.integrate.simpson` |
| Cumulative | `CUMULATIVE` | `scipy.integrate.cumulative_trapezoid` |

### 4.5.3 Tangent Line

At point $(x_0, y_0)$:

$$y_{tangent}(x) = f'(x_0) \cdot (x - x_0) + f(x_0)$$

### 4.5.4 Curvature

$$\kappa = \frac{|y''|}{(1 + y'^2)^{3/2}}$$

### 4.5.5 Arc Length

$$L = \int_a^b \sqrt{1 + \left(\frac{dy}{dx}\right)^2} \, dx$$

### 4.5.6 Extrema Detection

Finds local maxima and minima using `scipy.signal.argrelextrema`.

### 4.5.7 Inflection Points

Finds points where curvature changes sign (zero crossings of $y''$).

---

## 4.6 Series Expansions

**Source:** `series.py`

### `SeriesResult` Dataclass

| Field | Description |
|-------|-------------|
| `values` | Evaluated series values |
| `terms` | Number of terms used |
| `remainder` | Estimated remainder/error |
| `converged` | Whether series converged |

### 4.6.1 Exponential Series

$$e^x = \sum_{n=0}^{N} \frac{x^n}{n!}$$

### 4.6.2 Sine Series

$$\sin(x) = \sum_{n=0}^{N} \frac{(-1)^n x^{2n+1}}{(2n+1)!}$$

### 4.6.3 Cosine Series

$$\cos(x) = \sum_{n=0}^{N} \frac{(-1)^n x^{2n}}{(2n)!}$$

### 4.6.4 Natural Logarithm Series

$$\ln(1+x) = \sum_{n=1}^{N} \frac{(-1)^{n+1} x^n}{n} \qquad |x| \leq 1$$

### 4.6.5 Geometric Series

$$\frac{1}{1-x} = \sum_{n=0}^{N} x^n \qquad |x| < 1$$

### 4.6.6 Arctangent Series

$$\arctan(x) = \sum_{n=0}^{N} \frac{(-1)^n x^{2n+1}}{2n+1} \qquad |x| \leq 1$$

### 4.6.7 Hyperbolic Sine/Cosine

$$\sinh(x) = \sum_{n=0}^{N} \frac{x^{2n+1}}{(2n+1)!}$$

$$\cosh(x) = \sum_{n=0}^{N} \frac{x^{2n}}{(2n)!}$$

---

## 4.7 Noise Generation

**Source:** `noise.py`

### Noise Types

| Type | Enum | Spectral Density | Description |
|------|------|-----------------|-------------|
| White | `NoiseType.WHITE` | Flat | Equal power across frequencies |
| Pink | `NoiseType.PINK` | $\propto 1/f$ | Equal power per octave |
| Brown | `NoiseType.BROWN` | $\propto 1/f^2$ | Random walk / Brownian motion |
| Blue | `NoiseType.BLUE` | $\propto f$ | Differentiated white noise |
| Violet | `NoiseType.VIOLET` | $\propto f^2$ | Differentiated pink noise |
| Impulse | `NoiseType.IMPULSE` | — | Random sparse impulses |

### Disturbance Simulator

Generates composite disturbance profiles combining multiple noise types, steps, and sinusoidal perturbations for control system testing.

---

## 4.8 Signal Limits

**Source:** `limits.py`

### 4.8.1 Saturation

$$y = \begin{cases} y_{max} & x > y_{max} \\ x & y_{min} \leq x \leq y_{max} \\ y_{min} & x < y_{min} \end{cases}$$

**Modes:** `HARD`, `SOFT` (tanh-based), `SYMMETRIC`

### 4.8.2 Rate Limiter

$$y_k = y_{k-1} + \text{clamp}\left(x_k - y_{k-1},\ -R \cdot \Delta t,\ R \cdot \Delta t\right)$$

where $R$ is the maximum rate of change.

### 4.8.3 Deadband

$$y = \begin{cases} 0 & |x| < d \\ x - \text{sign}(x) \cdot d & |x| \geq d \end{cases}$$

### 4.8.4 Hysteresis

Models mechanical hysteresis with upper and lower thresholds creating a memory effect.

### 4.8.5 Backlash

Models mechanical backlash (play) in gear trains with configurable dead zone width.

---

## 4.9 I/O Operations

**Source:** `io.py`

### Supported Formats

| Format | Import | Export | Class |
|--------|--------|--------|-------|
| CSV | ✅ | ✅ | `SignalImporter` / `SignalExporter` |
| JSON | ✅ | ✅ | `SignalImporter` / `SignalExporter` |
| NumPy NPZ | ✅ | ✅ | `SignalLoader` |
| MATLAB .mat | ✅ | ✅ | `SignalLoader` |
| NumPy arrays | ✅ | ✅ | Direct conversion |

### BatchProcessor

Enables batch processing of multiple signal files with configurable processing pipelines.

---

*[← Process Calculators](./03_process_calculators.md) | [Back to Manual](./TOOLS_USER_MANUAL.md) | [Next: Scientific Modeling →](./05_scientific_modeling.md)*
