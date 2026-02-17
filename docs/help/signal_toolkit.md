# Signal Processing Tools

The Signal Processing category provides tools for generating, analyzing, and processing signals and waveforms. This includes standalone GUI applications and a shared library for integration into other tools.

## Function Generator

Generate and visualize mathematical functions and waveforms.

### Waveform Types

#### Standard Waveforms

| Waveform | Description            | Parameters                       |
| -------- | ---------------------- | -------------------------------- |
| Sine     | Sinusoidal oscillation | Frequency, amplitude, phase      |
| Square   | Digital-like on/off    | Frequency, amplitude, duty cycle |
| Triangle | Linear ramps           | Frequency, amplitude             |
| Sawtooth | Rising or falling ramp | Frequency, amplitude, direction  |

#### Mathematical Functions

| Function    | Description               |
| ----------- | ------------------------- |
| Polynomial  | User-defined coefficients |
| Exponential | Growth or decay           |
| Logarithmic | Log base e or 10          |
| Gaussian    | Normal distribution curve |

#### Custom Expressions

Enter custom mathematical expressions using:

- Variables: `t` (time), `x` (position)
- Functions: `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `abs`
- Constants: `pi`, `e`
- Operators: `+`, `-`, `*`, `/`, `**` (power)

Example: `sin(2*pi*10*t) + 0.5*sin(2*pi*30*t)`

### Parameters

| Parameter   | Description             | Typical Range  |
| ----------- | ----------------------- | -------------- |
| Frequency   | Oscillation rate (Hz)   | 0.001 - 100000 |
| Amplitude   | Peak value              | 0.001 - 1000   |
| Phase       | Phase offset (degrees)  | 0 - 360        |
| DC Offset   | Vertical shift          | -1000 to 1000  |
| Duty Cycle  | Square wave on-time     | 0 - 100%       |
| Duration    | Signal length (seconds) | 0.001 - 1000   |
| Sample Rate | Points per second       | 100 - 1000000  |

### Output Formats

- **Preview**: Interactive matplotlib plot
- **CSV**: Time and amplitude columns
- **NumPy**: .npy binary format
- **WAV**: Audio file (normalized to 16-bit)

---

## Polynomial Generator

Generate, analyze, and fit polynomial functions.

### Polynomial Operations

#### Coefficient Input

Enter polynomial as coefficients (highest power first):

- `[1, 0, -4]` represents x^2 - 4
- `[1, -6, 11, -6]` represents x^3 - 6x^2 + 11x - 6

#### Root Finding

Find real and complex roots of polynomials:

- Real roots (crossing points)
- Complex conjugate pairs
- Multiplicity detection

#### Curve Fitting

Fit a polynomial to data points:

1. Import data (x, y pairs)
2. Select polynomial degree
3. View fit quality metrics (R^2, RMSE)

### Calculus Operations

| Operation  | Description                   |
| ---------- | ----------------------------- |
| Derivative | Compute polynomial derivative |
| Integral   | Compute antiderivative        |
| Evaluate   | Calculate y for given x       |
| Roots      | Find zeros of polynomial      |

### Taylor Series

Expand functions as Taylor series around a point:

- Enter function expression
- Specify expansion point
- Choose number of terms
- Compare original vs approximation

---

## Signal Toolkit Library

Shared library providing signal processing primitives for other tools.

### Module: filters.py

Digital filter implementations.

#### Available Filters

| Filter Type | Description                   | Parameters           |
| ----------- | ----------------------------- | -------------------- |
| Lowpass     | Pass frequencies below cutoff | fc, order            |
| Highpass    | Pass frequencies above cutoff | fc, order            |
| Bandpass    | Pass frequencies in range     | f_low, f_high, order |
| Bandstop    | Block frequencies in range    | f_low, f_high, order |
| Notch       | Remove specific frequency     | f_notch, Q           |

#### Filter Designs

| Design            | Characteristics               |
| ----------------- | ----------------------------- |
| Butterworth       | Maximally flat passband       |
| Chebyshev Type I  | Sharp cutoff, passband ripple |
| Chebyshev Type II | Sharp cutoff, stopband ripple |
| Bessel            | Linear phase (no distortion)  |
| FIR               | Custom frequency response     |

#### Usage Example

```python
from signal_toolkit.filters import apply_lowpass_filter

# Apply 100 Hz lowpass to signal
filtered = apply_lowpass_filter(
    signal,
    cutoff_freq=100,
    sample_rate=1000,
    order=4,
    filter_type='butterworth'
)
```

### Module: calculus.py

Numerical differentiation and integration.

#### Differentiation

```python
from signal_toolkit.calculus import differentiate

# Compute derivative
derivative = differentiate(signal, dt=0.001, method='central')
```

Methods:

- `forward`: Forward difference (noisy)
- `backward`: Backward difference (noisy)
- `central`: Central difference (recommended)
- `savgol`: Savitzky-Golay (smoothed)

#### Integration

```python
from signal_toolkit.calculus import integrate

# Compute integral
integral = integrate(signal, dt=0.001, method='trapezoid')
```

Methods:

- `rectangle`: Simple summation
- `trapezoid`: Trapezoidal rule
- `simpson`: Simpson's rule (recommended)
- `cumulative`: Running integral

### Module: noise.py

Noise generation and analysis.

#### Noise Types

| Type  | Spectrum | Use Case        |
| ----- | -------- | --------------- |
| White | Flat     | General testing |
| Pink  | 1/f      | Audio, natural  |
| Brown | 1/f^2    | Low-frequency   |
| Blue  | f        | Testing         |

#### Usage Example

```python
from signal_toolkit.noise import generate_noise, add_noise

# Generate noise signal
noise = generate_noise('white', length=1000, amplitude=0.1)

# Add noise to signal
noisy_signal = add_noise(signal, snr_db=20)
```

### Module: fitting.py

Curve fitting algorithms.

#### Fit Types

| Type        | Description                          |
| ----------- | ------------------------------------ |
| Linear      | y = mx + b                           |
| Polynomial  | y = sum(a_i \* x^i)                  |
| Exponential | y = a * exp(b*x)                     |
| Power       | y = a \* x^b                         |
| Gaussian    | y = a * exp(-(x-mu)^2 / (2*sigma^2)) |
| Sinusoidal  | y = a * sin(2*pi*f*x + phi)          |

#### Usage Example

```python
from signal_toolkit.fitting import fit_curve

# Fit exponential to data
params, r_squared = fit_curve(
    x_data, y_data,
    fit_type='exponential'
)
```

### Module: limits.py

Limit detection and validation.

#### Functions

```python
from signal_toolkit.limits import (
    detect_peaks,
    detect_limits,
    check_bounds
)

# Find peaks in signal
peaks = detect_peaks(signal, min_height=0.5, min_distance=10)

# Detect upper/lower limits
upper, lower = detect_limits(signal, window=100)

# Check if signal within bounds
in_bounds = check_bounds(signal, lower=-10, upper=10)
```

### Module: io.py

Signal I/O utilities.

#### Supported Formats

| Format | Read | Write | Notes               |
| ------ | ---- | ----- | ------------------- |
| CSV    | Yes  | Yes   | Time, value columns |
| NumPy  | Yes  | Yes   | .npy binary         |
| WAV    | Yes  | Yes   | Audio format        |
| MATLAB | Yes  | Yes   | .mat files          |
| HDF5   | Yes  | Yes   | Large datasets      |

---

## Tips for Signal Processing

### Sampling Considerations

**Nyquist Theorem**: Sample rate must be at least 2x the highest frequency in the signal.

```
f_sample >= 2 * f_max
```

For practical applications, use 5-10x oversampling.

### Filter Design Tips

1. **Start with low order**: Higher order = sharper cutoff but more ringing
2. **Check phase response**: Use Bessel for time-domain accuracy
3. **Watch for instability**: Very high orders can be numerically unstable
4. **Consider zero-phase**: Apply filter forward and backward for no phase shift

### Differentiation Tips

1. **Noise amplification**: Differentiation amplifies high-frequency noise
2. **Pre-filter**: Apply lowpass filter before differentiating
3. **Use Savitzky-Golay**: Combines smoothing and differentiation

### Integration Tips

1. **DC offset**: Integration accumulates DC offset errors
2. **High-pass first**: Remove DC before integrating
3. **Drift correction**: May need to detrend result

---

## Quick Reference

### Common Operations

| Task               | Function                                        |
| ------------------ | ----------------------------------------------- |
| Generate sine wave | `Function Generator > Sine`                     |
| Filter signal      | `signal_toolkit.filters.apply_lowpass_filter()` |
| Compute derivative | `signal_toolkit.calculus.differentiate()`       |
| Fit polynomial     | `Polynomial Generator > Curve Fit`              |
| Add noise          | `signal_toolkit.noise.add_noise()`              |
| Find peaks         | `signal_toolkit.limits.detect_peaks()`          |

### Import Statement

```python
# Import all signal toolkit functions
from signal_toolkit import (
    filters,
    calculus,
    noise,
    fitting,
    limits,
    io
)
```

---

For more details, see the [User Manual](../USER_MANUAL.md) or the API documentation in the source code.
