# Function Generator

A web-based signal generation and visualization tool built with React, TypeScript, and Tauri for desktop deployment.

## Purpose

The Function Generator creates mathematical waveforms for testing, simulation, and educational purposes. It supports multiple waveform types with real-time visualization in both time and frequency domains, along with signal layer stacking for complex waveform synthesis.

## Key Features

- **12 Waveform Types**: Sinusoid, cosine, square, triangle, sawtooth, pulse, step, exponential, linear, polynomial, chirp, and constant
- **Signal Stacking**: Combine multiple waveforms with add/subtract operations
- **Dual Domain Display**: Time domain and frequency spectrum (FFT) views
- **Real-Time Updates**: Instant visualization as parameters change
- **Preset Configurations**: Quick access to common signal patterns
- **Signal Statistics**: Min, max, mean, and RMS calculations
- **Hanning Windowing**: Reduced spectral leakage in FFT analysis

## Installation / Prerequisites

### Web Development

```bash
cd web
npm install
npm run dev
```

### Desktop Application (Tauri)

```bash
cd web
npm install
npm run tauri dev
```

### Building for Production

```bash
npm run tauri build
```

## Usage Instructions

### Signal Layers Panel

1. Click "+ Add Layer" to create additional waveforms
2. Click a layer to select it for editing
3. Toggle ON/OFF to enable/disable layers
4. Set operation (Add/Subtract) for layer combination
5. Click X to remove a layer (minimum one layer required)

### Waveform Type Selection

Select from 12 available waveform types:
- **Sinusoid/Cosine**: Standard trigonometric functions
- **Square**: Rectangular wave with adjustable duty cycle
- **Triangle/Sawtooth**: Linear ramp waveforms
- **Pulse**: Single rectangular pulse
- **Step**: Heaviside step function
- **Exponential**: Decaying exponential
- **Linear/Polynomial**: Algebraic functions
- **Chirp**: Frequency sweep signal
- **Constant**: DC offset

### Time Parameters

- **Duration**: Signal length in seconds
- **Sample Rate**: Samples per second (Hz)

### Visualization

- **Time Domain Tab**: Amplitude vs. time plot
- **Frequency Domain Tab**: Magnitude spectrum via FFT
- **Layer Display Toggle**: Show/hide individual layer traces

### Quick Presets

Single signals:
- 1 Hz Sine, 10 Hz Sine, 50 Hz Square
- Chirp 1-50 Hz, Decay, Parabola

Stacked signals:
- Sine + Harmonic, AM Modulation
- Square - Sine, 3-Tone Chord

## Input Parameters

### Time Parameters

| Parameter | Unit | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| Duration | s | 0.01 - 100 | 1 | Signal length |
| Sample Rate | Hz | 10 - 100,000 | 1000 | Sampling frequency |

### Sinusoid/Cosine Parameters

| Parameter | Unit | Range | Description |
|-----------|------|-------|-------------|
| Amplitude | - | any | Peak amplitude |
| Frequency | Hz | 0.01+ | Oscillation frequency |
| Phase | rad | any | Phase offset |
| DC Offset | - | any | Vertical shift |

### Square Wave Parameters

| Parameter | Unit | Range | Description |
|-----------|------|-------|-------------|
| Amplitude | - | any | High/low level (symmetric) |
| Frequency | Hz | 0.01+ | Repetition rate |
| Duty Cycle | - | 0.01 - 0.99 | High fraction per period |
| DC Offset | - | any | Baseline shift |

### Pulse Parameters

| Parameter | Unit | Range | Description |
|-----------|------|-------|-------------|
| Amplitude | - | any | Pulse height |
| Start Time | s | 0+ | Pulse onset |
| Duration | s | 0.001+ | Pulse width |
| Baseline | - | any | Off-pulse value |

### Chirp Parameters

| Parameter | Unit | Range | Description |
|-----------|------|-------|-------------|
| Amplitude | - | any | Peak amplitude |
| Start Freq | Hz | 0.01+ | Initial frequency |
| End Freq | Hz | 0.01+ | Final frequency |
| Sweep Method | - | linear/exponential | Frequency progression |

### Polynomial Parameters

| Parameter | Description |
|-----------|-------------|
| Coefficients | Comma-separated: c0, c1, c2, ... |

Formula: y = c0 + c1*t + c2*t^2 + c3*t^3 + ...

## Output Format

### Signal Data

The generated signal consists of:
- **Time array**: [0, dt, 2dt, ..., T] where dt = 1/sampleRate
- **Values array**: Computed amplitude at each time point

### Statistics

| Statistic | Description |
|-----------|-------------|
| Samples | Total number of data points |
| Min | Minimum amplitude value |
| Max | Maximum amplitude value |
| Mean | Average (DC component) |
| RMS | Root mean square amplitude |

### Frequency Spectrum

FFT output with Hanning window correction:
- Frequency bins: 0 to Nyquist (sampleRate/2)
- Magnitude: Normalized amplitude per frequency

## Mathematical Models

### Sinusoidal Waveforms

```
y(t) = A * sin(2*pi*f*t + phi) + offset
y(t) = A * cos(2*pi*f*t + phi) + offset
```

### Square Wave

```
y(t) = A    if (t mod T) / T < dutyCycle
     = -A   otherwise
where T = 1/f
```

### Triangle Wave

```
y(t) = 4*A*(t/T) - A           for 0 <= phase < 0.5
     = -4*A*(t/T) + 3*A        for 0.5 <= phase < 1.0
where phase = (t mod T) / T
```

### Sawtooth Wave

```
y(t) = 2*A * ((t mod T) / T) - A
```

### Exponential Decay

```
y(t) = A * exp(-lambda * t) + offset
```

### Linear Function

```
y(t) = slope * t + intercept
```

### Polynomial

```
y(t) = sum(c[i] * t^i) for i = 0 to n
```

### Chirp (Frequency Sweep)

Linear sweep:
```
f(t) = f0 + (f1 - f0) * t / T
y(t) = A * sin(2*pi*f(t)*t)
```

Exponential sweep:
```
f(t) = f0 * (f1/f0)^(t/T)
y(t) = A * sin(2*pi*f(t)*t)
```

### FFT with Hanning Window

Window function:
```
w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))
```

Corrected magnitude:
```
X[k] = (2/N) * |sum(x[n]*w[n]*exp(-2*pi*i*k*n/N))| * (N/sum(w))
```

## Example Usage

### Example 1: Test Tone Generation

Generate a 440 Hz A4 note:
1. Select Sinusoid
2. Set Amplitude: 1.0
3. Set Frequency: 440 Hz
4. Duration: 0.5 s
5. Sample Rate: 44100 Hz

### Example 2: PWM-like Square Wave

Generate pulse-width modulated signal:
1. Select Square Wave
2. Frequency: 1000 Hz
3. Duty Cycle: 0.25 (25%)
4. View time domain for rectangular pulses

### Example 3: Frequency Sweep for System Testing

Create a chirp for frequency response analysis:
1. Select Chirp (Sweep)
2. Start Freq: 20 Hz
3. End Freq: 20000 Hz
4. Duration: 5 s
5. Sweep Method: Exponential (logarithmic)

### Example 4: AM Modulated Signal

Using signal stacking:
1. Layer 1: Sinusoid at 1000 Hz (carrier)
2. Layer 2: Sinusoid at 10 Hz, offset 0.5 (modulator)
3. Both layers set to Add

### Example 5: Harmonic Analysis

View square wave harmonics:
1. Generate 10 Hz square wave
2. Switch to Frequency Domain tab
3. Observe peaks at 10, 30, 50 Hz (odd harmonics)
4. Magnitude decreases as 1/n

## Troubleshooting

### Chart appears empty or flat

- Check that amplitude is non-zero
- Verify frequency is appropriate for duration (at least one cycle visible)
- Ensure layer is enabled (ON)

### FFT shows no peaks

- Increase duration to improve frequency resolution
- Check that signal frequency < sampleRate/2 (Nyquist)
- Verify signal has AC component (not just DC offset)

### Performance is slow

- Reduce sample rate for interactive editing
- Decrease duration for faster computation
- Complex polynomial coefficients increase computation time

### Layers not combining correctly

- Verify all desired layers are enabled (ON)
- Check add/subtract operations on each layer
- First layer is always additive (sets baseline)

### Unexpected frequency spectrum

- Hanning window reduces spectral leakage but broadens peaks
- Very short signals have poor frequency resolution
- Discontinuities at signal edges can cause artifacts

## Related Tools

- **Inertia Calculator**: Use generated signals for dynamic simulation input
- **C3D Viewer**: Compare synthetic signals with motion capture data
- **URDF Builder GUI**: Apply signals as joint trajectories
- **Data Processor**: Filter and analyze generated waveforms

## References

- Smith, S.W. (1997). The Scientist and Engineer's Guide to Digital Signal Processing
- Oppenheim, A.V. & Schafer, R.W. (2010). Discrete-Time Signal Processing
- Harris, F.J. (1978). On the Use of Windows for Harmonic Analysis with the DFT
- Recharts Documentation: https://recharts.org/
- Tauri Framework: https://tauri.app/
