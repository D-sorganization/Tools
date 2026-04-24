//! Signal processing kernel — waveform generation and basic signal operations.
//!
//! This module provides the canonical implementation of signal generation
//! functions used across PyQt6, Web (WASM), and Tauri surfaces.
//!
//! # Design by Contract
//! - All frequency values must be non-negative
//! - All time arrays must be non-empty
//! - Amplitude can be any finite f64
//!
//! # Functions
//! - `sinusoid` — y = A·sin(2πft + φ) + offset
//! - `cosine` — y = A·cos(2πft + φ) + offset
//! - `exponential` — y = A·exp(-λt) + offset
//! - `linear` — y = slope·t + intercept
//! - `step` — Heaviside step
//! - `square` — Square wave
//! - `triangle` — Triangle wave
//! - `chirp` — Linear frequency sweep
//! - `polynomial` — y = Σ cₙ·tⁿ

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Core generation functions
// ---------------------------------------------------------------------------

/// Generate a sinusoidal waveform: y = amplitude * sin(2π * freq * t + phase) + offset
///
/// # Preconditions
/// - `times` must not be empty
/// - `frequency` must be non-negative
pub fn sinusoid(
    times: &[f64],
    amplitude: f64,
    frequency: f64,
    phase: f64,
    offset: f64,
) -> Vec<f64> {
    debug_assert!(!times.is_empty(), "times must not be empty");
    debug_assert!(frequency >= 0.0, "frequency must be non-negative");
    times
        .iter()
        .map(|&t| amplitude * (2.0 * PI * frequency * t + phase).sin() + offset)
        .collect()
}

/// Generate a cosine waveform: y = amplitude * cos(2π * freq * t + phase) + offset
pub fn cosine(times: &[f64], amplitude: f64, frequency: f64, phase: f64, offset: f64) -> Vec<f64> {
    debug_assert!(!times.is_empty(), "times must not be empty");
    debug_assert!(frequency >= 0.0, "frequency must be non-negative");
    times
        .iter()
        .map(|&t| amplitude * (2.0 * PI * frequency * t + phase).cos() + offset)
        .collect()
}

/// Generate an exponential decay/growth: y = amplitude * exp(-decay_rate * (t - t0)) + offset
pub fn exponential(times: &[f64], amplitude: f64, decay_rate: f64, offset: f64) -> Vec<f64> {
    debug_assert!(!times.is_empty(), "times must not be empty");
    let t0 = times[0];
    times
        .iter()
        .map(|&t| amplitude * (-decay_rate * (t - t0)).exp() + offset)
        .collect()
}

/// Generate a linear ramp: y = slope * (t - t0) + intercept
pub fn linear(times: &[f64], slope: f64, intercept: f64) -> Vec<f64> {
    debug_assert!(!times.is_empty(), "times must not be empty");
    let t0 = times[0];
    times
        .iter()
        .map(|&t| slope * (t - t0) + intercept)
        .collect()
}

/// Generate a Heaviside step function.
pub fn step(times: &[f64], step_time: f64, step_value: f64, initial_value: f64) -> Vec<f64> {
    debug_assert!(!times.is_empty(), "times must not be empty");
    times
        .iter()
        .map(|&t| {
            if t >= step_time {
                step_value
            } else {
                initial_value
            }
        })
        .collect()
}

/// Generate a square wave: y = amplitude * sign(sin(2π * freq * t))
pub fn square(times: &[f64], frequency: f64, amplitude: f64, duty: f64) -> Vec<f64> {
    debug_assert!(!times.is_empty(), "times must not be empty");
    debug_assert!(frequency >= 0.0, "frequency must be non-negative");
    debug_assert!((0.0..=1.0).contains(&duty), "duty cycle must be in [0, 1]");
    times
        .iter()
        .map(|&t| {
            let phase = (t * frequency).fract();
            let phase = if phase < 0.0 { phase + 1.0 } else { phase };
            if phase < duty {
                amplitude
            } else {
                -amplitude
            }
        })
        .collect()
}

/// Generate a triangle wave with given frequency and amplitude.
pub fn triangle(times: &[f64], frequency: f64, amplitude: f64) -> Vec<f64> {
    debug_assert!(!times.is_empty(), "times must not be empty");
    debug_assert!(frequency >= 0.0, "frequency must be non-negative");
    times
        .iter()
        .map(|&t| {
            let phase = (t * frequency).fract();
            let phase = if phase < 0.0 { phase + 1.0 } else { phase };
            // Triangle: rises 0→1 in first half, falls 1→0 in second half
            let tri = if phase < 0.5 {
                4.0 * phase - 1.0
            } else {
                3.0 - 4.0 * phase
            };
            amplitude * tri
        })
        .collect()
}

/// Generate a linear chirp (frequency sweep from f0 to f1).
pub fn chirp(times: &[f64], f0: f64, f1: f64, amplitude: f64) -> Vec<f64> {
    debug_assert!(!times.is_empty(), "times must not be empty");
    debug_assert!(f0 >= 0.0, "f0 must be non-negative");
    debug_assert!(f1 >= 0.0, "f1 must be non-negative");
    let t0 = times[0];
    let t_end = times[times.len() - 1];
    let duration = t_end - t0;
    if duration <= 0.0 {
        return vec![0.0; times.len()];
    }
    let rate = (f1 - f0) / duration;
    times
        .iter()
        .map(|&t| {
            let dt = t - t0;
            amplitude * (2.0 * PI * (f0 * dt + 0.5 * rate * dt * dt)).sin()
        })
        .collect()
}

/// Evaluate a polynomial: y = c0 + c1·t + c2·t² + ... (ascending order coefficients).
pub fn polynomial(times: &[f64], coefficients: &[f64]) -> Vec<f64> {
    debug_assert!(!times.is_empty(), "times must not be empty");
    let t0 = times[0];
    times
        .iter()
        .map(|&t| {
            let dt = t - t0;
            let mut result = 0.0;
            let mut power = 1.0;
            for &c in coefficients {
                result += c * power;
                power *= dt;
            }
            result
        })
        .collect()
}

/// Generate a rectangular pulse.
pub fn pulse(
    times: &[f64],
    start_time: f64,
    duration: f64,
    amplitude: f64,
    baseline: f64,
) -> Vec<f64> {
    debug_assert!(!times.is_empty(), "times must not be empty");
    debug_assert!(duration >= 0.0, "duration must be non-negative");
    let end_time = start_time + duration;
    times
        .iter()
        .map(|&t| {
            if t >= start_time && t < end_time {
                amplitude
            } else {
                baseline
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// PyO3 Bindings (feature-gated)
// ---------------------------------------------------------------------------
#[cfg(feature = "python")]
pub mod py_bindings {
    use super::*;
    use numpy::{PyArray1, PyReadonlyArray1};
    use pyo3::prelude::*;

    #[pyfunction]
    #[pyo3(name = "sinusoid")]
    pub fn py_sinusoid<'py>(
        py: Python<'py>,
        times: PyReadonlyArray1<'py, f64>,
        amplitude: f64,
        frequency: f64,
        phase: f64,
        offset: f64,
    ) -> Bound<'py, PyArray1<f64>> {
        let t = times.as_slice().unwrap();
        let result = sinusoid(t, amplitude, frequency, phase, offset);
        PyArray1::from_vec(py, result)
    }

    #[pyfunction]
    #[pyo3(name = "cosine")]
    pub fn py_cosine<'py>(
        py: Python<'py>,
        times: PyReadonlyArray1<'py, f64>,
        amplitude: f64,
        frequency: f64,
        phase: f64,
        offset: f64,
    ) -> Bound<'py, PyArray1<f64>> {
        let t = times.as_slice().unwrap();
        let result = cosine(t, amplitude, frequency, phase, offset);
        PyArray1::from_vec(py, result)
    }

    #[pyfunction]
    #[pyo3(name = "exponential")]
    pub fn py_exponential<'py>(
        py: Python<'py>,
        times: PyReadonlyArray1<'py, f64>,
        amplitude: f64,
        decay_rate: f64,
        offset: f64,
    ) -> Bound<'py, PyArray1<f64>> {
        let t = times.as_slice().unwrap();
        let result = exponential(t, amplitude, decay_rate, offset);
        PyArray1::from_vec(py, result)
    }

    #[pyfunction]
    #[pyo3(name = "linear")]
    pub fn py_linear<'py>(
        py: Python<'py>,
        times: PyReadonlyArray1<'py, f64>,
        slope: f64,
        intercept: f64,
    ) -> Bound<'py, PyArray1<f64>> {
        let t = times.as_slice().unwrap();
        let result = linear(t, slope, intercept);
        PyArray1::from_vec(py, result)
    }

    #[pyfunction]
    #[pyo3(name = "step")]
    pub fn py_step<'py>(
        py: Python<'py>,
        times: PyReadonlyArray1<'py, f64>,
        step_time: f64,
        step_value: f64,
        initial_value: f64,
    ) -> Bound<'py, PyArray1<f64>> {
        let t = times.as_slice().unwrap();
        let result = step(t, step_time, step_value, initial_value);
        PyArray1::from_vec(py, result)
    }

    #[pyfunction]
    #[pyo3(name = "square")]
    pub fn py_square<'py>(
        py: Python<'py>,
        times: PyReadonlyArray1<'py, f64>,
        frequency: f64,
        amplitude: f64,
        duty: f64,
    ) -> Bound<'py, PyArray1<f64>> {
        let t = times.as_slice().unwrap();
        let result = square(t, frequency, amplitude, duty);
        PyArray1::from_vec(py, result)
    }

    #[pyfunction]
    #[pyo3(name = "triangle")]
    pub fn py_triangle<'py>(
        py: Python<'py>,
        times: PyReadonlyArray1<'py, f64>,
        frequency: f64,
        amplitude: f64,
    ) -> Bound<'py, PyArray1<f64>> {
        let t = times.as_slice().unwrap();
        let result = triangle(t, frequency, amplitude);
        PyArray1::from_vec(py, result)
    }

    #[pyfunction]
    #[pyo3(name = "chirp")]
    pub fn py_chirp<'py>(
        py: Python<'py>,
        times: PyReadonlyArray1<'py, f64>,
        f0: f64,
        f1: f64,
        amplitude: f64,
    ) -> Bound<'py, PyArray1<f64>> {
        let t = times.as_slice().unwrap();
        let result = chirp(t, f0, f1, amplitude);
        PyArray1::from_vec(py, result)
    }

    #[pyfunction]
    #[pyo3(name = "polynomial")]
    pub fn py_polynomial<'py>(
        py: Python<'py>,
        times: PyReadonlyArray1<'py, f64>,
        coefficients: PyReadonlyArray1<'py, f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let t = times.as_slice().unwrap();
        let c = coefficients.as_slice().unwrap();
        let result = polynomial(t, c);
        PyArray1::from_vec(py, result)
    }

    #[pyfunction]
    #[pyo3(name = "pulse")]
    pub fn py_pulse<'py>(
        py: Python<'py>,
        times: PyReadonlyArray1<'py, f64>,
        start_time: f64,
        duration: f64,
        amplitude: f64,
        baseline: f64,
    ) -> Bound<'py, PyArray1<f64>> {
        let t = times.as_slice().unwrap();
        let result = pulse(t, start_time, duration, amplitude, baseline);
        PyArray1::from_vec(py, result)
    }
}

// ---------------------------------------------------------------------------
// Unit tests (TDD)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| start + (end - start) * (i as f64) / ((n - 1) as f64))
            .collect()
    }

    #[test]
    fn test_sinusoid_zero_at_origin() {
        let t = linspace(0.0, 1.0, 1000);
        let y = sinusoid(&t, 1.0, 1.0, 0.0, 0.0);
        assert!((y[0]).abs() < 1e-10, "sin(0) should be 0");
    }

    #[test]
    fn test_sinusoid_amplitude() {
        let t = linspace(0.0, 1.0, 10000);
        let y = sinusoid(&t, 3.0, 1.0, 0.0, 0.0);
        let max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!((max - 3.0).abs() < 0.01, "max should be ~3.0, got {}", max);
    }

    #[test]
    fn test_sinusoid_with_offset() {
        let t = linspace(0.0, 1.0, 1000);
        let y = sinusoid(&t, 1.0, 1.0, 0.0, 5.0);
        let mean: f64 = y.iter().sum::<f64>() / y.len() as f64;
        assert!(
            (mean - 5.0).abs() < 0.01,
            "mean should be ~5.0, got {}",
            mean
        );
    }

    #[test]
    fn test_cosine_one_at_origin() {
        let t = linspace(0.0, 1.0, 1000);
        let y = cosine(&t, 1.0, 1.0, 0.0, 0.0);
        assert!((y[0] - 1.0).abs() < 1e-10, "cos(0) should be 1");
    }

    #[test]
    fn test_exponential_decay() {
        let t = linspace(0.0, 10.0, 100);
        let y = exponential(&t, 1.0, 1.0, 0.0);
        assert!((y[0] - 1.0).abs() < 1e-10, "y(0) should be 1.0");
        // After several time constants, should be near zero
        assert!(y[y.len() - 1] < 0.001, "y(10) should be near 0");
    }

    #[test]
    fn test_linear_slope() {
        let t = linspace(0.0, 10.0, 100);
        let y = linear(&t, 2.0, 5.0);
        assert!((y[0] - 5.0).abs() < 1e-10, "y(0) = intercept = 5");
        assert!(
            (y[y.len() - 1] - 25.0).abs() < 0.01,
            "y(10) = 2*10 + 5 = 25"
        );
    }

    #[test]
    fn test_step_function() {
        let t = linspace(0.0, 10.0, 100);
        let y = step(&t, 5.0, 1.0, 0.0);
        assert_eq!(y[0], 0.0, "before step should be 0");
        assert_eq!(y[y.len() - 1], 1.0, "after step should be 1");
    }

    #[test]
    fn test_square_wave_values() {
        let t = linspace(0.0, 2.0, 1000);
        let y = square(&t, 2.0, 1.0, 0.5);
        // All values should be +1 or -1
        for val in &y {
            assert!(
                (*val - 1.0).abs() < 1e-10 || (*val + 1.0).abs() < 1e-10,
                "square wave value should be ±1, got {}",
                val
            );
        }
    }

    #[test]
    fn test_triangle_wave_range() {
        let t = linspace(0.0, 2.0, 1000);
        let y = triangle(&t, 2.0, 3.0);
        let max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = y.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(max <= 3.0 + 0.01, "max should be <= amplitude");
        assert!(min >= -3.0 - 0.01, "min should be >= -amplitude");
    }

    #[test]
    fn test_chirp_starts_at_f0() {
        let t = linspace(0.0, 1.0, 1000);
        let y = chirp(&t, 1.0, 10.0, 1.0);
        assert!((y[0]).abs() < 0.01, "chirp at t=0 should start near 0");
    }

    #[test]
    fn test_polynomial_constant() {
        let t = linspace(0.0, 5.0, 100);
        let y = polynomial(&t, &[7.0]);
        for val in &y {
            assert!(
                (val - 7.0).abs() < 1e-10,
                "constant poly should be 7.0, got {}",
                val
            );
        }
    }

    #[test]
    fn test_polynomial_quadratic() {
        let t = linspace(0.0, 5.0, 100);
        // y = 1 + 2t + 3t^2
        let y = polynomial(&t, &[1.0, 2.0, 3.0]);
        // at t=0: y=1
        assert!((y[0] - 1.0).abs() < 1e-10);
        // at t=5: y = 1 + 10 + 75 = 86
        assert!((y[y.len() - 1] - 86.0).abs() < 0.1);
    }

    #[test]
    fn test_pulse_shape() {
        let t = linspace(0.0, 10.0, 1000);
        let y = pulse(&t, 2.0, 3.0, 5.0, 0.0);
        assert_eq!(y[0], 0.0, "before pulse should be baseline");
        // Find a point during the pulse (t ≈ 3.5)
        let mid_idx = (1000.0 * 3.5 / 10.0) as usize;
        assert_eq!(y[mid_idx], 5.0, "during pulse should be amplitude");
    }
}
