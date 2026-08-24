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
//! - `moving_average` — centered rectangular smoothing matching NumPy `same`
//! - `exponential_smoothing` — first-order recursive smoothing

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

// ---------------------------------------------------------------------------
// Edge-preserving smoothing
// ---------------------------------------------------------------------------

/// Apply a 1-D bilateral filter to a signal.
///
/// The bilateral filter is an edge-preserving smoother that combines a
/// spatial Gaussian kernel (distance from the centre sample) with an
/// intensity Gaussian kernel (value similarity to the centre sample). It is
/// the canonical Python implementation in
/// `signal_toolkit.filters.apply_bilateral_filter` ported sample-for-sample
/// — the per-sample window is asymmetric near the edges, matching the
/// reference `start = max(0, i - half)`, `end = min(n, i + half + 1)`
/// convention used by NumPy.
///
/// # Parameters
/// - `values`: input signal samples.
/// - `window_size`: full window width (the half-window is `window_size / 2`,
///   integer division — odd and even values are accepted to preserve parity
///   with the existing Python helper).
/// - `sigma_space`: spatial Gaussian sigma; must be > 0.
/// - `sigma_intensity`: intensity Gaussian sigma; must be > 0.
///
/// # Numerical notes
/// - Weights are accumulated with the same `+ 1e-10` denominator stabiliser
///   the Python reference uses, so per-sample outputs match to within
///   floating-point rounding tolerance (`< 1e-12` typical).
/// - Single-pass and allocation-free in the hot loop. The PyO3 wrapper
///   releases the GIL via `py.allow_threads`, so callers can drive the
///   filter from worker threads without serialising on Python.
pub fn bilateral_filter(
    values: &[f64],
    window_size: usize,
    sigma_space: f64,
    sigma_intensity: f64,
) -> Vec<f64> {
    debug_assert!(sigma_space > 0.0, "sigma_space must be > 0");
    debug_assert!(sigma_intensity > 0.0, "sigma_intensity must be > 0");

    let n = values.len();
    let mut out = vec![0.0; n];
    if n == 0 {
        return out;
    }

    let half = window_size / 2;
    let inv_two_sigma_space_sq = 1.0 / (2.0 * sigma_space * sigma_space);
    let inv_two_sigma_int_sq = 1.0 / (2.0 * sigma_intensity * sigma_intensity);

    for i in 0..n {
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(n);
        let centre = values[i];

        let mut weight_sum = 0.0;
        let mut value_sum = 0.0;
        // The index `j` is used both to compute the signed spatial offset
        // (`j - i`) and to index `values[j]`; an iterator-based form would
        // require zipping `(start..end)` with `&values[start..end]` and is
        // strictly less clear here, so silence clippy locally.
        #[allow(clippy::needless_range_loop)]
        for j in start..end {
            // Spatial weight: signed offset from the centre.
            let dpos = j as f64 - i as f64;
            let spatial = (-dpos * dpos * inv_two_sigma_space_sq).exp();

            // Intensity weight: value similarity to the centre sample.
            let dval = values[j] - centre;
            let intensity = (-dval * dval * inv_two_sigma_int_sq).exp();

            let w = spatial * intensity;
            weight_sum += w;
            value_sum += w * values[j];
        }

        // Match the Python reference's `+ 1e-10` denominator stabiliser so
        // outputs stay bit-for-bit comparable across the parity boundary.
        out[i] = value_sum / (weight_sum + 1e-10);
    }

    out
}

/// Apply centered moving-average smoothing.
///
/// The output alignment matches `numpy.convolve(values, ones(window)/window, mode="same")`.
/// Near the boundaries this is equivalent to zero-padding outside the input signal.
pub fn moving_average(values: &[f64], window_size: usize) -> Vec<f64> {
    debug_assert!(window_size > 0, "window_size must be >= 1");

    let n = values.len();
    if n == 0 {
        return Vec::new();
    }
    if window_size == 0 {
        return vec![f64::NAN; n];
    }

    let mut prefix = Vec::with_capacity(n + 1);
    prefix.push(0.0);
    for &value in values {
        prefix.push(prefix.last().copied().unwrap_or(0.0) + value);
    }

    let full_len = n + window_size - 1;
    let start = (window_size - 1) / 2;
    let mut out = Vec::with_capacity(n);
    for k in start..(start + n).min(full_len) {
        let input_start = k.saturating_sub(window_size - 1);
        let input_end = k.min(n - 1) + 1;
        out.push((prefix[input_end] - prefix[input_start]) / window_size as f64);
    }
    out
}

/// Apply first-order exponential smoothing.
///
/// Uses the canonical recurrence `out[0] = values[0]` and
/// `out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]`.
pub fn exponential_smoothing(values: &[f64], alpha: f64) -> Vec<f64> {
    debug_assert!(
        alpha > 0.0 && alpha <= 1.0,
        "alpha must be in the interval (0, 1]"
    );

    if values.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(values.len());
    out.push(values[0]);
    for &value in values.iter().skip(1) {
        let previous = *out.last().expect("output contains the initial sample");
        out.push(alpha * value + (1.0 - alpha) * previous);
    }
    out
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

/// Apply Least Mean Squares (LMS) adaptive filter.
///
/// Returns `(filtered, error)` where `filtered[i]` is the filter output and
/// `error[i] = reference[i] - filtered[i]`.
///
/// # Preconditions
/// - `signal.len() == reference.len()`
/// - `order >= 1`
/// - `step_size > 0`
pub fn lms_filter(
    signal: &[f64],
    reference: &[f64],
    order: usize,
    step_size: f64,
) -> (Vec<f64>, Vec<f64>) {
    debug_assert_eq!(
        signal.len(),
        reference.len(),
        "signal and reference must match"
    );
    debug_assert!(order >= 1, "order must be >= 1");
    debug_assert!(step_size > 0.0, "step_size must be > 0");

    let n = signal.len();
    let mut w = vec![0.0_f64; order];
    let mut y = vec![0.0_f64; n];
    let mut e = vec![0.0_f64; n];

    for i in order..n {
        // Dot product of weights with reversed window
        let mut yi = 0.0_f64;
        for k in 0..order {
            yi += w[k] * signal[i - order + k];
        }
        y[i] = yi;
        e[i] = reference[i] - yi;
        // Weight update: w += step_size * error * x_window (reversed)
        let ei = e[i];
        for k in 0..order {
            w[k] += step_size * ei * signal[i - order + k];
        }
    }
    (y, e)
}

/// Apply Recursive Least Squares (RLS) adaptive filter.
///
/// Returns `(filtered, error)` where `filtered[i]` is the filter output and
/// `error[i] = reference[i] - filtered[i]`.
///
/// # Preconditions
/// - `signal.len() == reference.len()`
/// - `order >= 1`
/// - `forgetting_factor` in `(0, 1]`
/// - `delta > 0`
pub fn rls_filter(
    signal: &[f64],
    reference: &[f64],
    order: usize,
    forgetting_factor: f64,
    delta: f64,
) -> (Vec<f64>, Vec<f64>) {
    debug_assert_eq!(
        signal.len(),
        reference.len(),
        "signal and reference must match"
    );
    debug_assert!(order >= 1, "order must be >= 1");
    debug_assert!(forgetting_factor > 0.0 && forgetting_factor <= 1.0);
    debug_assert!(delta > 0.0, "delta must be > 0");

    let n = signal.len();
    let mut w = vec![0.0_f64; order];
    // P = (1/delta) * I
    let mut p: Vec<Vec<f64>> = (0..order)
        .map(|i| {
            let mut row = vec![0.0_f64; order];
            row[i] = 1.0 / delta;
            row
        })
        .collect();
    let mut y = vec![0.0_f64; n];
    let mut e = vec![0.0_f64; n];
    let lam = forgetting_factor;
    let inv_lam = 1.0 / lam;

    let mut x_window = vec![0.0_f64; order];
    let mut k_vec = vec![0.0_f64; order];
    let mut px = vec![0.0_f64; order]; // P * x

    for i in order..n {
        // Build reversed window: x_window[k] = signal[i - order + k]
        for k in 0..order {
            x_window[k] = signal[i - order + k];
        }

        // Output
        let yi: f64 = w.iter().zip(x_window.iter()).map(|(wi, xi)| wi * xi).sum();
        y[i] = yi;
        e[i] = reference[i] - yi;

        // px = P * x_window
        for row in 0..order {
            px[row] = p[row]
                .iter()
                .zip(x_window.iter())
                .map(|(pij, xj)| pij * xj)
                .sum();
        }
        // denom = lam + x_window^T * px
        let denom: f64 = lam
            + x_window
                .iter()
                .zip(px.iter())
                .map(|(xi, pxi)| xi * pxi)
                .sum::<f64>();
        // k = px / denom
        for row in 0..order {
            k_vec[row] = px[row] / denom;
        }
        // Weight update: w += k * error
        let ei = e[i];
        for row in 0..order {
            w[row] += k_vec[row] * ei;
        }
        // P update: P = (P - k * (px^T)) / lam
        // Note: k * px^T is the outer product
        for row in 0..order {
            for col in 0..order {
                p[row][col] = (p[row][col] - k_vec[row] * px[col]) * inv_lam;
            }
        }
    }
    (y, e)
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

    /// PyO3 binding for `bilateral_filter`.
    ///
    /// Mirrors the `apply_bilateral_filter` signature in
    /// `signal_toolkit.filters` so the eventual Python facade can be a
    /// one-line swap. The compute is run inside `py.allow_threads` so
    /// long signals do not block Python worker threads.
    #[pyfunction]
    #[pyo3(name = "bilateral_filter")]
    pub fn py_bilateral_filter<'py>(
        py: Python<'py>,
        values: PyReadonlyArray1<'py, f64>,
        window_size: usize,
        sigma_space: f64,
        sigma_intensity: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        // Use `<=` rather than `!(x > 0.0)` so NaN inputs are caught as
        // invalid here too (NaN comparisons return false, which would
        // otherwise slip past a `> 0.0` precondition).
        if sigma_space.is_nan() || sigma_space <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "sigma_space must be > 0",
            ));
        }
        if sigma_intensity.is_nan() || sigma_intensity <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "sigma_intensity must be > 0",
            ));
        }
        // Copy the input so the GIL-free section owns its data — the
        // `PyReadonlyArray1` borrow cannot cross `allow_threads`.
        let v = values.as_slice().unwrap().to_vec();
        let result =
            py.detach(move || bilateral_filter(&v, window_size, sigma_space, sigma_intensity));
        Ok(PyArray1::from_vec(py, result))
    }

    /// PyO3 binding for `moving_average`.
    #[pyfunction]
    #[pyo3(name = "moving_average")]
    pub fn py_moving_average<'py>(
        py: Python<'py>,
        values: PyReadonlyArray1<'py, f64>,
        window_size: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if window_size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "window_size must be >= 1",
            ));
        }
        let v = values.as_slice().unwrap().to_vec();
        let result = py.detach(move || moving_average(&v, window_size));
        Ok(PyArray1::from_vec(py, result))
    }

    /// PyO3 binding for `exponential_smoothing`.
    #[pyfunction]
    #[pyo3(name = "exponential_smoothing")]
    pub fn py_exponential_smoothing<'py>(
        py: Python<'py>,
        values: PyReadonlyArray1<'py, f64>,
        alpha: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if alpha.is_nan() || alpha <= 0.0 || alpha > 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "alpha must be in (0, 1]",
            ));
        }
        let v = values.as_slice().unwrap().to_vec();
        let result = py.detach(move || exponential_smoothing(&v, alpha));
        Ok(PyArray1::from_vec(py, result))
    }

    /// PyO3 binding for `lms_filter`.
    ///
    /// Runs the full LMS loop natively in Rust without acquiring the GIL on
    /// every sample. Returns `(filtered_values, error_values)` as two
    /// contiguous NumPy arrays.
    ///
    /// # Arguments
    /// * `signal` — Input signal samples (read-only NumPy array).
    /// * `reference` — Desired / reference signal samples (same length).
    /// * `order` — Filter order (tap count); must be `>= 1`.
    /// * `step_size` — LMS step size µ (convergence parameter); must be `> 0`.
    #[pyfunction]
    #[pyo3(name = "lms_filter", signature = (signal, reference, order=10, step_size=0.01))]
    pub fn py_lms_filter<'py>(
        py: Python<'py>,
        signal: PyReadonlyArray1<'py, f64>,
        reference: PyReadonlyArray1<'py, f64>,
        order: usize,
        step_size: f64,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        if order == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "order must be >= 1",
            ));
        }
        if step_size <= 0.0 || step_size.is_nan() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "step_size must be > 0",
            ));
        }
        let x = signal.as_slice().unwrap().to_vec();
        let d = reference.as_slice().unwrap().to_vec();
        if x.len() != d.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "signal and reference must have the same length",
            ));
        }
        let (y, e) = py.detach(move || lms_filter(&x, &d, order, step_size));
        Ok((PyArray1::from_vec(py, y), PyArray1::from_vec(py, e)))
    }

    /// PyO3 binding for `rls_filter`.
    ///
    /// Runs the full RLS loop natively in Rust. Returns
    /// `(filtered_values, error_values)` as two contiguous NumPy arrays.
    ///
    /// # Arguments
    /// * `signal` — Input signal samples (read-only NumPy array).
    /// * `reference` — Desired / reference signal samples (same length).
    /// * `order` — Filter order (tap count); must be `>= 1`.
    /// * `forgetting_factor` — RLS forgetting factor λ; typically in (0, 1].
    /// * `delta` — Initial diagonal loading for the inverse correlation matrix P₀ = (1/δ)·I.
    #[pyfunction]
    #[pyo3(
        name = "rls_filter",
        signature = (signal, reference, order=10, forgetting_factor=0.99, delta=0.01)
    )]
    pub fn py_rls_filter<'py>(
        py: Python<'py>,
        signal: PyReadonlyArray1<'py, f64>,
        reference: PyReadonlyArray1<'py, f64>,
        order: usize,
        forgetting_factor: f64,
        delta: f64,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        if order == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "order must be >= 1",
            ));
        }
        if forgetting_factor <= 0.0 || forgetting_factor > 1.0 || forgetting_factor.is_nan() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "forgetting_factor must be in (0, 1]",
            ));
        }
        if delta <= 0.0 || delta.is_nan() {
            return Err(pyo3::exceptions::PyValueError::new_err("delta must be > 0"));
        }
        let x = signal.as_slice().unwrap().to_vec();
        let d = reference.as_slice().unwrap().to_vec();
        if x.len() != d.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "signal and reference must have the same length",
            ));
        }
        let (y, e) = py.detach(move || rls_filter(&x, &d, order, forgetting_factor, delta));
        Ok((PyArray1::from_vec(py, y), PyArray1::from_vec(py, e)))
    }
} // end pub mod py_bindings

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

    #[test]
    fn test_bilateral_filter_constant_signal_is_unchanged() {
        // A constant signal has zero intensity gradient, so the filter must
        // pass it through almost unchanged. The `+ 1e-10` denominator
        // stabiliser (matched against the Python reference for parity)
        // introduces a bias of order `value * 1e-10 / weight_sum`, so we
        // assert against a 1e-9 tolerance rather than full precision.
        let values = vec![3.5_f64; 64];
        let out = bilateral_filter(&values, 5, 1.0, 0.1);
        for (i, v) in out.iter().enumerate() {
            assert!(
                (v - 3.5).abs() < 1e-9,
                "constant in must equal constant out at i={}, got {}",
                i,
                v
            );
        }
    }

    #[test]
    fn test_bilateral_filter_preserves_step_edge() {
        // Build a sharp step. With small sigma_intensity, the filter should
        // preserve the discontinuity (samples on either side of the step
        // stay near their original values, not midpoint-averaged).
        let mut values = vec![0.0_f64; 32];
        for v in values.iter_mut().skip(16) {
            *v = 1.0;
        }
        let out = bilateral_filter(&values, 5, 1.0, 0.05);
        // Samples well away from the edge are unchanged.
        assert!(out[5].abs() < 1e-9, "left plateau should stay at 0");
        assert!(
            (out[26] - 1.0).abs() < 1e-9,
            "right plateau should stay at 1"
        );
        // Edge samples are pulled toward their own value, not the midpoint.
        assert!(out[15] < 0.1, "last 0-sample must not jump toward 0.5");
        assert!(out[16] > 0.9, "first 1-sample must not drop toward 0.5");
    }

    #[test]
    fn test_bilateral_filter_empty_input() {
        let out = bilateral_filter(&[], 5, 1.0, 0.1);
        assert!(out.is_empty(), "empty in must give empty out");
    }

    #[test]
    fn test_moving_average_matches_numpy_same_for_odd_window() {
        let values = [1.0, 2.0, 4.0, 8.0, 16.0];
        let out = moving_average(&values, 3);
        let expected = [1.0, 7.0 / 3.0, 14.0 / 3.0, 28.0 / 3.0, 8.0];
        for (actual, expected) in out.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn test_moving_average_matches_numpy_same_for_even_window() {
        let values = [1.0, 2.0, 4.0, 8.0];
        let out = moving_average(&values, 4);
        let expected = [0.75, 1.75, 3.75, 3.5];
        for (actual, expected) in out.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn test_exponential_smoothing_matches_python_recurrence() {
        let values = [10.0, 14.0, 13.0, 20.0];
        let out = exponential_smoothing(&values, 0.25);
        let expected = [10.0, 11.0, 11.5, 13.625];
        for (actual, expected) in out.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn test_lms_output_length_matches_input() {
        let n = 200;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let d: Vec<f64> = x.iter().map(|v| v * 0.8 + 0.1).collect();
        let (y, e) = lms_filter(&x, &d, 10, 0.01);
        assert_eq!(y.len(), n, "filtered length must match input");
        assert_eq!(e.len(), n, "error length must match input");
    }

    #[test]
    fn test_lms_error_decreases_over_stationary_input() {
        // For a stationary mapping d = 2*x, the LMS filter should learn the
        // coefficients and drive the error toward zero over time.
        let n = 2000;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.05).sin()).collect();
        let d: Vec<f64> = x.iter().map(|v| v * 2.0).collect();
        let (_, e) = lms_filter(&x, &d, 10, 0.005);
        let early_err: f64 = e[10..60].iter().map(|v| v * v).sum::<f64>() / 50.0;
        let late_err: f64 = e[1900..].iter().map(|v| v * v).sum::<f64>() / 100.0;
        assert!(
            late_err < early_err,
            "LMS error must decrease over time (early={early_err:.4}, late={late_err:.4})"
        );
    }

    #[test]
    fn test_rls_output_length_matches_input() {
        let n = 200;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let d: Vec<f64> = x.iter().map(|v| v * 0.5 + 0.05).collect();
        let (y, e) = rls_filter(&x, &d, 8, 0.99, 0.01);
        assert_eq!(y.len(), n);
        assert_eq!(e.len(), n);
    }

    #[test]
    fn test_rls_converges_faster_than_lms_on_stationary_input() {
        // RLS has faster convergence rate than LMS by design — verify
        // the squared error sum over the first 200 samples is lower for RLS.
        let n = 500;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).cos()).collect();
        let d: Vec<f64> = x.iter().map(|v| v * 1.5 - 0.2).collect();
        let (_, e_lms) = lms_filter(&x, &d, 10, 0.01);
        let (_, e_rls) = rls_filter(&x, &d, 10, 0.99, 0.01);
        let mse_lms: f64 = e_lms[10..200].iter().map(|v| v * v).sum::<f64>();
        let mse_rls: f64 = e_rls[10..200].iter().map(|v| v * v).sum::<f64>();
        assert!(
            mse_rls < mse_lms,
            "RLS should converge faster than LMS (rls={mse_rls:.4}, lms={mse_lms:.4})"
        );
    }
}
