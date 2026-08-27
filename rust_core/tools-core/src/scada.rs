//! SCADA alarm engine, safety interlock matrix, gasification process simulator,
//! and signal filters with PyO3 bindings.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::collections::HashMap;

/// Alarm state enumeration matching SCADA severity classifications.
#[pyclass(module = "tools_core.scada")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlarmState {
    Normal,
    Low,
    LoLo,
    High,
    HiHi,
}

/// Limits definition for a single tag.
#[pyclass(module = "tools_core.scada")]
#[derive(Debug, Clone, Copy)]
pub struct TagLimits {
    #[pyo3(get, set)]
    pub low: f64,
    #[pyo3(get, set)]
    pub lolo: f64,
    #[pyo3(get, set)]
    pub high: f64,
    #[pyo3(get, set)]
    pub hihi: f64,
}

#[pymethods]
impl TagLimits {
    #[new]
    pub fn new(low: f64, lolo: f64, high: f64, hihi: f64) -> Self {
        Self {
            low,
            lolo,
            high,
            hihi,
        }
    }
}

/// SCADA Alarm Engine tracking active state, severity, and acknowledgments
/// for up to 32 tags.
#[pyclass(module = "tools_core.scada")]
#[derive(Debug, Clone)]
pub struct AlarmEngine {
    #[pyo3(get)]
    pub tag_limits: HashMap<String, TagLimits>,

    // Internal trackers
    pub tag_values: HashMap<String, f64>,
    pub tag_states: HashMap<String, AlarmState>,
    pub tag_acknowledged: HashMap<String, bool>,
    pub tag_acknowledged_by: HashMap<String, Option<String>>,
}

#[pymethods]
impl AlarmEngine {
    #[new]
    pub fn new(limits: HashMap<String, HashMap<String, f64>>) -> PyResult<Self> {
        // Design by Contract: Enforce 32 tags limit
        if limits.len() > 32 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "AlarmEngine supports at most 32 tags",
            ));
        }

        let mut tag_limits = HashMap::new();
        let mut tag_values = HashMap::new();
        let mut tag_states = HashMap::new();
        let mut tag_acknowledged = HashMap::new();
        let mut tag_acknowledged_by = HashMap::new();

        for (tag_id, limit_map) in limits {
            let low = *limit_map.get("low").ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Missing 'low' limit for tag {}",
                    tag_id
                ))
            })?;
            let lolo = *limit_map.get("lolo").ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Missing 'lolo' limit for tag {}",
                    tag_id
                ))
            })?;
            let high = *limit_map.get("high").ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Missing 'high' limit for tag {}",
                    tag_id
                ))
            })?;
            let hihi = *limit_map.get("hihi").ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Missing 'hihi' limit for tag {}",
                    tag_id
                ))
            })?;

            // Design by Contract: Check thresholds are monotonic
            if lolo > low || low > high || high > hihi {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Limits for tag '{}' must satisfy lolo <= low <= high <= hihi (got lolo={}, low={}, high={}, hihi={})",
                    tag_id, lolo, low, high, hihi
                )));
            }

            tag_limits.insert(
                tag_id.clone(),
                TagLimits {
                    low,
                    lolo,
                    high,
                    hihi,
                },
            );
            tag_values.insert(tag_id.clone(), 0.0);
            tag_states.insert(tag_id.clone(), AlarmState::Normal);
            tag_acknowledged.insert(tag_id.clone(), false);
            tag_acknowledged_by.insert(tag_id.clone(), None);
        }

        Ok(Self {
            tag_limits,
            tag_values,
            tag_states,
            tag_acknowledged,
            tag_acknowledged_by,
        })
    }

    /// Update tag value and evaluate state. Returns list of new events if a limit is crossed.
    pub fn update_tag(
        &mut self,
        py: Python<'_>,
        tag_id: String,
        value: f64,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let limits = self.tag_limits.get(&tag_id).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("Tag '{}' not registered", tag_id))
        })?;

        let old_state = *self.tag_states.get(&tag_id).unwrap_or(&AlarmState::Normal);
        let new_state = if value <= limits.lolo {
            AlarmState::LoLo
        } else if value <= limits.low {
            AlarmState::Low
        } else if value >= limits.hihi {
            AlarmState::HiHi
        } else if value >= limits.high {
            AlarmState::High
        } else {
            AlarmState::Normal
        };

        self.tag_values.insert(tag_id.clone(), value);

        let mut events = Vec::new();
        if new_state != old_state {
            self.tag_states.insert(tag_id.clone(), new_state);

            // Reset acknowledgment on state change
            self.tag_acknowledged.insert(tag_id.clone(), false);
            self.tag_acknowledged_by.insert(tag_id.clone(), None);

            use pyo3::types::PyDict;
            let event_dict = PyDict::new(py);
            event_dict.set_item("tag_id", &tag_id)?;
            event_dict.set_item("previous_state", old_state)?;
            event_dict.set_item("current_state", new_state)?;
            event_dict.set_item("value", value)?;
            events.push(event_dict.into_any().unbind());
        }

        Ok(events)
    }

    /// Acknowledge active alarm for tag_id. Returns true if acknowledged successfully.
    pub fn acknowledge_alarm(&mut self, tag_id: String, user: String) -> PyResult<bool> {
        if !self.tag_limits.contains_key(&tag_id) {
            return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "Tag '{}' not registered",
                tag_id
            )));
        }

        let state = *self.tag_states.get(&tag_id).unwrap_or(&AlarmState::Normal);
        if state == AlarmState::Normal {
            Ok(false)
        } else {
            self.tag_acknowledged.insert(tag_id.clone(), true);
            self.tag_acknowledged_by.insert(tag_id.clone(), Some(user));
            Ok(true)
        }
    }

    /// Returns list of active alarms and their properties.
    pub fn get_active_alarms(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        use pyo3::types::PyDict;
        let mut active = Vec::new();

        for (tag_id, state) in &self.tag_states {
            if *state != AlarmState::Normal {
                let ack = *self.tag_acknowledged.get(tag_id).unwrap_or(&false);
                let ack_by = self.tag_acknowledged_by.get(tag_id).cloned().flatten();
                let val = *self.tag_values.get(tag_id).unwrap_or(&0.0);

                let severity = match state {
                    AlarmState::Normal => 0,
                    AlarmState::Low | AlarmState::High => 1,
                    AlarmState::LoLo | AlarmState::HiHi => 2,
                };

                let dict = PyDict::new(py);
                dict.set_item("tag_id", tag_id)?;
                dict.set_item("state", *state)?;
                dict.set_item("severity", severity)?;
                dict.set_item("acknowledged", ack)?;
                dict.set_item("acknowledged_by", ack_by)?;
                dict.set_item("value", val)?;
                active.push(dict.into_any().unbind());
            }
        }
        Ok(active)
    }

    /// Returns the current alarm state and details of a tag.
    pub fn get_alarm_state(&self, py: Python<'_>, tag_id: String) -> PyResult<Py<PyAny>> {
        use pyo3::types::PyDict;
        if !self.tag_limits.contains_key(&tag_id) {
            return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "Tag '{}' not registered",
                tag_id
            )));
        }

        let state = *self.tag_states.get(&tag_id).unwrap_or(&AlarmState::Normal);
        let ack = *self.tag_acknowledged.get(&tag_id).unwrap_or(&false);
        let ack_by = self.tag_acknowledged_by.get(&tag_id).cloned().flatten();
        let val = *self.tag_values.get(&tag_id).unwrap_or(&0.0);

        let dict = PyDict::new(py);
        dict.set_item("tag_id", &tag_id)?;
        dict.set_item("state", state)?;
        dict.set_item("acknowledged", ack)?;
        dict.set_item("acknowledged_by", ack_by)?;
        dict.set_item("value", val)?;

        Ok(dict.into_any().unbind())
    }
}

/// An individual interlock rule representation.
#[derive(Clone, Debug)]
pub struct Interlock {
    pub input_tag: String,
    pub operator: String,
    pub threshold: f64,
    pub actions: Vec<(String, f64)>,
}

/// Configurable Safety Interlock Matrix.
#[pyclass(module = "tools_core.scada")]
#[derive(Clone, Debug, Default)]
pub struct InterlockMatrix {
    pub interlocks: Vec<Interlock>,
}

#[pymethods]
impl InterlockMatrix {
    #[new]
    pub fn new() -> Self {
        Self {
            interlocks: Vec::new(),
        }
    }

    /// Register a safety interlock condition.
    pub fn register_interlock(
        &mut self,
        input_tag: String,
        operator: String,
        threshold: f64,
        actions: Vec<(String, f64)>,
    ) -> PyResult<()> {
        match operator.as_str() {
            ">" | "<" | ">=" | "<=" | "==" => {}
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid operator '{}'. Must be one of: '>', '<', '>=', '<=', '=='",
                    operator
                )))
            }
        }

        self.interlocks.push(Interlock {
            input_tag,
            operator,
            threshold,
            actions,
        });
        Ok(())
    }

    /// Check tag values and return actions to apply.
    pub fn evaluate(&self, tags: HashMap<String, f64>) -> Vec<(String, f64)> {
        let mut applied_actions = Vec::new();
        for interlock in &self.interlocks {
            if let Some(&val) = tags.get(&interlock.input_tag) {
                let triggered = match interlock.operator.as_str() {
                    ">" => val > interlock.threshold,
                    "<" => val < interlock.threshold,
                    ">=" => val >= interlock.threshold,
                    "<=" => val <= interlock.threshold,
                    "==" => (val - interlock.threshold).abs() < 1e-9,
                    _ => false,
                };

                if triggered {
                    for action in &interlock.actions {
                        applied_actions.push(action.clone());
                    }
                }
            }
        }
        applied_actions
    }
}

/// Dynamic Gasification Process Simulator.
/// Computes temperature in 4 zones, syngas flow rate, and pressure drop.
#[pyclass(module = "tools_core.scada")]
#[derive(Clone, Debug)]
pub struct GasificationSimulator {
    // Measured variables
    #[pyo3(get)]
    pub t_drying: f64,
    #[pyo3(get)]
    pub t_pyrolysis: f64,
    #[pyo3(get)]
    pub t_combustion: f64,
    #[pyo3(get)]
    pub t_reduction: f64,
    #[pyo3(get)]
    pub syngas_flow: f64,
    #[pyo3(get)]
    pub pressure_drop: f64,
    #[pyo3(get)]
    pub scenario: String,

    // Internal true states (unaffected by sensor fault simulation)
    pub true_t_drying: f64,
    pub true_t_pyrolysis: f64,
    pub true_t_combustion: f64,
    pub true_t_reduction: f64,
}

#[pymethods]
impl GasificationSimulator {
    #[new]
    pub fn new() -> Self {
        Self {
            t_drying: 100.0,
            t_pyrolysis: 350.0,
            t_combustion: 900.0,
            t_reduction: 700.0,
            syngas_flow: 0.0,
            pressure_drop: 0.0,
            scenario: "normal".to_string(),
            true_t_drying: 100.0,
            true_t_pyrolysis: 350.0,
            true_t_combustion: 900.0,
            true_t_reduction: 700.0,
        }
    }

    /// Load fault scenarios.
    pub fn set_scenario(&mut self, scenario: String) -> PyResult<()> {
        match scenario.as_str() {
            "normal" | "wet_feed" | "stuck_valve" | "thermocouple_failure" => {
                self.scenario = scenario;
                Ok(())
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown scenario '{}'. Supported: 'normal', 'wet_feed', 'stuck_valve', 'thermocouple_failure'",
                scenario
            ))),
        }
    }

    /// Step the simulation forward by dt seconds under the given setpoints/inputs.
    pub fn step(&mut self, dt: f64, inputs: HashMap<String, f64>) -> PyResult<()> {
        // Design by Contract: Enforce dt > 0
        if dt <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "dt must be positive",
            ));
        }

        let oxygen_setpoint = *inputs.get("oxygen_setpoint").unwrap_or(&0.0);
        let steam_setpoint = *inputs.get("steam_setpoint").unwrap_or(&0.0);
        let feedstock_rate = *inputs.get("feedstock_rate").unwrap_or(&0.0);
        let torch_power = *inputs.get("torch_power").unwrap_or(&0.0);

        // Design by Contract: Enforce non-negative physical inputs
        if oxygen_setpoint < 0.0
            || steam_setpoint < 0.0
            || feedstock_rate < 0.0
            || torch_power < 0.0
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Inputs must be non-negative",
            ));
        }

        // Fault scenario: oxygen valve stuck
        let actual_oxygen = if self.scenario == "stuck_valve" {
            15.0 // Stuck open at 15 slpm
        } else {
            oxygen_setpoint
        };

        // Steady state temperatures calculation
        let mut combustion_ss = 200.0 + 25.0 * actual_oxygen + 5.0 * torch_power
            - 8.0 * steam_setpoint
            - 0.5 * feedstock_rate;

        // Fault scenario: high feedstock moisture cools combustion
        if self.scenario == "wet_feed" {
            combustion_ss -= 300.0;
        }
        combustion_ss = combustion_ss.max(100.0);

        let pyrolysis_ss = (0.65 * self.true_t_combustion + 1.5 * feedstock_rate).max(50.0);
        let drying_ss = (0.45 * self.true_t_pyrolysis).max(40.0);
        let reduction_ss = (0.75 * self.true_t_combustion - 4.0 * steam_setpoint).max(50.0);

        // Discretized first-order response equations
        let alpha_comb = (dt / 5.0).min(1.0);
        let alpha_pyro = (dt / 10.0).min(1.0);
        let alpha_dry = (dt / 15.0).min(1.0);
        let alpha_red = (dt / 8.0).min(1.0);

        self.true_t_combustion += alpha_comb * (combustion_ss - self.true_t_combustion);
        self.true_t_pyrolysis += alpha_pyro * (pyrolysis_ss - self.true_t_pyrolysis);
        self.true_t_drying += alpha_dry * (drying_ss - self.true_t_drying);
        self.true_t_reduction += alpha_red * (reduction_ss - self.true_t_reduction);

        // Fault scenario: stuck/failed thermocouple sensor
        if self.scenario == "thermocouple_failure" {
            self.t_pyrolysis = 25.0; // Reads room temperature
        } else {
            self.t_pyrolysis = self.true_t_pyrolysis;
        }

        self.t_combustion = self.true_t_combustion;
        self.t_drying = self.true_t_drying;
        self.t_reduction = self.true_t_reduction;

        // Syngas flow model
        let flow_ss = (0.01 * self.true_t_combustion)
            * (0.4 * feedstock_rate + 0.6 * actual_oxygen + 0.3 * steam_setpoint);
        let alpha_flow = (dt / 1.0).min(1.0);
        self.syngas_flow += alpha_flow * (flow_ss - self.syngas_flow);
        self.syngas_flow = self.syngas_flow.max(0.0);

        // Pressure drop model
        let dp_ss = 0.003 * self.syngas_flow * self.syngas_flow * (self.true_t_reduction / 273.15);
        self.pressure_drop = dp_ss.max(0.0);

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Core Signal Filters
// ---------------------------------------------------------------------------

/// Apply centered moving-average smoothing (matching NumPy "same").
pub fn moving_average(values: &[f64], window_size: usize) -> Vec<f64> {
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

/// Apply first-order recursive exponential smoothing.
pub fn exponential_smoothing(values: &[f64], alpha: f64) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(values.len());
    out.push(values[0]);
    for &value in values.iter().skip(1) {
        let previous = *out.last().unwrap();
        out.push(alpha * value + (1.0 - alpha) * previous);
    }
    out
}

/// Computes the Savitzky-Golay convolution coefficients for a given window size and polynomial order.
fn compute_savgol_coefficients(window_size: usize, poly_order: usize) -> Result<Vec<f64>, String> {
    if window_size % 2 == 0 {
        return Err("window_size must be odd".to_string());
    }
    if poly_order >= window_size {
        return Err("poly_order must be less than window_size".to_string());
    }
    let k = (window_size - 1) / 2;
    let m = poly_order + 1;

    // 1. Build Vandermonde matrix J of size window_size x m
    let mut j_mat = vec![vec![0.0; m]; window_size];
    for i in 0..window_size {
        let x = (i as f64) - (k as f64);
        for j in 0..m {
            j_mat[i][j] = x.powi(j as i32);
        }
    }

    // 2. Compute J_T = J^T (size m x window_size)
    let mut j_t = vec![vec![0.0; window_size]; m];
    for i in 0..m {
        for j in 0..window_size {
            j_t[i][j] = j_mat[j][i];
        }
    }

    // 3. Compute A = J^T * J (size m x m)
    let mut a = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in 0..m {
            let mut sum = 0.0;
            for l in 0..window_size {
                sum += j_t[i][l] * j_mat[l][j];
            }
            a[i][j] = sum;
        }
    }

    // 4. Invert A using Gauss-Jordan elimination
    let mut a_inv = vec![vec![0.0; m]; m];
    for i in 0..m {
        a_inv[i][i] = 1.0;
    }

    let mut a_temp = a;
    for i in 0..m {
        // Find pivot
        let mut pivot_row = i;
        for r in (i + 1)..m {
            if a_temp[r][i].abs() > a_temp[pivot_row][i].abs() {
                pivot_row = r;
            }
        }
        if a_temp[pivot_row][i].abs() < 1e-12 {
            return Err("Matrix inversion failed: singular Vandermonde matrix".to_string());
        }
        if pivot_row != i {
            a_temp.swap(i, pivot_row);
            a_inv.swap(i, pivot_row);
        }
        let factor = a_temp[i][i];
        for c in 0..m {
            a_temp[i][c] /= factor;
            a_inv[i][c] /= factor;
        }
        for r in 0..m {
            if r != i {
                let factor = a_temp[r][i];
                for c in 0..m {
                    a_temp[r][c] -= factor * a_temp[i][c];
                    a_inv[r][c] -= factor * a_inv[i][c];
                }
            }
        }
    }

    // 5. Coefficients c = first row of A_inv * J_T
    let mut coeffs = vec![0.0; window_size];
    for j in 0..window_size {
        let mut sum = 0.0;
        for r in 0..m {
            sum += a_inv[0][r] * j_t[r][j];
        }
        coeffs[j] = sum;
    }

    Ok(coeffs)
}

/// Apply Savitzky-Golay polynomial smoothing to a 1D signal.
pub fn savitzky_golay(
    values: &[f64],
    window_size: usize,
    poly_order: usize,
) -> Result<Vec<f64>, String> {
    let n = values.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if window_size % 2 == 0 {
        return Err("window_size must be odd".to_string());
    }
    if poly_order >= window_size {
        return Err("poly_order must be less than window_size".to_string());
    }

    let coeffs = compute_savgol_coefficients(window_size, poly_order)?;
    let k = (window_size - 1) / 2;

    let mut out = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for w_idx in 0..window_size {
            let val_idx = if i + w_idx < k {
                0
            } else if i + w_idx - k >= n {
                n - 1
            } else {
                i + w_idx - k
            };
            sum += coeffs[w_idx] * values[val_idx];
        }
        out[i] = sum;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// PyO3 Bindings for Filters
// ---------------------------------------------------------------------------

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

#[pyfunction]
#[pyo3(name = "savitzky_golay")]
pub fn py_savitzky_golay<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    window_size: usize,
    poly_order: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if window_size % 2 == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "window_size must be odd",
        ));
    }
    if poly_order >= window_size {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "poly_order must be less than window_size",
        ));
    }
    let v = values.as_slice().unwrap().to_vec();
    let result = py
        .detach(move || savitzky_golay(&v, window_size, poly_order))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    Ok(PyArray1::from_vec(py, result))
}

// ---------------------------------------------------------------------------
// Rust-native Unit Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alarm_engine_basic() {
        let mut limits = HashMap::new();
        let mut t1 = HashMap::new();
        t1.insert("lolo".to_string(), 10.0);
        t1.insert("low".to_string(), 20.0);
        t1.insert("high".to_string(), 80.0);
        t1.insert("hihi".to_string(), 90.0);
        limits.insert("T1".to_string(), t1);

        let mut engine = AlarmEngine::new(limits).unwrap();

        // Initial normal value
        Python::initialize();
        Python::attach(|py| {
            let events = engine.update_tag(py, "T1".to_string(), 50.0).unwrap();
            assert!(events.is_empty());
            assert_eq!(engine.tag_states["T1"], AlarmState::Normal);

            // Go to High
            let events = engine.update_tag(py, "T1".to_string(), 85.0).unwrap();
            assert_eq!(events.len(), 1);
            assert_eq!(engine.tag_states["T1"], AlarmState::High);
            assert_eq!(engine.tag_acknowledged["T1"], false);

            // Acknowledge alarm
            let ack = engine
                .acknowledge_alarm("T1".to_string(), "OperatorA".to_string())
                .unwrap();
            assert!(ack);
            assert_eq!(engine.tag_acknowledged["T1"], true);
            assert_eq!(
                engine.tag_acknowledged_by["T1"],
                Some("OperatorA".to_string())
            );

            // Go to Normal
            let events = engine.update_tag(py, "T1".to_string(), 50.0).unwrap();
            assert_eq!(events.len(), 1);
            assert_eq!(engine.tag_states["T1"], AlarmState::Normal);
            assert_eq!(engine.tag_acknowledged["T1"], false);
        });
    }

    #[test]
    fn test_interlock_matrix() {
        let mut matrix = InterlockMatrix::new();
        matrix
            .register_interlock(
                "T_pyrolysis".to_string(),
                ">".to_string(),
                95.0,
                vec![("O2_valve".to_string(), 0.0), ("E_stop".to_string(), 1.0)],
            )
            .unwrap();

        let mut tags = HashMap::new();
        tags.insert("T_pyrolysis".to_string(), 90.0);
        let actions = matrix.evaluate(tags.clone());
        assert!(actions.is_empty());

        tags.insert("T_pyrolysis".to_string(), 96.0);
        let actions = matrix.evaluate(tags);
        assert_eq!(actions.len(), 2);
        assert_eq!(actions[0], ("O2_valve".to_string(), 0.0));
        assert_eq!(actions[1], ("E_stop".to_string(), 1.0));
    }

    #[test]
    fn test_gasification_simulator() {
        let mut sim = GasificationSimulator::new();
        let mut inputs = HashMap::new();
        inputs.insert("oxygen_setpoint".to_string(), 10.0);
        inputs.insert("steam_setpoint".to_string(), 2.0);
        inputs.insert("feedstock_rate".to_string(), 5.0);
        inputs.insert("torch_power".to_string(), 50.0);

        sim.step(0.1, inputs.clone()).unwrap();
        assert!(sim.t_combustion > 0.0);
        assert!(sim.t_pyrolysis > 0.0);

        // Stuck valve scenario
        sim.set_scenario("stuck_valve".to_string()).unwrap();
        sim.step(0.1, inputs.clone()).unwrap();
        assert_eq!(sim.scenario, "stuck_valve");
    }

    #[test]
    fn test_savgol_coefficients() {
        let coeffs = compute_savgol_coefficients(5, 2).unwrap();
        // Checked analytically: [-3/35, 12/35, 17/35, 12/35, -3/35]
        assert!((coeffs[0] - (-3.0 / 35.0)).abs() < 1e-12);
        assert!((coeffs[1] - (12.0 / 35.0)).abs() < 1e-12);
        assert!((coeffs[2] - (17.0 / 35.0)).abs() < 1e-12);
        assert!((coeffs[3] - (12.0 / 35.0)).abs() < 1e-12);
        assert!((coeffs[4] - (-3.0 / 35.0)).abs() < 1e-12);
    }
}
