use serde::{Deserialize, Serialize};
use std::process::Command;

/// Python engine bridge — spawns Python subprocess for compute tasks.
///
/// The shared plot engine in `src/shared/python/plot_engine/` provides
/// PlotlyConverter.convert() which outputs Plotly.js JSON. This bridge
/// calls that converter from Rust so the React frontend can render
/// cross-platform plots.

#[derive(Debug, Serialize, Deserialize)]
struct PlotlyResult {
    data: serde_json::Value,
    layout: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct TrendlineResult {
    trend_type: String,
    coefficients: Vec<f64>,
    equation: String,
    r_squared: f64,
}

/// Compute a plot specification through the Python engine.
///
/// Takes a PlotSpec JSON string, passes it to Python's PlotlyConverter,
/// and returns the Plotly.js-compatible JSON (data + layout).
#[tauri::command]
fn compute_plot(spec_json: String) -> Result<String, String> {
    let script = format!(
        r#"
import json
from shared.python.plot_engine.plotly_converter import PlotlyConverter
from shared.python.plot_engine.specs import PlotSpec
spec = PlotSpec.model_validate_json('{}')
converter = PlotlyConverter()
result = converter.convert(spec)
print(json.dumps(result))
"#,
        spec_json.replace('\'', "\\'").replace('\n', "\\n")
    );

    run_python_script(&script)
}

/// Compute a trendline through the Python engine.
///
/// Takes x/y data and trendline parameters, returns coefficients,
/// equation string, R² value, and prediction arrays.
#[tauri::command]
fn compute_trendline(
    x: Vec<f64>,
    y: Vec<f64>,
    trend_type: String,
    degree: Option<u32>,
) -> Result<String, String> {
    let x_json = serde_json::to_string(&x).map_err(|e| e.to_string())?;
    let y_json = serde_json::to_string(&y).map_err(|e| e.to_string())?;
    let deg = degree.unwrap_or(2);

    let script = format!(
        r#"
import json
import numpy as np
from shared.python.plot_engine.trendline import compute_trendline
x = np.array({x_json})
y = np.array({y_json})
result = compute_trendline(x, y, "{trend_type}", degree={deg})
print(json.dumps({{
    "trend_type": result.trend_type,
    "coefficients": result.coefficients,
    "equation": result.equation,
    "r_squared": result.r_squared,
    "x_pred": result.x_pred.tolist(),
    "y_pred": result.y_pred.tolist(),
}}))
"#
    );

    run_python_script(&script)
}

/// Apply a filter through the Python engine.
///
/// Takes data and filter configuration, returns filtered data as JSON.
#[tauri::command]
fn apply_filter(data_json: String, filter_config_json: String) -> Result<String, String> {
    let script = format!(
        r#"
import json
import pandas as pd
from data_processor.core.signal_processor import SignalProcessor
data = pd.DataFrame(json.loads('{data_json}'))
config = json.loads('{filter_config_json}')
processor = SignalProcessor()
result = processor.apply_filter(data, config)
print(result.to_json(orient='records'))
"#
    );

    run_python_script(&script)
}

/// Helper: run a Python script and capture stdout.
///
/// Sets PYTHONPATH so that imports resolve without sys.path hacks.
/// Assumes the binary runs from the `src-tauri` directory (4 levels
/// below `src/`).
fn run_python_script(script: &str) -> Result<String, String> {
    let src_root = "../../../../src";
    let data_proc = "../../../../src/data_processing/data_processor/python";
    let pythonpath = format!("{}:{}", src_root, data_proc);

    let output = Command::new("python")
        .args(["-c", script])
        .env("PYTHONPATH", &pythonpath)
        .output()
        .map_err(|e| format!("Failed to spawn Python: {}", e))?;

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        Ok(stdout)
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        Err(format!("Python error: {}", stderr))
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            compute_plot,
            compute_trendline,
            apply_filter,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
