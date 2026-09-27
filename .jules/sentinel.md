## $(date +%Y-%m-%d) - Prevent MATLAB code injection in matlab_quality_utils
**Vulnerability:** User-controlled file paths containing single quotes could break out of MATLAB's string literal parsing in `_run_matlab_script` when passed via `subprocess.run`, allowing arbitrary MATLAB command execution.
**Learning:** Even when `subprocess.run` uses `shell=False`, arguments passed to an interpreter (like `matlab -batch`) are still subject to that interpreter's own syntax and escaping rules.
**Prevention:** Explicitly validate input paths to reject dangerous characters (like single quotes) before they are interpolated into command strings for execution by another interpreter.
