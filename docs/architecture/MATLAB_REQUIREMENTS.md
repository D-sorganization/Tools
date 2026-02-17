# MATLAB Requirements

This repository contains tools that integrate with MATLAB. To use these features, the following requirements must be met.

## Version Requirements

- **MATLAB Version:** R2020a or later is required.
- **Python Integration:** Ensure your Python version is compatible with your MATLAB version (check [MathWorks Python Compatibility](https://www.mathworks.com/support/requirements/python-compatibility.html)).

## Required Toolboxes

The following toolboxes are typically required for the MATLAB tools in this repository:

- **Symbolic Math Toolbox** (for symbolic calculations)
- **Control System Toolbox** (for control theory analysis)
- **Signal Processing Toolbox** (for signal analysis)
- **Simulink** (if running Simulink models)

## Setup

1. **Install MATLAB Engine API for Python:**
   Run the following commands in your terminal (adjust path to your MATLAB installation):

   ```bash
   cd "C:\Program Files\MATLAB\R202Xy\extern\engines\python"
   python setup.py install
   ```

   Or on Linux/macOS:

   ```bash
   cd /usr/local/MATLAB/R202Xy/extern/engines/python
   python setup.py install
   ```

2. **Verify Installation:**
   Run the following Python code:
   ```python
   import matlab.engine
   eng = matlab.engine.start_matlab()
   eng.quit()
   ```

## Fallback Mechanism

If MATLAB is not installed or the engine API is unavailable, the tools will attempt to use fallback Python implementations (e.g., using `numpy`, `scipy`, or `sympy`) where possible. However, some specialized features may be disabled.

## Troubleshooting

- **"No module named matlab":** The MATLAB Engine API is not installed in your current Python environment.
- **"MATLAB execution error":** Check your license and path configuration.
