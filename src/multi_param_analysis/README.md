# Multi-Parameter Analysis Tool

A PyQt6-based GUI application for performing multi-parameter sensitivity analysis with visualization capabilities including tornado plots and correlation analysis.

## Purpose

The Multi-Parameter Analysis tool enables engineers and scientists to explore how changes in multiple input parameters affect system outputs. It provides:

- Two-dimensional parameter sweeps across configurable ranges
- Sensitivity analysis with variance-based indices
- Interactive visualization of parameter interactions
- Support for common benchmark optimization functions

## Key Features

- **Multi-Parameter Sweeps**: Configure two independent parameters with custom ranges and step counts
- **Sensitivity Analysis**: Calculate first-order sensitivity indices (S1, S2) and interaction effects
- **Variance-Based Metrics**: Decompose output variance by parameter contribution
- **Demo Functions**: Built-in test functions (Rosenbrock, Rastrigin, Sphere, Himmelblau, Beale)
- **Parallel Processing**: Optional multi-threaded execution for faster analysis
- **Results Visualization**: Grid statistics, optimal point identification, and data preview
- **Dark Theme UI**: Modern Catppuccin Mocha color scheme

## Installation

### Prerequisites

- Python 3.10 or higher
- PyQt6
- NumPy

### Install Dependencies

```bash
pip install PyQt6 numpy
```

### From Repository

```bash
cd Tools/src/multi_param_analysis
python launch_pyqt6.py
```

## Usage Instructions

### Launching the Application

```bash
python -m multi_param_analysis.ui.pyqt6.main_window
```

Or use the launcher script:

```bash
python launch_pyqt6.py
```

### Running an Analysis

1. **Configure Parameter 1 (X-Axis)**:
   - Select variable from dropdown (Temperature, O2/Feed Ratio, etc.)
   - Set minimum and maximum values
   - Choose number of steps (2-100)

2. **Configure Parameter 2 (Y-Axis)**:
   - Select a different variable
   - Define range and step count

3. **Select Output Variable**:
   - Choose the response variable to analyze (Efficiency, H2 Yield, etc.)

4. **Set Analysis Options** (Options tab):
   - Enable/disable parallel processing
   - Configure max workers
   - Toggle sensitivity calculations
   - Enable result normalization
   - Select demo function for testing

5. **Run Analysis**:
   - Click "Run Analysis" button
   - Results appear in the Results tab

## Input Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| Parameter 1 Variable | X-axis input variable | Temperature | Dropdown selection |
| Parameter 1 Min | Lower bound for parameter 1 | 600 | -1e6 to 1e6 |
| Parameter 1 Max | Upper bound for parameter 1 | 1200 | -1e6 to 1e6 |
| Parameter 1 Steps | Number of points along X | 10 | 2-100 |
| Parameter 2 Variable | Y-axis input variable | O2/Feed Ratio | Dropdown selection |
| Parameter 2 Min | Lower bound for parameter 2 | 0.1 | -1e6 to 1e6 |
| Parameter 2 Max | Upper bound for parameter 2 | 0.5 | -1e6 to 1e6 |
| Parameter 2 Steps | Number of points along Y | 10 | 2-100 |
| Max Workers | Parallel processing threads | 4 | 1-32 |

## Output Format

### Statistics Panel

- **Grid Points**: Total number of evaluation points (steps1 x steps2)
- **Min/Max Value**: Range of output values
- **Mean Value**: Average across all grid points
- **Std Deviation**: Standard deviation of results
- **Optimal X/Y**: Parameter values at minimum output

### Sensitivity Analysis

```
Variance-Based Sensitivity Indices
========================================

First-order index (S1) for Temperature:
  S1 = 0.4523 (45.2% of variance)

First-order index (S2) for O2/Feed Ratio:
  S2 = 0.3215 (32.2% of variance)

Interaction effect:
  S12 = 0.2262 (22.6% of variance)
```

### Data Preview

Displays a truncated grid of output values for visual inspection.

## Example Usage

### Basic Sensitivity Study

```python
# From command line
python launch_pyqt6.py

# Configure in GUI:
# Parameter 1: Temperature, 600-1200, 20 steps
# Parameter 2: O2/Feed Ratio, 0.1-0.5, 20 steps
# Output: Efficiency
# Demo Function: Rosenbrock
# Click "Run Analysis"
```

### Programmatic Usage

```python
import numpy as np

# Create parameter grid
param1 = np.linspace(600, 1200, 20)
param2 = np.linspace(0.1, 0.5, 20)
X, Y = np.meshgrid(param1, param2)

# Evaluate function (example: Rosenbrock)
x_norm = 10 * (X - X.min()) / (X.max() - X.min()) - 5
y_norm = 10 * (Y - Y.min()) / (Y.max() - Y.min()) - 5
Z = (1 - x_norm)**2 + 100 * (y_norm - x_norm**2)**2

# Calculate sensitivity indices
total_var = Z.var()
s1 = Z.mean(axis=0).var() / total_var  # Parameter 1 effect
s2 = Z.mean(axis=1).var() / total_var  # Parameter 2 effect
```

## Troubleshooting

### Application Won't Start

```
ModuleNotFoundError: No module named 'PyQt6'
```
**Solution**: Install PyQt6: `pip install PyQt6`

### Display Issues on High-DPI Screens

**Solution**: Set environment variable before launching:
```bash
export QT_AUTO_SCREEN_SCALE_FACTOR=1
python launch_pyqt6.py
```

### Slow Performance with Large Grids

**Solution**:
- Reduce step count for initial exploration
- Enable parallel processing in Options tab
- Increase max workers based on CPU cores

### Results Show NaN or Inf

**Solution**: Check parameter ranges - some functions have limited valid domains.

## Related Tools

- **Optimizer GUI**: For finding optimal parameter values using gradient-based methods
- **Financial Calculator**: For economic sensitivity analysis
- **Data Processor**: For preprocessing input data before analysis

## Technical Notes

### Sensitivity Index Interpretation

- **S1 near 1.0**: Output dominated by parameter 1
- **S2 near 1.0**: Output dominated by parameter 2
- **High S12**: Significant parameter interaction effects
- **S1 + S2 + S12 = 1**: Total variance decomposition

### Demo Functions

| Function | Formula | Minimum Location |
|----------|---------|------------------|
| Rosenbrock | (1-x)^2 + 100(y-x^2)^2 | (1, 1) |
| Rastrigin | 20 + x^2 + y^2 - 10(cos(2pi*x) + cos(2pi*y)) | (0, 0) |
| Sphere | x^2 + y^2 | (0, 0) |
| Himmelblau | (x^2+y-11)^2 + (x+y^2-7)^2 | Multiple |
| Beale | Sum of three quadratic terms | (3, 0.5) |

## Version History

- **1.0.0**: Initial release with PyQt6 GUI
- **1.1.0**: Added parallel processing support
- **1.2.0**: Integrated Catppuccin Mocha theme
