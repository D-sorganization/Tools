# Optimizer GUI

A PyQt6-based graphical interface for configuring and running optimization algorithms using scipy.optimize with support for multiple algorithms, bounds, constraints, and real-time convergence monitoring.

## Purpose

The Optimizer GUI provides an accessible interface for solving optimization problems without writing code. It supports:

- Configurable optimization parameters with bounds
- Multiple optimization algorithms (Adam, L-BFGS-B, Differential Evolution)
- Real-time optimization history tracking
- Gradient-based and gradient-free methods

## Key Features

- **Multiple Algorithms**: Adam optimizer, L-BFGS-B, Grid Search, Differential Evolution
- **Parameter Management**: Add/remove parameters dynamically with bounds
- **Adam Hyperparameters**: Full control over learning rate, beta1, beta2, epsilon
- **Convergence Settings**: Configurable tolerance, max iterations, gradient step size
- **Bounds Support**: Define min/max constraints for each parameter
- **History Tracking**: View optimization progress iteration by iteration
- **Dark Theme UI**: Modern Catppuccin Mocha styling

## Installation

### Prerequisites

- Python 3.10 or higher
- PyQt6
- NumPy
- SciPy (for advanced algorithms)

### Install Dependencies

```bash
pip install PyQt6 numpy scipy
```

### From Repository

```bash
cd Tools/src/optimizer_gui
python launch_pyqt6.py
```

## Usage Instructions

### Launching the Application

```bash
python -m optimizer_gui.ui.pyqt6.main_window
```

Or use the launcher:

```bash
python launch_pyqt6.py
```

### Setting Up an Optimization

1. **Define Parameters** (Parameters tab):
   - View/edit default parameters in the table
   - Click "Add Parameter" to add new variables
   - Set Name, Initial value, Min, and Max for each
   - Remove unwanted parameters with "Remove Selected"

2. **Configure Optimization Goal**:
   - Check "Maximize" for maximization problems
   - Uncheck for minimization (default)

3. **Set Algorithm Settings** (Adam Settings tab):
   - Learning Rate: Step size (default: 0.01)
   - Beta1: Momentum coefficient (default: 0.9)
   - Beta2: RMSprop coefficient (default: 0.999)
   - Epsilon: Numerical stability (default: 1e-8)

4. **Configure Convergence**:
   - Max Iterations: Upper limit on iterations
   - Tolerance: Convergence threshold
   - Gradient Step: Finite difference step size

5. **Select Method**:
   - Adam: Gradient-based with momentum
   - Grid Search: Exhaustive search
   - L-BFGS-B: Quasi-Newton method
   - Differential Evolution: Evolutionary algorithm

6. **Run Optimization**:
   - Click "Run Optimization"
   - Results appear in Results tab

## Input Parameters

### Optimization Parameters Table

| Column | Description | Example |
|--------|-------------|---------|
| Name | Variable identifier | Temperature |
| Initial | Starting value | 800 |
| Min | Lower bound | 600 |
| Max | Upper bound | 1200 |

### Adam Hyperparameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| Learning Rate | Step size per iteration | 0.01 | 0.0001-1.0 |
| Beta1 | First moment decay rate | 0.9 | 0.0-0.999 |
| Beta2 | Second moment decay rate | 0.999 | 0.0-0.9999 |
| Epsilon | Division stability constant | 1e-8 | 1e-10-1e-4 |

### Convergence Settings

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| Max Iterations | Maximum optimization steps | 100 | 1-10000 |
| Tolerance | Convergence threshold | 1e-6 | 1e-10-1.0 |
| Gradient Step | Numerical gradient delta | 0.001 | 1e-8-1.0 |

## Output Format

### Best Result Panel

```
Best Objective: -0.000023
Iterations: 87
Converged: Yes
```

### Best Parameters Panel

```
Temperature: 800.2341
O2/Feed Ratio: 0.2998
Steam/Feed Ratio: 0.5012
Pressure: 1.0001
```

### Optimization History

```
Iteration | Objective  | Parameters
------------------------------------------------------------
   1      | 1250.4523  | Temperature=800.0000, O2/Feed Ratio=0.3000
   2      | 1189.2341  | Temperature=801.2500, O2/Feed Ratio=0.2987
  ...
  87      |   0.0000   | Temperature=800.2341, O2/Feed Ratio=0.2998
```

## Example Usage

### Basic Optimization

```bash
# Launch GUI
python launch_pyqt6.py

# In GUI:
# 1. Keep default parameters
# 2. Set Method: Adam
# 3. Click "Run Optimization"
```

### Multi-Start Optimization

```python
# Programmatic approach with scipy
from scipy.optimize import minimize

def objective(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

bounds = [(600, 1200), (0.1, 0.5)]
x0 = [800, 0.3]

result = minimize(
    objective,
    x0,
    method='L-BFGS-B',
    bounds=bounds,
    options={'maxiter': 100}
)
print(f"Optimal: {result.x}, Value: {result.fun}")
```

### Custom Objective Function

The demo uses the Rosenbrock function:

```python
# f(x, y) = (1-x)^2 + 100*(y-x^2)^2
# Minimum at (1, 1) with f(1, 1) = 0
```

## Algorithm Selection Guide

| Algorithm | Best For | Pros | Cons |
|-----------|----------|------|------|
| Adam | Smooth objectives | Fast, momentum | Requires gradient |
| L-BFGS-B | Bounded problems | Memory efficient | Local optima |
| Grid Search | Small spaces | Finds global | Slow for many params |
| Differential Evolution | Noisy objectives | Global search | Many evaluations |

## Troubleshooting

### Optimization Not Converging

**Symptoms**: Iterations reach max without improvement

**Solutions**:
- Increase max iterations
- Reduce learning rate
- Widen parameter bounds
- Try different algorithm (Differential Evolution for global)

### Slow Convergence

**Solutions**:
- Increase learning rate (carefully)
- Reduce tolerance for faster termination
- Use Grid Search for initial estimate, then refine

### Parameters Stuck at Bounds

**Solutions**:
- Widen bounds if physically reasonable
- Check if minimum is outside search space
- Verify objective function behavior at bounds

### Numerical Instabilities

**Symptoms**: NaN or Inf in results

**Solutions**:
- Reduce learning rate
- Increase epsilon value
- Check objective function for singularities
- Use larger gradient step size

### Memory Issues with Grid Search

**Solutions**:
- Reduce grid resolution
- Use smaller parameter ranges
- Switch to Adam or L-BFGS-B

## Related Tools

- **Multi-Parameter Analysis**: For sensitivity analysis across parameter space
- **Financial Calculator**: For economic objective functions
- **Data Processor**: For preparing input data

## Technical Notes

### Adam Algorithm

The Adam optimizer combines momentum and RMSprop:

```
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)
theta = theta - lr * m_hat / (sqrt(v_hat) + epsilon)
```

### Gradient Computation

Numerical gradients use central differences:

```
df/dx ≈ (f(x+h) - f(x-h)) / (2h)
```

### Constraint Handling

Bounds are enforced via projection:
```python
values = np.clip(values, lower_bounds, upper_bounds)
```

## Version History

- **1.0.0**: Initial release with Adam optimizer
- **1.1.0**: Added L-BFGS-B and Differential Evolution
- **1.2.0**: Integrated Catppuccin Mocha theme
- **1.3.0**: Enhanced convergence monitoring
