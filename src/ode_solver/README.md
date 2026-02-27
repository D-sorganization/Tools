# ODE Solver

A numerical ordinary differential equation (ODE) solver supporting multiple integration methods including RK4, RK45, and BDF for stiff equations. This tool provides both a graphical interface and programmatic API for solving systems of differential equations.

## Purpose

The ODE Solver enables engineers and scientists to:

- Solve first-order ODE systems with symbolic equation input
- Handle stiff differential equations using implicit methods
- Visualize solution trajectories over time
- Model dynamic chemical, thermal, and mechanical systems
- Perform parameter sensitivity studies

## Key Features

- **Multiple Integration Methods**: RK4, RK45 (adaptive), BDF (stiff systems)
- **Symbolic Equation Input**: Define derivatives using mathematical expressions
- **Preset Examples**: Common ODE systems ready to solve
- **Parameter Support**: Define and modify model parameters
- **Adaptive Step Size**: Automatic error control with RK45
- **Results Visualization**: Tabular output with solution summary
- **Catppuccin Mocha Theme**: Modern dark interface design

## Installation / Prerequisites

### Required Dependencies

```bash
pip install PyQt6 numpy scipy
```

### Optional Dependencies

```bash
# For symbolic parsing (enhanced equation support)
pip install sympy
```

### Python Version

- Python 3.10 or higher recommended

## Usage Instructions

### Launch Desktop Application

```bash
python launch_pyqt6.py
```

### Programmatic Usage

```python
from upstream_drift_tools.process_calculators.ode_solver import ODESolver
import numpy as np

# Define ODE system
derivatives = {
    "y": "-k*y"
}
parameters = {"k": 0.1}

# Create solver and solve
solver = ODESolver(derivatives, parameters)
t_span = (0, 50)
y0 = [100]
t_eval = np.linspace(0, 50, 100)

solution = solver.solve(t_span, y0, t_eval=t_eval)
print(f"Final value: y({t_span[1]}) = {solution.y[0][-1]:.4f}")
```

## Input Parameters

### Derivatives Format

Define derivatives as `variable: expression` pairs:

```
y: -k*y
T: k*(T_env - T)
x: v
v: -omega**2*x
```

### Parameters Format

Define parameters as `name: value` pairs:

```
k: 0.1
omega: 1.0
T_env: 350
```

### Initial Conditions Format

Define initial values as `variable: value` pairs:

```
y: 100
x: 1
v: 0
```

### Time Parameters

| Parameter     | Description             | Range       | Default |
| ------------- | ----------------------- | ----------- | ------- |
| Start Time    | Integration start time  | 0 - 10^6    | 0       |
| End Time      | Integration end time    | 0.1 - 10^6  | 20      |
| Output Points | Number of output points | 10 - 10,000 | 100     |

### Preset Examples

| Preset              | Equations                                      | Description             |
| ------------------- | ---------------------------------------------- | ----------------------- |
| Exponential Decay   | dy/dt = -k\*y                                  | First-order decay       |
| Heating/Cooling     | dT/dt = k\*(T_env - T)                         | Newton's law of cooling |
| Harmonic Oscillator | dx/dt = v, dv/dt = -omega^2\*x                 | Simple harmonic motion  |
| Damped Oscillator   | dx/dt = v, dv/dt = -2*zeta*omega*v - omega^2*x | Damped oscillation      |
| Lotka-Volterra      | dx/dt = a*x - b*x*y, dy/dt = -c*y + d*x*y      | Predator-prey model     |

## Output Format

### Solution Object

The solver returns a `scipy.integrate.OdeSolution` object with:

| Attribute  | Description                                    |
| ---------- | ---------------------------------------------- |
| `t`        | Array of time points                           |
| `y`        | 2D array of solution values (variables x time) |
| `t_events` | Event times (if events defined)                |
| `y_events` | Solution at event times                        |

### Results Display

The GUI displays:

- System definition with equations
- Parameter values
- Initial conditions
- Time range and number of points
- Final values for each variable
- Min/max summary for each variable
- Tabular data for sample points

## Mathematical Models

### Runge-Kutta 4th Order (RK4)

Classic fixed-step explicit method:

```
k1 = h * f(t_n, y_n)
k2 = h * f(t_n + h/2, y_n + k1/2)
k3 = h * f(t_n + h/2, y_n + k2/2)
k4 = h * f(t_n + h, y_n + k3)

y_{n+1} = y_n + (k1 + 2*k2 + 2*k3 + k4) / 6
```

- Order: 4
- Error: O(h^5) per step, O(h^4) global
- Suitable for: Non-stiff problems with smooth solutions

### Runge-Kutta-Fehlberg (RK45)

Adaptive step size method using embedded error estimation:

```
y_{n+1} = y_n + sum(b_i * k_i)     (5th order)
y*_{n+1} = y_n + sum(b*_i * k_i)   (4th order)

error = |y_{n+1} - y*_{n+1}|
```

Step size adjustment:

```
h_new = h * (tol / error)^(1/5)
```

- Order: 4(5)
- Adaptive: Yes
- Suitable for: General purpose, moderate accuracy

### Backward Differentiation Formula (BDF)

Implicit multistep method for stiff equations:

```
sum(alpha_k * y_{n-k}) = h * beta * f(t_{n+1}, y_{n+1})
```

For BDF-2:

```
3*y_{n+1} - 4*y_n + y_{n-1} = 2*h*f(t_{n+1}, y_{n+1})
```

- Order: 1-5 (variable)
- Implicit: Yes (requires Newton iteration)
- Suitable for: Stiff systems, chemical kinetics

### Stiffness Detection

A system is stiff when:

```
|lambda_max / lambda_min| >> 1
```

Where lambda are eigenvalues of the Jacobian. Symptoms include:

- Explicit methods require very small time steps
- Solution appears stable but computation is slow

## Example Usage

### Chemical Kinetics

```python
# First-order reaction: A -> B
# dC_A/dt = -k * C_A

derivatives = {"C_A": "-k*C_A"}
parameters = {"k": 0.05}  # 1/min
solver = ODESolver(derivatives, parameters)

solution = solver.solve(
    t_span=(0, 60),
    y0=[100],  # Initial concentration
    t_eval=np.linspace(0, 60, 61)
)

half_life = np.log(2) / 0.05
print(f"Half-life: {half_life:.1f} min")
```

### Batch Reactor Temperature

```python
# Exothermic batch reactor
# dT/dt = (Q_gen - Q_loss) / (m * Cp)

derivatives = {
    "T": "(r*delta_H - U*A*(T - T_cool)) / (m*Cp)",
    "C": "-k*C"
}
parameters = {
    "r": 1.0,
    "delta_H": -50000,
    "U": 500,
    "A": 2,
    "T_cool": 300,
    "m": 1000,
    "Cp": 4000,
    "k": 0.1
}

solver = ODESolver(derivatives, parameters)
solution = solver.solve((0, 100), [300, 10])
```

### Van der Pol Oscillator (Stiff)

```python
# mu >> 1 creates stiff system
derivatives = {
    "x": "y",
    "y": "mu*(1 - x**2)*y - x"
}
parameters = {"mu": 1000}  # Very stiff

solver = ODESolver(derivatives, parameters, method="BDF")
solution = solver.solve((0, 3000), [2, 0])
```

### Pendulum Motion

```python
derivatives = {
    "theta": "omega",
    "omega": "-(g/L)*sin(theta)"
}
parameters = {"g": 9.81, "L": 1.0}

solver = ODESolver(derivatives, parameters)
solution = solver.solve((0, 10), [0.5, 0])  # 0.5 rad initial angle
```

## Troubleshooting

| Issue                       | Cause                             | Solution                                   |
| --------------------------- | --------------------------------- | ------------------------------------------ |
| "No derivatives defined"    | Empty derivatives field           | Enter at least one derivative equation     |
| "Missing initial condition" | Variable without initial value    | Add initial condition for all variables    |
| "Integration step failed"   | Stiff system with explicit method | Switch to BDF method                       |
| Solution oscillates wildly  | Time step too large               | Increase output points                     |
| Very slow computation       | Stiff system detection            | Use BDF for chemical kinetics              |
| "Invalid expression"        | Syntax error in equation          | Check operator syntax (use \*\* for power) |

### Common Expression Errors

| Error   | Correction                             |
| ------- | -------------------------------------- |
| `x^2`   | Use `x**2` for exponentiation          |
| `sin x` | Use `sin(x)` with parentheses          |
| `2x`    | Use `2*x` with explicit multiplication |
| `e^x`   | Use `exp(x)` for exponential           |

### Stiff System Indicators

- Solution changes on vastly different time scales
- Chemical reactions with fast/slow species
- Heat transfer with large thermal gradients
- Electronic circuits with disparate RC time constants

## Related Tools

- **Thermal Profile Predictor**: Uses ODE solver for temperature predictions
- **PSA Package**: Dynamic adsorption modeling
- **Steam Engine Calculator**: Thermodynamic property inputs for energy balances
- **Flow Rate Converter**: Flow rate unit handling for material balances

## References

### Numerical Methods

| Method | Reference                                |
| ------ | ---------------------------------------- |
| RK4    | Classical 4th-order Runge-Kutta          |
| RK45   | Dormand-Prince 4(5) pair (scipy default) |
| BDF    | Backward Differentiation Formula (LSODA) |
| Radau  | Implicit Runge-Kutta for stiff problems  |

### Stability Regions

| Method | Stability   | Best For                   |
| ------ | ----------- | -------------------------- |
| RK4    | Conditional | Smooth, non-stiff problems |
| RK45   | Conditional | General purpose            |
| BDF    | A-stable    | Stiff systems              |
| Radau  | L-stable    | Very stiff systems         |

### Accuracy Guidelines

| Tolerance | Typical Use              |
| --------- | ------------------------ |
| 1e-3      | Quick estimates          |
| 1e-6      | Engineering calculations |
| 1e-9      | Scientific accuracy      |
| 1e-12     | High-precision work      |

## Current Features

- Purpose: Solve systems of ordinary differential equations symbolically
- Category: Mathematics
- Python files in tool path: 9
- Surface support: PyQt6=implemented, Web manifest=no, Web implementation=present
- Test visibility: 2 name-matched test files under tests/

## Implementation State

- PyQt6 launcher: Implemented
- Web surface declared in manifest: No
- Web surface implementation: Implemented
- README last reviewed: 2026-02-27

## Implementation Gaps

- No structural gaps detected from manifest/surface scan.
