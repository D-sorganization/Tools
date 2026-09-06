# Scientific Modeling Tools

The Scientific Modeling category provides simulation and modeling tools for scientific and engineering applications. These tools range from interactive visualizations to numerical solvers.

## Solar System Model

Interactive 3D visualization of the solar system with accurate orbital mechanics.

### Features

- **Accurate Positions**: Planetary positions based on JPL ephemeris data
- **Orbital Paths**: Visualize orbital trajectories
- **Time Controls**: Play, pause, speed adjustment
- **3D Camera**: Interactive viewing with pan, zoom, rotate
- **Moon Systems**: Major moons for outer planets
- **Asteroid Belt**: Representative asteroid distribution
- **Comet Tracking**: Major comet orbits

### Controls

| Input            | Action                |
| ---------------- | --------------------- |
| Left-click drag  | Rotate view           |
| Right-click drag | Pan view              |
| Scroll wheel     | Zoom in/out           |
| Click planet     | Center and show info  |
| Space bar        | Play/pause simulation |
| +/- keys         | Adjust time speed     |

### Time Controls

| Speed  | Simulation Rate    |
| ------ | ------------------ |
| 1x     | Real-time          |
| 60x    | 1 minute = 1 hour  |
| 1440x  | 1 minute = 1 day   |
| 10080x | 1 minute = 1 week  |
| 43200x | 1 minute = 1 month |

### Planet Information

Click any planet to view:

- Orbital parameters (semi-major axis, eccentricity)
- Physical properties (mass, radius)
- Current distance from Sun
- Orbital period
- Known moons

---

## RRT Path Planner

Rapidly-exploring Random Trees path planning for robotics applications.

### Algorithm Overview

RRT (Rapidly-exploring Random Tree) is a sampling-based motion planning algorithm:

1. Start with initial position
2. Sample random point in space
3. Extend tree toward sample
4. Check for collisions
5. Repeat until goal reached

### Features

- **3D Environment**: Navigate around obstacles
- **Dual Implementation**: MATLAB and Python versions
- **Star Wars Theme**: Fun visualization with TIE fighters
- **AI Pursuit**: Dynamic replanning for moving targets
- **Cinematic Camera**: Multiple viewing angles

### Parameters

| Parameter      | Description                       | Typical Range |
| -------------- | --------------------------------- | ------------- |
| Start Position | Initial robot position            | Within bounds |
| Goal Position  | Target position                   | Within bounds |
| Step Size      | Maximum extension distance        | 0.1 - 2.0     |
| Max Iterations | Planning limit                    | 1000 - 50000  |
| Goal Bias      | Probability of sampling goal      | 0.05 - 0.20   |
| Goal Tolerance | Distance to consider goal reached | 0.1 - 1.0     |

### Obstacle Types

- Spheres
- Boxes
- Cylinders
- Custom meshes

### Outputs

- Path waypoints (x, y, z coordinates)
- Path length
- Planning time
- Tree visualization

---

## ODE Solver

Solve systems of ordinary differential equations with interactive GUI.

### Preset Examples

#### Exponential Decay

```
dy/dt = -k*y
```

Models: Radioactive decay, cooling, chemical reactions

#### Harmonic Oscillator

```
d2x/dt2 = -omega^2 * x
```

Models: Springs, pendulums, LC circuits

#### Lotka-Volterra (Predator-Prey)

```
dx/dt = alpha*x - beta*x*y
dy/dt = delta*x*y - gamma*y
```

Models: Population dynamics, ecology

#### Van der Pol Oscillator

```
d2x/dt2 - mu*(1-x^2)*dx/dt + x = 0
```

Models: Nonlinear oscillations, limit cycles

### Custom ODE Input

Enter ODEs in Python syntax:

```python
def derivatives(t, y):
    # y[0], y[1], ... are state variables
    dydt = np.zeros(len(y))
    dydt[0] = y[1]           # dx/dt = v
    dydt[1] = -9.81          # dv/dt = -g
    return dydt
```

### Solver Methods

| Method | Type                 | Best For             |
| ------ | -------------------- | -------------------- |
| RK45   | Explicit, adaptive   | General problems     |
| RK23   | Explicit, adaptive   | Less accuracy needed |
| DOP853 | Explicit, high-order | High accuracy        |
| Radau  | Implicit             | Stiff systems        |
| BDF    | Implicit, multi-step | Stiff systems        |
| LSODA  | Auto-switching       | Unknown stiffness    |

### Stiffness

A system is "stiff" when:

- Contains widely different time scales
- Explicit methods require tiny time steps
- Implicit methods are more efficient

**Indicators of stiffness**:

- Eigenvalues with large negative real parts
- Ratio of largest to smallest time constant > 1000

### Outputs

- Solution curves vs. time
- Phase portraits
- Numerical data export
- Stability analysis (for linear systems)

---

## Thermal Profile Predictor

Predict temperature profiles in heated vessels over time.

### Model

Uses lumped capacitance thermal model:

```
m*Cp * dT/dt = Q_in - h*A*(T - T_ambient)
```

Where:

- m\*Cp = Thermal mass (J/K)
- Q_in = Heat input (W)
- h\*A = Heat loss coefficient (W/K)
- T = Temperature (K)
- T_ambient = Ambient temperature (K)

### Inputs

| Parameter       | Description                  | Units |
| --------------- | ---------------------------- | ----- |
| Thermal Mass    | Energy storage capacity      | J/K   |
| Heat Loss Coeff | Convection/conduction losses | W/K   |
| Initial Temp    | Starting temperature         | C     |
| Ambient Temp    | Environment temperature      | C     |
| Power Profile   | Heating power vs time        | W     |

### Power Profiles

| Profile     | Description              |
| ----------- | ------------------------ |
| Constant    | Fixed power level        |
| Linear Ramp | Power increases linearly |
| Step        | Sudden power changes     |
| Custom      | User-defined profile     |

### Outputs

- Temperature vs. time curve
- Steady-state temperature
- Time to reach target temperature
- Condensation risk (if dewpoint provided)
- Energy consumption

---

## Multi-Parameter Analysis

Sensitivity analysis across multiple parameter dimensions.

### Demo Functions

| Function   | Description       | Optimum  |
| ---------- | ----------------- | -------- |
| Rosenbrock | Curved valley     | (1, 1)   |
| Rastrigin  | Many local minima | (0, 0)   |
| Sphere     | Simple convex     | (0, 0)   |
| Himmelblau | Four equal minima | Multiple |

### Analysis Methods

#### Grid Search

Evaluate function on regular grid of parameter values.

**Pros**: Complete coverage, easy to implement
**Cons**: Computationally expensive for many parameters

#### Monte Carlo

Random sampling of parameter space.

**Pros**: Scales better with dimensions
**Cons**: May miss local features

#### Sobol Sensitivity

Variance-based global sensitivity analysis.

**Outputs**:

- First-order indices (main effects)
- Total-order indices (including interactions)

### Visualization

- Contour plots (2D)
- Surface plots (3D)
- Sensitivity bar charts
- Parameter correlation plots

---

## Optimizer GUI (legacy shim)

`src/optimizer_gui` is now a compatibility launcher only. The standalone
PyQt6 optimizer GUI that used to live there was consolidated into the
Movement Optimizer application (`src/movement_optimizer`), and the drifted
vendored copy of its models was removed (Tools #3983). Launching
`python src/optimizer_gui/launch_pyqt6.py` opens the canonical Movement
Optimizer application.

---

## Tips for Scientific Modeling

### ODE Solving

1. **Start simple**: Test with known analytic solutions
2. **Check conservation**: Verify energy/mass conservation if applicable
3. **Adjust tolerances**: Balance accuracy vs. computation time
4. **Handle stiffness**: Use implicit solvers if explicit fails

### Optimization

1. **Scale parameters**: Normalize to similar ranges
2. **Set bounds**: Constrain to physically meaningful values
3. **Multiple starts**: Run from different initial guesses
4. **Monitor convergence**: Check if optimizer is stuck

### Visualization

1. **Use appropriate scales**: Log scale for wide ranges
2. **Label everything**: Units, parameters, conditions
3. **Export data**: Save numerical results, not just plots

---

## Common Equations Reference

### Exponential Growth/Decay

```
y(t) = y0 * exp(k*t)
```

### Simple Harmonic Motion

```
x(t) = A * cos(omega*t + phi)
omega = sqrt(k/m)
```

### Heat Transfer

```
Q = m*Cp*(T2 - T1)
Q = h*A*(T_surface - T_fluid)
```

### First-Order Kinetics

```
dC/dt = -k*C
C(t) = C0 * exp(-k*t)
```

---

For detailed documentation, see the [User Manual](../USER_MANUAL.md).
