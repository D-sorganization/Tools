# Double Pendulum Golf Swing Simulator

A PyQt6-based visualization tool for exploring the dynamics of a **driven double pendulum** 
modeled as a simplified golf swing. Uses **Lagrangian mechanics with relative (generalized) 
coordinates** to show how off-diagonal mass matrix terms enable passive energy transfer in 
kinematic chains.

## Key Concepts

This simulator demonstrates a fundamental insight about multi-body dynamics:

- **Diagonal mass matrix terms** (M11, M22) represent each joint's "self-inertia" — the 
  resistance of each segment to its own angular acceleration.
- **Off-diagonal terms** (M12 = M21) represent **cross-coupling** — how torque at one joint 
  accelerates the other segment through the physical linkage.

In a golf swing, the off-diagonal coupling is what allows the arm's rotation to drive the 
club's acceleration **without requiring wrist torque**. The coupling is maximized when the 
segments are aligned (phi ≈ 0), which is exactly the configuration at impact.

## Coordinates

```
    Shoulder (fixed pivot)
        |
        | segment 1 (arm): angle θ₁ from vertical
        |
    Wrist (joint)
        |
        | segment 2 (club): angle φ relative to arm
        |
    Club tip
```

- `θ₁ = 0, φ = 0` → both segments hanging straight down (equilibrium)
- Positive angles = counterclockwise

## Installation

```bash
# Clone or copy to your repo, then:
cd double_pendulum_golf
pip install -e ".[dev]"
```

## Usage

```bash
# Run the GUI
python -m double_pendulum_golf

# Or use the console script
pendulum-golf
```

### GUI Layout

| Panel | Description |
|-------|-------------|
| **Left** | Parameter inputs, initial conditions, torque polynomials, presets |
| **Center** | Animated pendulum with tip trail |
| **Right** | Real-time mass matrix display, force balance, energy |

### Torque Polynomials

Torques are specified as polynomial coefficients: `c0, c1, c2, ...`

This evaluates as: `τ(t) = c0 + c1·t + c2·t² + ...`

Example: `-25, 10` gives `τ(t) = -25 + 10t` (strong initial torque that decreases)

### Presets

- **Golf Swing (passive wrist)**: Shoulder-driven swing with zero wrist torque — 
  demonstrates passive release through coupling
- **Golf Swing (active wrist)**: Adds small wrist torque for comparison
- **Free Double Pendulum**: No torques, chaotic dynamics
- **Straight Drop**: Near-vertical release

## Running Tests

```bash
pytest
pytest --cov=double_pendulum_golf --cov-report=term-missing
```

## Architecture

```
src/double_pendulum_golf/
├── physics.py          # Core EOM: mass matrix, Coriolis, gravity
├── simulation.py       # Integration engine, polynomial torques
├── gui/
│   ├── main_window.py      # Top-level orchestration
│   ├── pendulum_widget.py  # Animation canvas
│   ├── matrix_widget.py    # Real-time matrix display
│   └── controls_widget.py  # Input panel with presets
```

### Design Principles

- **DRY**: Common patterns extracted (LabeledInput widget, shared coordinate transforms)
- **DbC** (Design by Contract): Preconditions/postconditions via assertions throughout physics code
- **TDD**: Comprehensive test suite covering mass matrix properties, energy conservation, 
  known analytical values, and contract violations

## Physics Reference

### Mass Matrix (point masses at tips)

```
M11 = (m1 + m2)·L1² + m2·L2² + 2·m2·L1·L2·cos(φ)
M12 = M21 = m2·L2² + m2·L1·L2·cos(φ)
M22 = m2·L2²
```

### Equations of Motion

```
M(q)·q̈ = τ - C(q,q̇)·q̇ - G(q)
```

where C contains Coriolis/centrifugal terms and G contains gravitational torques.

## License

MIT
