# Gasification Equilibrium Calculator

A universal thermodynamic equilibrium calculator for gasification systems.
Works in **Python**, **MATLAB**, and **GNU Octave**.

## Quick Start

### Python
```bash
# Interactive GUI
python3 run_app.py

# Headless (no GUI)
python3 run_app.py --headless

# Run tests
python3 -m pytest tests/ -v
```

### As a Python Module
```python
from gasification_equilibrium.python.engine import GasificationEngine

engine = GasificationEngine()
result = engine.solve(temperature=1200, pressure=101325,
                      feed={'C': 1.0, 'H': 2.0, 'O': 1.0})
print(result.composition_dict())
print(f"H2/CO = {result.h2_co_ratio:.3f}")
```

### MATLAB / Octave
```matlab
cd matlab/
result = gasification_equilibrium(1200, 101325, struct('C',1,'H',2,'O',1));
disp(result.mole_frac)

% With process injections
result = gasification_equilibrium(1200, 101325, struct('C',1,'O',0.5), ...
    'steam_carbon', 1.0, 'o2_flow', 0.3, 'use_air', true);

% Run tests
test_gasification()
```

## Features

| Feature | Description |
|---------|-------------|
| **Single Point** | Equilibrium at any T, P with instant bar/pie charts |
| **Temperature Sweep** | Composition vs temperature with H2/CO, CGE metrics |
| **Surface Plots** | 3D surfaces over T x (steam/carbon, O2/carbon, pressure) |
| **Feed Editor** | 8 presets (coals, biomass, petcoke, MSW, gas) + custom CHONS |
| **Process Inputs** | Steam, O2/air, N2 purge, CH4, C3H8, natural gas injection |

## Architecture (SOLID)

```
gasification_equilibrium/
├── python/
│   ├── thermo_data.py    # NASA polynomial database (15 species incl. C3H8, Ar)
│   ├── feed.py           # Feed composition builder (SRP, OCP)
│   ├── solver.py         # Gibbs minimization solver (SRP)
│   ├── engine.py         # Thin orchestrator (DIP, OCP)
│   ├── metrics.py        # Post-solve metrics (SRP, pure functions)
│   ├── sweeps.py         # Temperature/surface sweep strategies (SRP)
│   ├── plots.py          # Stateless plot rendering (SRP)
│   ├── theme.py          # Visual constants (SRP)
│   ├── app.py            # Interactive 4-tab matplotlib GUI
│   └── __init__.py
├── matlab/
│   ├── thermo_data.m     # NASA data (14 species, MATLAB/Octave)
│   ├── gibbs_dimensionless.m
│   ├── gasification_equilibrium.m  # Solver (fmincon/sqp) with injections
│   ├── temperature_sweep.m
│   ├── surface_sweep.m
│   ├── gasification_app.m          # Interactive GUI
│   └── test_gasification.m
├── tests/
│   ├── test_thermo_data.py   # 42 tests - DB integrity, NASA polynomials
│   ├── test_feed.py          # 52 tests - feed composition, injections
│   ├── test_solver.py        # 36 tests - Gibbs minimizer, known equilibria
│   ├── test_engine.py        # 38 tests - orchestrator, feed modes, sweeps
│   ├── test_metrics.py       # 23 tests - mole fractions, H2/CO, CGE
│   └── test_sweeps.py        # 13 tests - sweep strategies
├── run_app.py            # Universal launcher
└── README.md
```

## Thermodynamic Model

- **Method**: Gibbs free energy minimization (SLSQP optimization)
- **Data**: NASA 7-coefficient polynomials (200-3500 K)
- **Species**: H2, CO, CO2, H2O, CH4, N2, O2, C2H4, C2H6, H2S, NH3, SO2, C3H8, Ar, C(s)
- **Constraints**: Element balance conservation (C, H, O, N, S)
- **Phases**: Ideal gas mixture + pure solid (graphite)
- **Initial guess**: NNLS (non-negative least squares) with uniform blend

## Design Principles

- **SOLID**: SRP (9 focused modules), OCP (new injections via composition), DIP (engine delegates)
- **TDD**: 204 tests covering thermodynamics, solver, feed, metrics, sweeps, engine, contracts
- **DbC**: Precondition/postcondition assertions on all public functions
- **DRY**: Single thermodynamic database shared across all calculations
- **Orthogonality**: Engine is independent of UI; works as standalone library
