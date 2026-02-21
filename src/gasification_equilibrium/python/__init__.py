"""Gasification Equilibrium Calculator - Universal thermodynamic equilibrium solver.

A standalone, reusable module for computing chemical equilibrium compositions
in gasification systems via Gibbs free energy minimization.

Works as both a standalone interactive application and an importable library.

Usage as library:
    from gasification_equilibrium.python.engine import GasificationEngine
    engine = GasificationEngine()
    result = engine.solve(temperature=1200, pressure=101325,
                          feed={'C': 1.0, 'H': 1.5, 'O': 0.5})

Usage as app:
    python -m gasification_equilibrium.python.app
"""

__version__ = "1.0.0"
