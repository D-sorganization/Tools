"""
Generic polynomial torque builder (DRY — single source of truth).

Replaces the per-module ``make_polynomial_torque`` functions in
simulation.py, simulation_triple.py, and simulation_golfer.py.

Usage::

    from .torque_utils import make_polynomial_torque

    # Double pendulum (2-joint)
    tf = make_polynomial_torque([5.0], [0.0])

    # Triple pendulum (3-joint)
    tf = make_polynomial_torque([5.0], [0.0], [1.0])

    # Golfer (7-joint)
    tf = make_polynomial_torque([1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0])
"""

from __future__ import annotations

from collections.abc import Callable

from shared.python.swing_sim.torque_profiles import TorquePolynomial


def make_polynomial_torque(
    *coeffs_per_joint: list[float],
) -> Callable[[float], tuple[float, ...]]:
    """Create a torque function from polynomial coefficients for N joints.

    Each argument is a list of polynomial coefficients for one joint:
        tau_i(t) = c0 + c1*t + c2*t^2 + ...

    Preconditions:
        - At least one joint must be specified.
        - Each coefficient list must have >= 1 element.

    Postconditions:
        - Returned function produces N finite values for finite t.

    Parameters
    ----------
    *coeffs_per_joint : list[float]
        Variable number of coefficient lists, one per joint.

    Returns
    -------
    Callable[[float], tuple[float, ...]]
        A function that takes time t and returns a tuple of torques.
    """
    if not (len(coeffs_per_joint) >= 1):
        raise ValueError("Need at least one joint")

    polynomials: list[TorquePolynomial] = []
    for i, coeffs in enumerate(coeffs_per_joint):
        if not (len(coeffs) >= 1):
            raise ValueError(f"Need at least one coefficient for joint {i}, got {len(coeffs)}")
        polynomials.append(TorquePolynomial(tuple(coeffs)))

    def torque_func(t: float) -> tuple[float, ...]:
        return tuple(polynomial.evaluate(t) for polynomial in polynomials)

    return torque_func
