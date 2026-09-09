"""Nonlinear inertial shaft acceleration and work from prescribed moving grips."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._grip_contracts import _finite_norm, finite_array
from ._grip_finite_response import FiniteGripResponse, finite_grip_response
from ._grip_moving_kinematics import MaterialPointMotion, moving_grip_kinematics
from ._shaft_chain import ChainLinearization
from ._shaft_equilibrium import _check_strains
from ._shaft_inertia import SectionKinetics
from ._shaft_moving_contracts import (
    InertialMovingChain,
    MovingChainControls,
    MovingChainState,
)
from ._shaft_moving_contracts import (
    MovingGripAttachment as MovingGripAttachment,
)
from ._shaft_spectrum import _validated_mass


@dataclass(frozen=True)
class MovingChainResponse:
    """Owned body-twist derivatives and an inertial work/energy snapshot.

    Twist derivatives are not physical linear accelerations in material axes;
    those also include omega cross v. Total energy includes shaft kinetic,
    section elastic and relative grip inertance/elastic storage. Force-only
    potential is excluded; all prescribed load work uses the existing ports.
    Positive grip-on-anchor power is energy delivered out to the driver.
    No trajectory, stability or physical/acoustic validation is inferred.
    """

    twist_rates: tuple[tuple[float, ...], ...]
    grip_responses: tuple[FiniteGripResponse, ...]
    shaft_kinetic_energy_j: float
    elastic_energy_j: float
    total_energy_j: float
    energy_rate_w: float
    applied_power_w: float
    anchor_power_w: float
    dissipated_power_w: float
    balance_residual: float

    @property
    def power_residual_w(self) -> float:
        return (
            self.energy_rate_w
            - self.applied_power_w
            + self.anchor_power_w
            + self.dissipated_power_w
        )

    @property
    def stability_status(self) -> str:
        return "unqualified"


@dataclass(frozen=True)
class _Assembly:
    kinetics: SectionKinetics
    elastic: ChainLinearization
    mass: np.ndarray
    known_wrench: np.ndarray


def _kinetics(chain: InertialMovingChain, state: MovingChainState) -> SectionKinetics:
    size = 6 * chain.shaft.node_count
    poses, velocity = np.asarray(state.poses), np.asarray(state.twists).ravel()
    mass, rate, bias = np.zeros((size, size)), np.zeros((size, size)), np.zeros(size)
    energy = 0.0
    for index, inertia in enumerate(chain.shaft.inertias):
        rows = slice(6 * index, 6 * (index + 2))
        result = inertia.evaluate(poses[index : index + 2], velocity[rows])
        mass[rows, rows] += result.mass
        rate[rows, rows] += result.mass_rate
        bias[rows] += result.bias
        energy += result.kinetic_energy_j
    return SectionKinetics(mass, bias, rate, energy)


def _root_motion(
    state: MovingChainState, node: int, rates: np.ndarray
) -> MaterialPointMotion:
    return MaterialPointMotion(
        np.asarray(state.poses)[node],
        np.asarray(state.twists)[node],
        rates[node],
        state.observer_id,
    )


def _assemble(chain: InertialMovingChain, state: MovingChainState) -> _Assembly:
    kinetics = _kinetics(chain, state)
    elastic_chain = chain.shaft.elastic
    elastic = elastic_chain.linearize(state.poses)
    mass = kinetics.mass.copy()
    known = kinetics.bias + elastic.residual
    zero_rates = np.zeros((chain.shaft.node_count, 6))
    for port in chain.grips:
        root = _root_motion(state, port.node, zero_rates)
        kinematics = moving_grip_kinematics(root, port.anchor)
        mapped = np.asarray(port.grip.inertance_factor) @ kinematics.root_motion_map
        response = finite_grip_response(port.grip, root, port.anchor)
        rows = slice(6 * port.node, 6 * (port.node + 1))
        mass[rows, rows] += mapped.T @ mapped
        known[rows] -= response.root_wrench
    return _Assembly(kinetics, elastic, mass, known)


def _solve(
    assembly: _Assembly, controls: MovingChainControls
) -> tuple[np.ndarray, float]:
    size, scales = len(assembly.known_wrench), controls.scales
    coordinates = np.tile([scales.length_m] * 3 + [1.0] * 3, size // 6)
    with np.errstate(under="raise"):
        mass = assembly.mass * coordinates[:, None] * coordinates[None, :]
        force = assembly.known_wrench * coordinates
    _validated_mass(mass, scales)
    acceleration = np.linalg.solve(mass, -force)
    defect = _finite_norm(mass @ acceleration + force, "acceleration residual")
    denominator = _finite_norm(mass, "mass") * _finite_norm(
        acceleration, "acceleration"
    ) + _finite_norm(force, "force")
    residual = float(defect / denominator) if denominator > 0 else float(defect)
    if (
        not np.isfinite(denominator)
        or not np.isfinite(residual)
        or residual > scales.residual_tolerance
    ):
        raise ValueError("moving chain acceleration residual is unresolved")
    rates = finite_array(coordinates * acceleration, (size,), "body twist rates")
    return rates.reshape(-1, 6), residual


def _applied_power(chain: InertialMovingChain, state: MovingChainState) -> float:
    elastic = chain.shaft.elastic
    poses, twists = np.asarray(state.poses), np.asarray(state.twists)
    return float(
        sum(
            item.load.power(poses[item.node], twists[item.node])
            for item in elastic.loads
        )
    )


def _finish(
    assembly: _Assembly,
    chain: InertialMovingChain,
    state: MovingChainState,
    solution: tuple[np.ndarray, float],
) -> MovingChainResponse:
    rates, residual = solution
    grips = tuple(
        finite_grip_response(
            port.grip, _root_motion(state, port.node, rates), port.anchor
        )
        for port in chain.grips
    )
    velocity = np.asarray(state.twists).ravel()
    applied = _applied_power(chain, state)
    kinetics, elastic = assembly.kinetics, assembly.elastic
    shaft_rate = float(
        velocity @ kinetics.mass @ rates.ravel()
        + velocity @ kinetics.mass_rate @ velocity / 2
    )
    elastic_rate = float(velocity @ elastic.residual + applied)
    total_energy = (
        kinetics.kinetic_energy_j
        + elastic.elastic_energy_j
        + sum(g.storage.inertial_energy_j + g.storage.elastic_energy_j for g in grips)
    )
    energy_rate = (
        shaft_rate + elastic_rate + sum(g.storage.stored_energy_rate_w for g in grips)
    )
    anchor = sum(g.anchor_power_w for g in grips)
    dissipation = sum(g.storage.dissipated_power_w for g in grips)
    finite_array(
        [total_energy, energy_rate, applied, anchor, dissipation],
        (5,),
        "moving energy/power",
    )
    return MovingChainResponse(
        twist_rates=tuple(tuple(float(x) for x in row) for row in rates),
        grip_responses=grips,
        shaft_kinetic_energy_j=kinetics.kinetic_energy_j,
        elastic_energy_j=elastic.elastic_energy_j,
        total_energy_j=total_energy,
        energy_rate_w=energy_rate,
        applied_power_w=applied,
        anchor_power_w=anchor,
        dissipated_power_w=dissipation,
        balance_residual=residual,
    )


def moving_chain_response(
    chain: InertialMovingChain,
    state: MovingChainState,
    controls: MovingChainControls,
) -> MovingChainResponse:
    """Solve all nodal body-twist derivatives with exact moving-port inertance.

    Preconditions: explicit inertial observer, proper current poses, finite
    twists/anchor states, declared strain domain and resolved positive mass.
    Postconditions: checked scaled acceleration residual and fresh work/energy
    output. No node is clamped, no mass is regularized and no motion history is
    inferred. Actual trajectory integration/convergence remains separate.
    """
    if not isinstance(chain, InertialMovingChain) or not isinstance(
        state, MovingChainState
    ):
        raise TypeError("expected InertialMovingChain and MovingChainState")
    if not isinstance(controls, MovingChainControls):
        raise TypeError("expected MovingChainControls")
    frame = chain.shaft.frame
    if state.observer_id != frame.frame_id:
        raise ValueError("state and shaft observers must agree")
    if len(np.asarray(state.poses)) != chain.shaft.node_count:
        raise ValueError("state node count does not match the shaft")
    _check_strains(chain.shaft.elastic, np.asarray(state.poses), controls.strain_limits)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            assembly = _assemble(chain, state)
            return _finish(assembly, chain, state, _solve(assembly, controls))
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as error:
        raise ValueError("moving chain numerical evaluation failed") from error


__all__ = ()
