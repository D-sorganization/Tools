"""Contract tests for the swing objective cross-evaluation comparison.

The comparison only means something if two things hold: every swing genuinely
leads on its own objective (otherwise the solver is returning local optima and
the table is noise), and the reader can tell "the objectives agree" apart from
"the torque limit decided everything". Both are pinned here.

Closes #4770.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from double_pendulum_golf.physics import JointLimits, PendulumParams, TorqueClamp
from double_pendulum_golf.swing_objectives.comparison import (
    COMPARISON_SCHEMA_VERSION,
    SwingComparison,
    compare_objectives,
    comparison_from_payload,
    comparison_to_payload,
)

_PARAMS = PendulumParams(m1=5.0, m2=0.30, L1=0.65, L2=1.10, mClub=0.20)
_SUBSET = ("clubhead_speed", "centrifugal", "coriolis")


def _config():
    """Feasible tour-plausible downswing configuration."""
    from double_pendulum_golf.swing_objectives.downswing import DownswingConfig

    return DownswingConfig(
        params=_PARAMS,
        node_count=15,
        duration_s=0.34,
        initial_state=np.array([2.618, 1.745, 0.0, 0.0]),
        impact_theta1_rad=0.0,
        torque_clamp=TorqueClamp(max_torque1=180.0, max_torque2=20.0),
        joint_limits=JointLimits(
            phi_min=-0.175, phi_max=2.094, theta1_min=-4.0, theta1_max=4.0
        ),
    )


@pytest.fixture(scope="module")
def comparison() -> SwingComparison:
    """Run the comparison once for the whole module — each solve is seconds."""
    return compare_objectives(_config(), objective_keys=_SUBSET)


def test_comparison_solves_every_requested_objective(
    comparison: SwingComparison,
) -> None:
    """One converged, feasible swing per objective, in the requested order."""
    assert comparison.objective_keys == _SUBSET
    assert set(comparison.results) == set(_SUBSET)
    for key in _SUBSET:
        result = comparison.results[key]
        assert result.success, f"{key}: {result.message}"
        assert result.feasible, f"{key} defect {result.max_defect:.2e}"


def test_each_swing_leads_on_its_own_objective(comparison: SwingComparison) -> None:
    """The integrity check on the whole comparison.

    If some other swing beat the dedicated one on its own objective, the solver
    would be returning local optima and every conclusion drawn from the table
    would be unfounded.
    """
    matrix = comparison.matrix
    for index in range(len(_SUBSET)):
        column = matrix[:, index]
        assert column[index] == pytest.approx(100.0), (
            f"{_SUBSET[index]} did not lead its own column: {column}"
        )
        assert np.all(column <= 100.0 + 1e-9)


def test_matrix_is_square_finite_and_percentage_scaled(
    comparison: SwingComparison,
) -> None:
    """Row i, column j is objective j's value on swing i, as a % of that column's best."""
    matrix = comparison.matrix
    assert matrix.shape == (len(_SUBSET), len(_SUBSET))
    assert np.all(np.isfinite(matrix))
    assert np.all(matrix > 0.0)


def test_saturation_is_reported_so_agreement_can_be_interpreted(
    comparison: SwingComparison,
) -> None:
    """A reader must be able to tell agreement from a binding torque limit.

    If every objective simply pinned the torques to their bounds the swings
    would agree trivially, and the comparison would say nothing about the
    mechanisms.
    """
    for key in _SUBSET:
        fractions = comparison.torque_saturation[key]
        assert fractions.shape == (2,)
        assert np.all((fractions >= 0.0) & (fractions <= 1.0))


def test_raw_values_carry_units_and_match_the_results(
    comparison: SwingComparison,
) -> None:
    """The report keeps raw values, not only normalized percentages."""
    for row_key in _SUBSET:
        assert comparison.raw_values[row_key][row_key] == pytest.approx(
            comparison.results[row_key].objective_value
        )


def test_payload_round_trips_deterministically(comparison: SwingComparison) -> None:
    """The wire is canonical JSON: encode, decode and re-encode is a fixed point."""
    payload = comparison_to_payload(comparison)
    assert payload["schema_version"] == COMPARISON_SCHEMA_VERSION

    encoded = json.dumps(payload, sort_keys=True)
    restored = comparison_from_payload(json.loads(encoded))
    assert json.dumps(comparison_to_payload(restored), sort_keys=True) == encoded


def test_payload_fails_closed_on_a_missing_or_wrong_schema_version() -> None:
    """An unversioned or foreign payload is rejected rather than guessed at."""
    with pytest.raises(ValueError, match="schema_version"):
        comparison_from_payload({"objective_keys": list(_SUBSET)})
    with pytest.raises(ValueError, match="schema_version"):
        comparison_from_payload({"schema_version": "0.0.1", "objective_keys": list(_SUBSET)})


def test_payload_fails_closed_on_a_malformed_matrix(
    comparison: SwingComparison,
) -> None:
    """A matrix that is not square against the key list is rejected."""
    payload = comparison_to_payload(comparison)
    payload["matrix"] = [[1.0, 2.0], [3.0, 4.0]]
    with pytest.raises(ValueError, match="matrix"):
        comparison_from_payload(payload)


def test_compare_rejects_unknown_objective_keys() -> None:
    """Fails closed rather than silently comparing a smaller set."""
    with pytest.raises(KeyError, match="Unknown swing objective"):
        compare_objectives(_config(), objective_keys=("clubhead_speed", "vibes"))


def test_compare_requires_at_least_two_objectives() -> None:
    """A one-row comparison is meaningless, so it is refused."""
    with pytest.raises(ValueError, match="at least two"):
        compare_objectives(_config(), objective_keys=("clubhead_speed",))


def test_degeneracy_is_detected_when_the_constraints_pin_the_swing() -> None:
    """A collapsed feasible set must be reported, not shown as agreement.

    Close to the golfer's minimum downswing duration the constraints leave
    essentially one admissible trajectory, so every objective returns the same
    swing and the cross-evaluation matrix fills with 100% entries. That looks
    like the mechanisms unanimously agreeing; it is an artifact of the
    configuration, and callers must be able to tell the two apart.
    """
    from double_pendulum_golf.swing_objectives.downswing import DownswingConfig

    def build(duration_s: float, hub_torque: float, node_count: int) -> DownswingConfig:
        return DownswingConfig(
            params=_PARAMS,
            node_count=node_count,
            duration_s=duration_s,
            initial_state=np.array([2.618, 1.745, 0.0, 0.0]),
            impact_theta1_rad=0.0,
            torque_clamp=TorqueClamp(max_torque1=hub_torque, max_torque2=20.0),
            joint_limits=JointLimits(
                phi_min=-0.175, phi_max=2.094, theta1_min=-4.0, theta1_max=4.0
            ),
        )

    pinned = compare_objectives(build(0.34, 180.0, 17), objective_keys=_SUBSET)
    assert pinned.is_degenerate, (
        f"expected a collapsed feasible set, max swing distance "
        f"{pinned.max_swing_distance:.2e}"
    )
    assert np.allclose(pinned.matrix, 100.0, atol=1e-6)

    slack = compare_objectives(build(0.36, 250.0, 17), objective_keys=_SUBSET)
    assert not slack.is_degenerate
    assert slack.max_swing_distance > pinned.max_swing_distance


def test_degeneracy_flag_survives_the_wire(comparison: SwingComparison) -> None:
    """The warning must reach whoever reads the report, not just the caller."""
    payload = comparison_to_payload(comparison)
    assert payload["is_degenerate"] == comparison.is_degenerate
    assert comparison_from_payload(payload).is_degenerate == comparison.is_degenerate
