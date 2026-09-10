"""Nested-mesh rotating release study; fine mesh is not a continuum oracle."""

import json
from collections.abc import Callable

import numpy as np

from shared.python.golf_club._shaft_gripped_equilibrium import solve_gripped_chain
from shared.python.golf_club._shaft_gripped_operating import constant_gripped_model

from .test_shaft_equilibrium import _controls
from .test_shaft_gripped_operating import _scales
from .test_shaft_gripped_response import _model
from .test_shaft_rotating_disturbance import _DURATION_S
from .test_shaft_trajectory_rotation import _SPIN_RAD_S


def _release_state(elements: int) -> tuple[np.ndarray, np.ndarray]:
    """Same small tip force held before release; initial-state error stays visible."""
    chain, seed = _model(elements, damping=0, spin=_SPIN_RAD_S)
    equilibrium = solve_gripped_chain(chain, seed, _controls())
    scales = _scales()
    model = constant_gripped_model(chain, equilibrium.poses, _controls(), scales)
    _, _, _, stiffness = model.pencil.arrays()
    load = np.zeros(len(stiffness))
    # Synthetic 10 micronewton force in two bending planes, removed at t=0.
    load[-6:-4] = scales.length_m * 1e-5 * np.array([1.0, -0.5])
    displacement = np.linalg.solve(stiffness, load - np.asarray(model.scaled_residual))
    initial = np.r_[displacement, np.zeros_like(displacement)]
    final = np.asarray(model.scaled_state_at(initial, _DURATION_S))
    indices = np.arange(0, elements + 1, elements // 4)
    selected = np.r_[
        (6 * indices[:, None] + np.arange(6)).ravel(),
        len(stiffness) + (6 * indices[:, None] + np.arange(6)).ravel(),
    ]
    return initial[selected], final[selected]


def test_rotating_release_initial_and_dynamic_states_converge_on_nested_meshes(
    record_property: Callable[[str, object], None],
) -> None:
    # The same five material positions and scaled position/velocity entries
    # are compared on every mesh. Temporal propagation is a matrix exponential.
    meshes = (4, 8, 16, 32)
    states = [_release_state(elements) for elements in meshes]
    initial_reference, final_reference = states[-1]
    reference_scale = np.linalg.norm(final_reference)
    assert reference_scale > 1e-6
    rows = []
    for elements, (initial, final) in zip(meshes[:-1], states[:-1], strict=True):
        rows.append(
            {
                "elements": elements,
                "relative_initial_error": float(
                    np.linalg.norm(initial - initial_reference) / reference_scale
                ),
                "relative_final_error": float(
                    np.linalg.norm(final - final_reference) / reference_scale
                ),
                "relative_motion_increment_error": float(
                    np.linalg.norm(
                        (final - initial) - (final_reference - initial_reference)
                    )
                    / reference_scale
                ),
            }
        )
    record_property(
        "study",
        json.dumps({"reference_elements": 32, "time_s": _DURATION_S, "rows": rows}),
    )
    for field in ("relative_initial_error", "relative_final_error"):
        errors = [row[field] for row in rows]
        assert errors[0] > errors[1] > errors[2]
        assert errors[-1] < 0.01
        assert 2 < errors[0] / errors[1] < 8
        assert 2 < errors[1] / errors[2] < 8
