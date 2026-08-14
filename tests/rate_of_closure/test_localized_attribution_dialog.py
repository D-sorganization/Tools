"""Explicit paired-study design dialog and normalization tests."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from rate_of_closure.ui.pyqt6.localized_attribution_dialog import (  # noqa: E402
    LocalizedAttributionRunDialog,
    StateTargetSelection,
    build_localized_attribution_design,
)
from shared.python.swing_sim.run_config import (  # noqa: E402
    SHOULDER_JOINT_ID,
    WRIST_JOINT_ID,
)
from shared.python.swing_sim.variation import (  # noqa: E402
    CATEGORY_SWING,
    NoiseSpec,
    VariationPlan,
)

from .test_variation_simulation_request import (  # noqa: E402
    _SHOULDER_TORQUE_OFFSET,
    _WRIST_TORQUE_OFFSET,
    _base_config,
    _localized_spec,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _mixed_plan() -> VariationPlan:
    yaw = f"{CATEGORY_SWING}.yaw_deg"
    shoulder = _localized_spec(_SHOULDER_TORQUE_OFFSET, SHOULDER_JOINT_ID)
    wrist = _localized_spec(
        _WRIST_TORQUE_OFFSET, WRIST_JOINT_ID, window=(0.05, 0.08), scale=3.0
    )
    return VariationPlan(
        mode="swing",
        base_variables={yaw: 1.25},
        noise=(NoiseSpec(yaw, scale=0.5), shoulder, wrist),
        n_runs=99,
        seed=12,
    )


def test_design_filters_globals_and_builds_fixed_state_impact_shot_targets() -> None:
    plan = _mixed_plan()
    localized = plan.noise[1:]
    deltas = {localized[0].spec_id: 2.5, localized[1].spec_id: -1.5}

    design = build_localized_attribution_design(
        plan,
        _base_config(),
        deltas,
        StateTargetSelection("swing.clubhead.reference", 0.02),
    )

    assert design.source_plan.noise == localized
    assert design.source_plan.n_runs == 4
    assert design.source_plan.groups == ()
    assert design.source_plan.base_variables == plan.base_variables
    assert design.intervention_deltas_nm == deltas
    assert len(design.targets) == 17
    assert {target.kind for target in design.targets} == {"state", "impact", "shot"}
    assert all(
        target.point_id == "swing.clubhead.reference" and target.time_s == 0.02
        for target in design.targets
        if target.kind == "state"
    )


def test_design_rejects_zero_delta_before_execution() -> None:
    plan = _mixed_plan()
    localized = plan.noise[1:]
    with pytest.raises(Exception, match="nonzero"):
        build_localized_attribution_design(
            plan,
            _base_config(),
            {localized[0].spec_id: 0.0, localized[1].spec_id: 1.0},
            StateTargetSelection("swing.clubhead.reference", 0.02),
        )


def test_dialog_exposes_exact_source_locus_and_exact_grid_selectors(qtbot) -> None:  # type: ignore[no-untyped-def]
    dialog = LocalizedAttributionRunDialog(_mixed_plan(), _base_config())
    qtbot.addWidget(dialog)

    assert dialog._sources.rowCount() == 2
    assert dialog._sources.item(0, 0).text() == _mixed_plan().noise[1].spec_id
    assert dialog._sources.item(0, 2).text() == SHOULDER_JOINT_ID
    assert dialog._sources.item(0, 3).text() == "[0.02, 0.04) s"
    assert dialog._point.currentData() == "swing.clubhead.reference"
    assert isinstance(dialog._time.currentData(), float)
    assert "global Monte Carlo factors remain fixed" in dialog._explanation.text()
    assert "4 explicit trials" in dialog._summary.text()

    design = dialog.build_design()
    assert design.source_plan.n_runs == 4
    assert set(design.intervention_deltas_nm) == {
        _mixed_plan().noise[1].spec_id,
        _mixed_plan().noise[2].spec_id,
    }
