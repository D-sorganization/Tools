import numpy as np
import pytest

from lower_body_model.builder import build_lower_body_xml
from lower_body_model.hip_rotation import InclinedPlaneHipRotationTarget
from lower_body_model.simulator import LowerBodySimulator


def test_target_reverses_from_clockwise_to_counterclockwise() -> None:
    target = InclinedPlaneHipRotationTarget(duration_sec=2.0, sample_count=5)

    assert target.rotation_degrees_at(0.0) == pytest.approx(0.0)
    assert target.rotation_degrees_at(target.reversal_time_sec) == pytest.approx(-45.0)
    assert target.rotation_degrees_at(2.0) == pytest.approx(45.0)
    assert [sample.time_sec for sample in target.sample()] == pytest.approx(
        [0.0, 0.5, 1.0, 1.5, 2.0]
    )


def test_target_samples_inclined_plane_deterministically() -> None:
    target = InclinedPlaneHipRotationTarget(duration_sec=1.0, incline_degrees=30.0)

    first = target.plane_point_at(0.25)
    second = target.plane_point_at(0.25)

    assert np.array_equal(first, second)
    assert first[2] != pytest.approx(0.0)
    assert np.linalg.norm(first) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"duration_sec": 0.0},
        {"duration_sec": 1.0, "backswing_degrees": 0.0},
        {"duration_sec": 1.0, "counterclockwise_degrees": 0.0},
        {"duration_sec": 1.0, "incline_degrees": 90.0},
        {"duration_sec": 1.0, "sample_count": 1},
    ],
)
def test_target_rejects_nonphysical_parameters(kwargs: dict[str, float]) -> None:
    with pytest.raises(AssertionError):
        InclinedPlaneHipRotationTarget(**kwargs)


def test_simulator_applies_target_to_both_hip_sockets() -> None:
    simulator = LowerBodySimulator(build_lower_body_xml())
    simulator.configure_hip_rotation_target(
        duration_sec=2.0, incline_degrees=15.0, sample_count=5
    )

    applied = simulator.apply_hip_rotation_target(1.0)

    assert applied == {"rotation_deg": -45.0, "incline_deg": 15.0}
    for side in ("r", "l"):
        assert simulator.data.qpos[
            simulator.jnt_qpos_idx[f"{side}_hip_z"]
        ] == pytest.approx(np.radians(-45.0))
        assert simulator.data.qpos[
            simulator.jnt_qpos_idx[f"{side}_hip_x"]
        ] == pytest.approx(np.radians(15.0))


def test_simulator_steps_with_configured_target_history() -> None:
    simulator = LowerBodySimulator(build_lower_body_xml())
    simulator.configure_hip_rotation_target(duration_sec=1.0)

    simulator.step()

    assert simulator.history
    diagnostics = simulator.compute_diagnostics()
    assert diagnostics["hip_rotation_target"]["rotation_deg"] <= 0.0


def test_history_frame_exposes_hip_rotation_target_diagnostics() -> None:
    simulator = LowerBodySimulator(build_lower_body_xml())
    target = simulator.configure_hip_rotation_target(
        duration_sec=2.0, incline_degrees=15.0, sample_count=5
    )

    simulator.step()

    frame = simulator.history[-1]
    history_diag = simulator.get_history_diagnostics(0)

    assert frame["hip_rotation_target"] is not None
    assert history_diag["time_sec"] == pytest.approx(frame["time"])
    assert history_diag["hip_rotation_target"]["rotation_deg"] == pytest.approx(
        target.rotation_degrees_at(frame["time"])
    )
    assert history_diag["hip_rotation_target"]["incline_deg"] == pytest.approx(15.0)
