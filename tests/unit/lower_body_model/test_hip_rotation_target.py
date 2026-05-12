import pytest

pytest.importorskip("numpy")
import numpy as np

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
    with pytest.raises(ValueError):
        InclinedPlaneHipRotationTarget(**kwargs)


def test_rotation_degrees_at_rejects_negative_time() -> None:
    target = InclinedPlaneHipRotationTarget(duration_sec=1.0)
    with pytest.raises(ValueError):
        target.rotation_degrees_at(-0.1)


def test_lateral_shift_at_is_zero_during_backswing_and_smoothsteps_to_finish() -> None:
    target = InclinedPlaneHipRotationTarget(
        duration_sec=2.0, lateral_shift_m=0.08, sample_count=2
    )

    assert target.lateral_shift_at(0.0) == pytest.approx(0.0)
    assert target.lateral_shift_at(0.5) == pytest.approx(0.0)
    assert target.lateral_shift_at(1.0) == pytest.approx(0.0)
    # At mid-downswing the smoothstep value is s(0.5) = 0.5.
    assert target.lateral_shift_at(1.5) == pytest.approx(0.08 * 0.5, abs=1e-6)
    assert target.lateral_shift_at(2.0) == pytest.approx(0.08, abs=1e-6)


def test_lateral_shift_at_rejects_negative_time() -> None:
    target = InclinedPlaneHipRotationTarget(duration_sec=1.0, lateral_shift_m=0.05)
    with pytest.raises(ValueError):
        target.lateral_shift_at(-0.1)


def test_lateral_shift_m_range_enforced() -> None:
    with pytest.raises(ValueError):
        InclinedPlaneHipRotationTarget(duration_sec=1.0, lateral_shift_m=1.0)


def test_target_quaternion_at_zero_rotation_is_identity() -> None:
    target = InclinedPlaneHipRotationTarget(duration_sec=1.0, incline_degrees=12.0)
    q = target.target_quaternion_at(0.0)
    assert q[0] == pytest.approx(1.0)
    assert np.allclose(q[1:], 0.0)


def test_target_quaternion_at_rotation_axis_reflects_incline() -> None:
    """A non-zero incline must push the rotation axis off pure +Z."""
    target = InclinedPlaneHipRotationTarget(
        duration_sec=2.0, incline_degrees=30.0, backswing_degrees=45.0
    )
    # Use the reversal point where rotation_deg = -backswing_degrees.
    q = target.target_quaternion_at(1.0)
    # Axis-angle: vec = sin(angle/2) * axis.
    vec = q[1:]
    vec_norm = np.linalg.norm(vec)
    assert vec_norm > 0.0
    axis = vec / vec_norm
    # With incline=30 the axis has a significant X component (sin(30)=0.5).
    assert abs(axis[0]) > 0.3
    assert axis[1] == pytest.approx(0.0, abs=1e-6)
    assert abs(axis[2]) > 0.7


def test_simulator_pelvis_driver_tracks_lateral_shift() -> None:
    target = InclinedPlaneHipRotationTarget(
        duration_sec=1.0,
        backswing_degrees=10.0,
        counterclockwise_degrees=20.0,
        incline_degrees=5.0,
        lateral_shift_m=0.10,
    )
    sim = LowerBodySimulator(build_lower_body_xml())
    sim.setup_initial_pose()
    sim.set_pelvis_inclined_rotation(target)

    initial_y = float(sim.data.xpos[sim.pelvis_body_id][1])

    # Drive the simulator to just past the reversal point.
    while sim.data.time < 0.9:
        sim.step()

    final_y = float(sim.data.xpos[sim.pelvis_body_id][1])
    # The pelvis should have shifted in +Y during the downswing phase.
    assert (
        final_y - initial_y > 0.01
    ), f"expected +Y shift; got {final_y - initial_y:.4f}"


def test_set_pelvis_inclined_rotation_rejects_bad_gains() -> None:
    sim = LowerBodySimulator(build_lower_body_xml())
    target = InclinedPlaneHipRotationTarget(duration_sec=1.0)
    with pytest.raises(ValueError):
        sim.set_pelvis_inclined_rotation(target, position_kp=-1.0)
    with pytest.raises(TypeError):
        sim.set_pelvis_inclined_rotation(target, orientation_kp="loose")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        sim.set_pelvis_inclined_rotation("not a target")  # type: ignore[arg-type]


def test_clear_pelvis_inclined_rotation_removes_driver() -> None:
    sim = LowerBodySimulator(build_lower_body_xml())
    target = InclinedPlaneHipRotationTarget(duration_sec=1.0, lateral_shift_m=0.05)
    sim.set_pelvis_inclined_rotation(target)
    sim.step()
    sim.clear_pelvis_inclined_rotation()
    assert sim._pelvis_driver_target is None
    assert np.allclose(sim.data.xfrc_applied[sim.pelvis_body_id], 0.0)


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
