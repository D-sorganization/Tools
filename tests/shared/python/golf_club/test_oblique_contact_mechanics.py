"""Unit tests for non-spherical oblique contact and moving center of pressure.

Covers:
1. Curved face geometry with bulge/roll curvature and non-spherical projectiles.
2. Moving Center of Pressure (COP) kinematics, lever arms, and gear-effect torque.
3. Face/hosel structural modes with generalized force coupling and modal deflection.
4. Coupled oblique contact response and strict energy balance conservation.
5. Galilean observer frame invariance.
"""

from __future__ import annotations

import numpy as np
import pytest

from shared.python.swing_sim.impact._curved_contact_geometry import (
    CurvedFaceGeometry,
    NonSphericalContactGeometry,
)
from shared.python.swing_sim.impact._face_hosel_modes import (
    FaceHoselModalState,
    FaceHoselModalSystem,
    standard_driver_face_hosel_modes,
)
from shared.python.swing_sim.impact._moving_cop_kinematics import MovingCOPKinematics
from shared.python.swing_sim.impact._oblique_contact_response import (
    ObliqueContactModel,
    ObliqueContactState,
    evaluate_oblique_contact,
)
from shared.python.swing_sim.impact._spatial_contact_kinematics import ContactBodyState
from shared.python.swing_sim.impact.contact import KelvinVoigtContactLaw


def _make_body_state(
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation_matrix: np.ndarray | None = None,
    linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    observer_id: str = "inertial",
) -> ContactBodyState:
    """Construct a ContactBodyState with 4x4 pose and 6-vector twist."""
    pose = np.eye(4)
    if rotation_matrix is not None:
        pose[:3, :3] = rotation_matrix
    pose[:3, 3] = position
    twist = (*linear_velocity, *angular_velocity)
    return ContactBodyState(pose=pose, twist=twist, observer_id=observer_id)


def test_curved_face_geometry_validation_and_surface() -> None:
    """CurvedFaceGeometry validates parameters and computes surface elevation/normal."""
    # Positive parameters valid
    geom = CurvedFaceGeometry(
        bulge_radius_m=0.300,  # 300 mm bulge
        roll_radius_m=0.280,  # 280 mm roll
        face_half_width_m=0.050,  # 50 mm toe-heel
        face_half_height_m=0.030,  # 30 mm crown-sole
    )
    assert geom.bulge_radius_m == 0.300
    assert geom.roll_radius_m == 0.280

    # Non-positive parameters refused
    with pytest.raises(ValueError, match="bulge_radius_m"):
        CurvedFaceGeometry(
            bulge_radius_m=0.0,
            roll_radius_m=0.280,
            face_half_width_m=0.050,
            face_half_height_m=0.030,
        )
    with pytest.raises(ValueError, match="roll_radius_m"):
        CurvedFaceGeometry(
            bulge_radius_m=0.300,
            roll_radius_m=-0.1,
            face_half_width_m=0.050,
            face_half_height_m=0.030,
        )

    # Surface elevation at center is 0
    assert geom.surface_elevation_m(0.0, 0.0) == 0.0
    # Surface curves away (negative z) away from center
    elev_toe = geom.surface_elevation_m(0.020, 0.0)
    assert elev_toe < 0.0
    expected_toe_elev = -0.5 * (0.020**2) / 0.300
    np.testing.assert_allclose(elev_toe, expected_toe_elev, rtol=1e-10)

    # Surface normal at center is (0, 0, 1)
    np.testing.assert_allclose(
        geom.surface_normal(0.0, 0.0), (0.0, 0.0, 1.0), atol=1e-12
    )

    # Normal off-center has outward components
    norm_toe = geom.surface_normal(0.020, 0.0)
    assert norm_toe[0] > 0.0  # Points outward towards toe
    assert norm_toe[2] > 0.0  # Dominated by outward z
    np.testing.assert_allclose(np.linalg.norm(norm_toe), 1.0, atol=1e-12)


def test_non_spherical_contact_cop_projection() -> None:
    """NonSphericalContactGeometry finds exact closest surface point (COP)."""
    face_geom = CurvedFaceGeometry(
        bulge_radius_m=0.254,  # 10 inch bulge
        roll_radius_m=0.254,  # 10 inch roll
        face_half_width_m=0.060,
        face_half_height_m=0.035,
    )
    contact_geom = NonSphericalContactGeometry(
        face=face_geom,
        ball_radii_m=(0.02135, 0.02135, 0.02135),  # Standard golf ball
    )

    # Centered ball in front of face
    cop_x, cop_y, cop_z, normal = contact_geom.project_cop(0.0, 0.0, 0.030)
    np.testing.assert_allclose((cop_x, cop_y, cop_z), (0.0, 0.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(normal, (0.0, 0.0, 1.0), atol=1e-12)

    # Off-center toe strike: ball center at (0.015, 0.0, 0.025)
    cop_x, cop_y, cop_z, normal = contact_geom.project_cop(0.015, 0.0, 0.025)
    assert 0.0 < cop_x < 0.015
    assert cop_y == 0.0
    assert cop_z < 0.0
    # Line from COP to ball center must align with surface normal
    diff = np.array([0.015 - cop_x, 0.0 - cop_y, 0.025 - cop_z])
    diff_unit = diff / np.linalg.norm(diff)
    np.testing.assert_allclose(diff_unit, normal, atol=1e-10)


def test_moving_cop_kinematics_and_gear_effect_torque() -> None:
    """MovingCOPKinematics computes COP position, lever arms, and gear torque."""
    face_geom = CurvedFaceGeometry(
        bulge_radius_m=0.254,
        roll_radius_m=0.254,
        face_half_width_m=0.060,
        face_half_height_m=0.035,
    )
    contact_geom = NonSphericalContactGeometry(
        face=face_geom,
        ball_radii_m=(0.02135, 0.02135, 0.02135),
    )

    # Head centered at origin, COM slightly back (-0.030 m along z)
    face_state = _make_body_state(
        position=(0.0, 0.0, 0.0),
        linear_velocity=(0.0, 0.0, 40.0),  # 40 m/s clubhead speed
        angular_velocity=(0.0, 0.0, 0.0),
    )
    # Ball centered at toe strike location (x=0.015 m, gap compressed)
    ball_state = _make_body_state(
        position=(0.015, 0.0, 0.020),  # Compressed against radius 0.02135
        linear_velocity=(0.0, 0.0, 0.0),
    )

    kinematics = MovingCOPKinematics(
        face=face_state,
        ball=ball_state,
        geometry=contact_geom,
        head_com_material_offset_m=(0.0, 0.0, -0.035),
    )

    # Gap is negative (in compression)
    assert kinematics.gap_m < 0.0
    assert kinematics.is_in_contact

    # COP coordinates on face
    assert kinematics.cop_material_m[0] > 0.0  # Toe side

    # Lever arm from head COM to COP has positive x component
    lever_arm = kinematics.lever_arm_to_head_com_m
    assert lever_arm[0] > 0.010

    # Normal impact force on face (directed backwards along -z)
    contact_force_on_ball = np.array([0.0, 0.0, 2000.0])  # 2000 N normal force
    load_pair = kinematics.load_pair(contact_force_on_ball)

    # Face reaction is -2000 N in z direction
    # Torque about COM: lever_arm x F_face
    # F_face has negative z: (rx, ry, rz) x (0, 0, -F) = (-ry*F, rx*F, 0)
    # Since rx > 0 and Fz < 0, torque about y is positive (+rx * Fz_mag),
    # which rotates the clubhead open (counter-clockwise viewed from above with +y up).
    # This is the classic physical GEAR EFFECT!
    face_wrench = np.asarray(load_pair.face_wrench)
    face_torque_y = face_wrench[4]  # index 4 is torque around y
    assert face_torque_y > 10.0  # Twists clubhead open


def test_migration_velocity_vs_material_velocity() -> None:
    """Work conjugacy strictly uses material point velocities, not point migration."""
    face_geom = CurvedFaceGeometry(
        bulge_radius_m=0.254,
        roll_radius_m=0.254,
        face_half_width_m=0.060,
        face_half_height_m=0.035,
    )
    contact_geom = NonSphericalContactGeometry(
        face=face_geom,
        ball_radii_m=(0.02135, 0.02135, 0.02135),
    )

    face_state = _make_body_state(
        position=(0.0, 0.0, 0.0),
        linear_velocity=(0.0, 0.0, 45.0),
        angular_velocity=(10.0, 0.0, 0.0),  # Rotating face
    )
    ball_state = _make_body_state(
        position=(0.010, 0.005, 0.021),
        linear_velocity=(2.0, -1.0, 0.0),
        angular_velocity=(0.0, 100.0, 0.0),  # Spinning ball
    )

    kinematics = MovingCOPKinematics(
        face=face_state,
        ball=ball_state,
        geometry=contact_geom,
        head_com_material_offset_m=(0.0, 0.0, -0.030),
    )

    force = np.array([50.0, -30.0, 1500.0])
    load_pair = kinematics.load_pair(force)

    # Power must match F . v_rel_material exactly
    v_rel_mat = np.asarray(kinematics.relative_velocity_mps)
    expected_power = float(np.dot(force, v_rel_mat))
    np.testing.assert_allclose(load_pair.power_w, expected_power, rtol=1e-12)


def test_face_hosel_modal_system_dynamics_and_energy() -> None:
    """FaceHoselModalSystem simulates flexible modes with energy balance."""
    modes = standard_driver_face_hosel_modes()
    assert len(modes.modes) >= 3  # Trampoline, hosel bending, hosel torsion

    # Initial state at rest
    state = FaceHoselModalState.zeros(modes.mode_count)
    assert state.generalized_coordinates.shape == (modes.mode_count,)
    assert state.generalized_velocities.shape == (modes.mode_count,)

    # Apply a normal impact force at center (0, 0)
    normal_force_n = 2500.0
    cop_coords = (0.0, 0.0)  # Face center

    gen_forces = modes.generalized_forces(cop_coords, normal_force_n)
    # Trampoline mode has maximum shape at center, so it gets strong excitation
    assert gen_forces[0] > 1000.0

    # Compute modal accelerations
    accel = modes.modal_accelerations(state, gen_forces)
    assert accel[0] > 0.0  # Accelerated in response to normal impact

    # Advance state with non-zero velocities
    v_modal = np.zeros(modes.mode_count)
    v_modal[0] = 5.0  # 5 m/s modal velocity on trampoline mode
    q_modal = np.zeros(modes.mode_count)
    q_modal[0] = 0.0002  # 0.2 mm deflection
    perturbed_state = FaceHoselModalState(
        generalized_coordinates=q_modal,
        generalized_velocities=v_modal,
    )

    # Deflection at center
    deflection = modes.modal_deflection_m(perturbed_state, (0.0, 0.0))
    assert deflection > 0.0

    # Modal energy and dissipation
    e_modal = modes.modal_energy_j(perturbed_state)
    assert e_modal > 0.0

    p_diss = modes.modal_dissipation_power_w(perturbed_state)
    assert p_diss > 0.0  # Positive dissipation (passivity!)


def test_coupled_oblique_contact_response_and_energy_conservation() -> None:
    """Coupled oblique impact response strictly balances energy across all channels."""
    face_geom = CurvedFaceGeometry(
        bulge_radius_m=0.254,
        roll_radius_m=0.254,
        face_half_width_m=0.060,
        face_half_height_m=0.035,
    )
    contact_geom = NonSphericalContactGeometry(
        face=face_geom,
        ball_radii_m=(0.02135, 0.02135, 0.02135),
    )
    modes = standard_driver_face_hosel_modes()
    contact_law = KelvinVoigtContactLaw(
        stiffness_n_per_m=1.2e6,
        damping_n_s_per_m=25.0,
    )

    model = ObliqueContactModel(
        geometry=contact_geom,
        modes=modes,
        normal_law=contact_law,
        friction_coefficient=0.25,
        tangential_stiffness_n_per_m=8.0e5,
        head_com_material_offset_m=(0.0, 0.0, -0.035),
    )

    # State in contact with oblique relative velocity
    face_state = _make_body_state(
        position=(0.0, 0.0, 0.0),
        linear_velocity=(0.0, 0.0, 42.0),
    )
    ball_state = _make_body_state(
        position=(0.008, -0.004, 0.020),  # Compressed into face
        linear_velocity=(5.0, -2.0, 0.0),  # Oblique slip
    )
    modal_state = FaceHoselModalState.zeros(modes.mode_count)

    state = ObliqueContactState(
        face=face_state,
        ball=ball_state,
        modes=modal_state,
        tangential_deflection_m=(1e-5, -5e-6, 0.0),
    )

    response = evaluate_oblique_contact(model, state)

    # Contact is active
    assert response.normal_force_n > 0.0
    assert response.friction_force_n[0] != 0.0 or response.friction_force_n[1] != 0.0

    # Friction obeys Coulomb cone
    friction_mag = np.linalg.norm(response.friction_force_n[:2])
    coulomb_limit = model.friction_coefficient * response.normal_force_n
    assert friction_mag <= coulomb_limit + 1e-10

    # Check power balance residual
    assert abs(response.power_residual_w) < 1e-10


def test_observer_frame_invariance() -> None:
    """Contact kinematics and forces are invariant under frame translation/rotation."""
    face_geom = CurvedFaceGeometry(
        bulge_radius_m=0.254,
        roll_radius_m=0.254,
        face_half_width_m=0.060,
        face_half_height_m=0.035,
    )
    contact_geom = NonSphericalContactGeometry(
        face=face_geom,
        ball_radii_m=(0.02135, 0.02135, 0.02135),
    )

    # Frame 1: Inertial frame
    face_1 = _make_body_state(
        position=(0.0, 0.0, 0.0),
        linear_velocity=(0.0, 0.0, 40.0),
        observer_id="frame1",
    )
    ball_1 = _make_body_state(
        position=(0.010, 0.005, 0.020),
        linear_velocity=(0.0, 0.0, 0.0),
        observer_id="frame1",
    )
    kin_1 = MovingCOPKinematics(
        face=face_1,
        ball=ball_1,
        geometry=contact_geom,
        head_com_material_offset_m=(0.0, 0.0, -0.030),
    )

    # Frame 2: Shifted by (10, -5, 20) m and translated at constant (15, 30, -5) m/s
    shift_pos = np.array([10.0, -5.0, 20.0])
    shift_vel = np.array([15.0, 30.0, -5.0])

    face_2 = _make_body_state(
        position=tuple(shift_pos),
        linear_velocity=tuple(np.array([0.0, 0.0, 40.0]) + shift_vel),
        observer_id="frame2",
    )
    ball_2 = _make_body_state(
        position=tuple(np.array([0.010, 0.005, 0.020]) + shift_pos),
        linear_velocity=tuple(shift_vel),
        observer_id="frame2",
    )
    kin_2 = MovingCOPKinematics(
        face=face_2,
        ball=ball_2,
        geometry=contact_geom,
        head_com_material_offset_m=(0.0, 0.0, -0.030),
    )

    # Invariants
    np.testing.assert_allclose(kin_1.gap_m, kin_2.gap_m, atol=1e-12)
    np.testing.assert_allclose(kin_1.gap_rate_mps, kin_2.gap_rate_mps, atol=1e-12)
    np.testing.assert_allclose(kin_1.cop_material_m, kin_2.cop_material_m, atol=1e-12)
    np.testing.assert_allclose(
        kin_1.relative_velocity_mps, kin_2.relative_velocity_mps, atol=1e-12
    )


def test_refusals_and_contract_violations() -> None:
    """DbC contract checks reject invalid geometry, modes, and observer states."""
    face_geom = CurvedFaceGeometry(
        bulge_radius_m=0.254,
        roll_radius_m=0.254,
        face_half_width_m=0.060,
        face_half_height_m=0.035,
    )
    # Mismatched observer IDs
    face_state = _make_body_state(observer_id="frame_A")
    ball_state = _make_body_state(observer_id="frame_B")
    contact_geom = NonSphericalContactGeometry(face=face_geom)

    with pytest.raises(ValueError, match="same observer"):
        MovingCOPKinematics(face=face_state, ball=ball_state, geometry=contact_geom)

    # Empty modes system refused
    with pytest.raises(ValueError, match="cannot be empty"):
        FaceHoselModalSystem(())

    # Incompatible generalized coordinates/velocities dimensions
    with pytest.raises(ValueError, match="same dimension"):
        FaceHoselModalState(
            generalized_coordinates=np.zeros(2),
            generalized_velocities=np.zeros(3),
        )


def test_heel_and_vertical_strikes_gear_effect() -> None:
    """Heel, high, and low strikes generate consistent gear effect torques."""
    face_geom = CurvedFaceGeometry(
        bulge_radius_m=0.254,
        roll_radius_m=0.254,
        face_half_width_m=0.060,
        face_half_height_m=0.035,
    )
    contact_geom = NonSphericalContactGeometry(face=face_geom)
    face_state = _make_body_state(
        position=(0.0, 0.0, 0.0),
        linear_velocity=(0.0, 0.0, 40.0),
    )

    # 1. Heel strike (x = -0.015 m)
    ball_heel = _make_body_state(position=(-0.015, 0.0, 0.020))
    kin_heel = MovingCOPKinematics(
        face=face_state,
        ball=ball_heel,
        geometry=contact_geom,
        head_com_material_offset_m=(0.0, 0.0, -0.035),
    )
    load_heel = kin_heel.load_pair(np.array([0.0, 0.0, 2000.0]))
    # Heel strike torque around y must be negative (twists clubhead closed)
    torque_y_heel = np.asarray(load_heel.face_wrench)[4]
    assert torque_y_heel < -10.0

    # 2. High strike (y = +0.012 m above center)
    ball_high = _make_body_state(position=(0.0, 0.012, 0.020))
    kin_high = MovingCOPKinematics(
        face=face_state,
        ball=ball_high,
        geometry=contact_geom,
        head_com_material_offset_m=(0.0, 0.0, -0.035),
    )
    load_high = kin_high.load_pair(np.array([0.0, 0.0, 2000.0]))
    # High strike: rx=0, ry>0, rz>0.
    # r x (-Fz) = (ry, 0) x (0, -Fz) = (-ry * Fz, 0, 0) in torque.
    # Since ry > 0 and F_face_z < 0: (-ry * -Fz) = -ry * Fz, around x:
    # ry * (-Fz) -> (-ry * (-2000)) = -ry*(-F) = -ry * (-Fz)...
    # r x F = (ry*Fz - rz*Fy) i + (rz*Fx - rx*Fz) j + (rx*Fy - ry*Fx) k
    # With F = (0, 0, -2000):
    # tx = ry * (-2000) - 0 = -2000 * ry < 0 (increases effective dynamic loft!)
    torque_x_high = np.asarray(load_high.face_wrench)[3]
    assert torque_x_high < -10.0
