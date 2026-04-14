import numpy as np
import pytest
from chaotic_pendulum.config import PhysicsConfig, RenderConfig
from chaotic_pendulum.physics import PhysicsEngine


def test_physics_dbc_valid_constraints() -> None:
    """Pre-conditions DbC passes strictly."""
    cfg = PhysicsConfig(m1=1.0, m2=1.0, gravity=9.81)
    assert cfg.m1 == 1.0


def test_physics_dbc_invalid_mass1() -> None:
    """DbC triggers ValueError on non-positive mass 1."""
    with pytest.raises(ValueError, match="Mass 1"):
        PhysicsConfig(m1=-1.0)


def test_physics_dbc_invalid_mass2() -> None:
    """DbC triggers ValueError on non-positive mass 2."""
    with pytest.raises(ValueError, match="Mass 2"):
        PhysicsConfig(m2=0.0)


def test_physics_dbc_invalid_length1() -> None:
    """DbC triggers ValueError on non-positive length 1."""
    with pytest.raises(ValueError, match="Length 1"):
        PhysicsConfig(l1=-0.5)


def test_physics_dbc_invalid_length2() -> None:
    """DbC triggers ValueError on non-positive length 2."""
    with pytest.raises(ValueError, match="Length 2"):
        PhysicsConfig(l2=0.0)


def test_physics_dbc_invalid_gravity() -> None:
    """DbC triggers ValueError on non-positive gravity."""
    with pytest.raises(ValueError, match="Gravity"):
        PhysicsConfig(gravity=-9.81)


def test_physics_dbc_invalid_damp1() -> None:
    """DbC triggers ValueError on negative damping 1."""
    with pytest.raises(ValueError, match="Damping"):
        PhysicsConfig(damp1=-0.1)


def test_physics_dbc_invalid_damp2() -> None:
    """DbC triggers ValueError on negative damping 2."""
    with pytest.raises(ValueError, match="Damping"):
        PhysicsConfig(damp2=-1.0)


def test_render_config_dbc_invalid_fps() -> None:
    """DbC triggers ValueError on non-positive FPS."""
    with pytest.raises(ValueError, match="FPS"):
        RenderConfig(fps=0)


def test_render_config_dbc_invalid_duration() -> None:
    """DbC triggers ValueError on non-positive duration."""
    with pytest.raises(ValueError, match="Duration"):
        RenderConfig(duration=-5)


def test_render_config_dbc_invalid_history_sec() -> None:
    """DbC triggers ValueError on non-positive history_sec."""
    with pytest.raises(ValueError, match="History"):
        RenderConfig(history_sec=0.0)


def test_physics_engine_none_config() -> None:
    """DbC triggers TypeError when config is None."""
    with pytest.raises(TypeError, match="Config cannot be None"):
        PhysicsEngine(None)  # type: ignore[arg-type]


def test_physics_engine_solver() -> None:
    """Solving outputs valid dict struct."""
    cfg = PhysicsConfig()
    engine = PhysicsEngine(cfg)

    # Simulating 0.1s
    res = engine.solve(0.1, 0.05)
    assert "pos" in res
    assert "v1" in res
    assert "v2" in res

    # V1 CF force is inward-directed, length equivalent to steps
    assert len(res["v1"]["centrifugal"][0]) == 2
    assert len(res["v2"]["coriolis"][0]) == 2


def test_physics_engine_solve_invalid_duration() -> None:
    """DbC triggers ValueError on non-positive duration."""
    cfg = PhysicsConfig()
    engine = PhysicsEngine(cfg)
    with pytest.raises(ValueError, match="Duration"):
        engine.solve(-1.0, 0.05)


def test_physics_engine_solve_invalid_dt() -> None:
    """DbC triggers ValueError on non-positive dt."""
    cfg = PhysicsConfig()
    engine = PhysicsEngine(cfg)
    with pytest.raises(ValueError, match="Duration"):
        engine.solve(1.0, 0.0)


def test_equations_of_motion() -> None:
    """Ensures RHS ODE signature maps cleanly."""
    cfg = PhysicsConfig(damp1=1.0, damp2=1.0)  # test damping
    engine = PhysicsEngine(cfg)

    derivatives = engine.equations_of_motion(0.0, [0.0, 1.0, 0.0, -1.0])
    assert len(derivatives) == 4


def test_equations_of_motion_invalid_state() -> None:
    """DbC triggers ValueError when state vector is wrong length."""
    cfg = PhysicsConfig()
    engine = PhysicsEngine(cfg)
    with pytest.raises(ValueError, match="State vector"):
        engine.equations_of_motion(0.0, [0.0, 1.0, 0.0])


def test_physics_force_vectors_tdd() -> None:
    """Verify analytical magnitude of Cartesian CF/Coriolis forces for first frame."""
    cfg = PhysicsConfig(
        m1=1.0,
        m2=2.0,
        l1=1.0,
        l2=1.0,
        theta1=np.pi / 2,
        omega1=1.0,
        theta2=0.0,
        omega2=2.0,
    )
    engine = PhysicsEngine(cfg)
    res = engine.solve(0.1, 0.05)

    # Intial values (t=0): w1 = 1.0, w2 = 2.0
    # CF_1 magnitude expected: m1 * l1 * w1^2 = 1.0 * 1.0 * 1.0 = 1.0
    c1_x, c1_y = res["v1"]["centrifugal"][0][0], res["v1"]["centrifugal"][1][0]
    assert np.isclose(np.hypot(c1_x, c1_y), 1.0)

    # Coriolis Force on 2: 2 * m2 * l2 * w1 * (w2 - w1)
    # 2 * 2.0 * 1.0 * 1.0 * (2.0 - 1.0) = 4.0
    cor2_x, cor2_y = res["v2"]["coriolis"][0][0], res["v2"]["coriolis"][1][0]
    assert np.isclose(np.hypot(cor2_x, cor2_y), 4.0)
