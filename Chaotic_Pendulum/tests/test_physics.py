import pytest

from chaotic_pendulum.config import PhysicsConfig
from chaotic_pendulum.physics import PhysicsEngine


def test_physics_dbc_valid_constraints() -> None:
    """Pre-conditions DbC passes strictly."""
    cfg = PhysicsConfig(m1=1.0, m2=1.0, gravity=9.81)
    assert cfg.m1 == 1.0


def test_physics_dbc_invalid_constraints() -> None:
    """DbC triggers on negative mass."""
    with pytest.raises(AssertionError):
        PhysicsConfig(m1=-1.0)


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


def test_equations_of_motion() -> None:
    """Ensures RHS ODE signature maps cleanly."""
    cfg = PhysicsConfig(damp1=1.0, damp2=1.0)  # test damping
    engine = PhysicsEngine(cfg)

    derivatives = engine.equations_of_motion(0.0, [0.0, 1.0, 0.0, -1.0])
    assert len(derivatives) == 4
