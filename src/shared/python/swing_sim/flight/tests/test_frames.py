"""Frame-adapter tests: app frame <-> flight frame, both directions."""

from __future__ import annotations

import numpy as np
import pytest

from shared.python.swing_sim.flight import from_flight_frame, to_flight_frame


@pytest.mark.unit
def test_basis_vectors_map_correctly() -> None:
    # App frame: x target, y up, z right. Flight frame: x fwd, y left, z up.
    np.testing.assert_allclose(
        to_flight_frame(np.array([1.0, 0.0, 0.0])), [1.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        to_flight_frame(np.array([0.0, 1.0, 0.0])), [0.0, 0.0, 1.0]
    )  # app up -> flight up
    np.testing.assert_allclose(
        to_flight_frame(np.array([0.0, 0.0, 1.0])), [0.0, -1.0, 0.0]
    )  # app right -> flight -left


@pytest.mark.unit
def test_round_trip_is_identity_both_directions() -> None:
    rng = np.random.default_rng(42)
    vecs = rng.normal(size=(64, 3))
    np.testing.assert_allclose(from_flight_frame(to_flight_frame(vecs)), vecs)
    np.testing.assert_allclose(to_flight_frame(from_flight_frame(vecs)), vecs)
    single = np.array([1.5, -2.0, 0.25])
    np.testing.assert_allclose(from_flight_frame(to_flight_frame(single)), single)


@pytest.mark.unit
def test_adapters_are_proper_rotations() -> None:
    """Norms and handedness (cross products) are preserved."""
    rng = np.random.default_rng(7)
    a, b = rng.normal(size=3), rng.normal(size=3)
    fa, fb = to_flight_frame(a), to_flight_frame(b)
    assert np.linalg.norm(fa) == pytest.approx(np.linalg.norm(a))
    np.testing.assert_allclose(
        np.cross(fa, fb), to_flight_frame(np.cross(a, b)), atol=1e-12
    )


@pytest.mark.unit
def test_rejects_bad_shapes_and_nonfinite() -> None:
    with pytest.raises(ValueError, match="shape"):
        to_flight_frame(np.zeros(4))
    with pytest.raises(ValueError, match="finite"):
        from_flight_frame(np.array([np.inf, 0.0, 0.0]))
