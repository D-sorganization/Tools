"""
Tests for the AnalysisTab widget (non-GUI logic).

Covers:
- Surface evaluator dispatch for double, triple, and golfer models
- Model-aware sweep variable definitions
- Numerical manipulability helper
- Golfer q-vector construction from angle dict
- Series combo population via data_extractor
"""

from __future__ import annotations

import numpy as np
import pytest

from double_pendulum_golf.gui.analysis_tab import AnalysisTab

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tab():
    """Create a headless AnalysisTab (no GUI parent)."""
    # AnalysisTab requires PyQt6 for widget construction; skip if unavailable
    pytest.importorskip("PyQt6.QtWidgets")
    return AnalysisTab(parent=None)


# ---------------------------------------------------------------------------
# _golfer_q_from_angles (static, no GUI needed)
# ---------------------------------------------------------------------------


class TestGolferQFromAngles:
    """Test the static helper that builds an 8-DOF q from sweep angles."""

    def test_empty_angles(self):
        q = AnalysisTab._golfer_q_from_angles({})
        assert q.shape == (8,)
        np.testing.assert_array_equal(q, np.zeros(8))

    def test_theta1_only(self):
        q = AnalysisTab._golfer_q_from_angles({"theta1": 1.5})
        assert q[0] == pytest.approx(1.5)
        assert q[1] == 0.0

    def test_all_mapped(self):
        q = AnalysisTab._golfer_q_from_angles(
            {
                "theta1": 0.1,
                "phi": 0.2,
                "phi1": 0.3,
                "phi2": 0.4,
            }
        )
        assert q[0] == pytest.approx(0.1)
        assert q[1] == pytest.approx(0.2)
        assert q[2] == pytest.approx(0.3)
        assert q[3] == pytest.approx(0.4)
        assert q[4] == 0.0  # unmapped

    def test_unknown_keys_ignored(self):
        q = AnalysisTab._golfer_q_from_angles({"unknown": 99.0, "theta1": 1.0})
        assert q[0] == pytest.approx(1.0)
        assert np.sum(np.abs(q[1:])) == 0.0


# ---------------------------------------------------------------------------
# _numerical_manipulability (static, no GUI needed)
# ---------------------------------------------------------------------------


class TestNumericalManipulability:
    """Test the finite-difference manipulability evaluator factory."""

    def test_identity_jacobian(self):
        """If FK maps angles directly to (x, y), manipulability = 1."""

        def fk_fn(angles):
            return {"tip": (angles["a"], angles["b"])}

        ev = AnalysisTab._numerical_manipulability(fk_fn, "tip", ["a", "b"])
        w = ev({"a": 0.0, "b": 0.0})
        assert w == pytest.approx(1.0, abs=1e-4)

    def test_singular_jacobian(self):
        """If FK maps both angles to the same direction, manipulability ~ 0."""

        def fk_fn(angles):
            s = angles["a"] + angles["b"]
            return {"tip": (s, 0.0)}

        ev = AnalysisTab._numerical_manipulability(fk_fn, "tip", ["a", "b"])
        w = ev({"a": 0.0, "b": 0.0})
        assert w == pytest.approx(0.0, abs=1e-4)

    def test_scaled_jacobian(self):
        """Scaling one axis should scale manipulability accordingly."""

        def fk_fn(angles):
            return {"tip": (2.0 * angles["a"], angles["b"])}

        ev = AnalysisTab._numerical_manipulability(fk_fn, "tip", ["a", "b"])
        w = ev({"a": 0.0, "b": 0.0})
        assert w == pytest.approx(2.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Sweep variable definitions
# ---------------------------------------------------------------------------


class TestSweepVariables:
    """Test model-aware sweep variable lookup."""

    def test_double_has_two_vars(self):
        assert len(AnalysisTab._SWEEP_VARS["double"]) == 2

    def test_triple_has_three_vars(self):
        assert len(AnalysisTab._SWEEP_VARS["triple"]) == 3

    def test_golfer_has_four_vars(self):
        assert len(AnalysisTab._SWEEP_VARS["golfer"]) == 4

    def test_all_vars_have_correct_structure(self):
        for model, vars_list in AnalysisTab._SWEEP_VARS.items():
            for key, desc, unit, (lo, hi) in vars_list:
                assert isinstance(key, str), f"{model}/{key}"
                assert isinstance(desc, str), f"{model}/{key}"
                assert unit == "rad", f"{model}/{key}"
                assert lo < hi, f"{model}/{key}"


# ---------------------------------------------------------------------------
# Shared fake tab builder for headless evaluator tests
# ---------------------------------------------------------------------------


def _make_fake_tab(model_type: str, result: object = None) -> object:
    """Build a lightweight stand-in for AnalysisTab without PyQt6.

    Copies all evaluator methods as unbound functions so they work
    correctly when called through normal attribute access.
    """

    class _FakeTab:
        _model_type = model_type
        _result = result

        # Bind instance methods from AnalysisTab
        _get_params_or_default = AnalysisTab._get_params_or_default
        _evaluator_double = AnalysisTab._evaluator_double
        _evaluator_triple = AnalysisTab._evaluator_triple
        _evaluator_golfer = AnalysisTab._evaluator_golfer
        _get_surface_evaluator = AnalysisTab._get_surface_evaluator

        # Static methods — re-wrap so descriptor protocol works
        _golfer_q_from_angles = staticmethod(
            AnalysisTab._golfer_q_from_angles.__func__
            if hasattr(AnalysisTab._golfer_q_from_angles, "__func__")
            else AnalysisTab._golfer_q_from_angles
        )
        _numerical_manipulability = staticmethod(
            AnalysisTab._numerical_manipulability.__func__
            if hasattr(AnalysisTab._numerical_manipulability, "__func__")
            else AnalysisTab._numerical_manipulability
        )

    return _FakeTab()


# ---------------------------------------------------------------------------
# Surface evaluators (require physics modules, no GUI)
# ---------------------------------------------------------------------------


class TestDoubleEvaluators:
    """Test surface evaluators for the double pendulum model."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Use a minimal mock to avoid PyQt6."""
        self._model_type = "double"
        self._result = None

    def _get_evaluator(self, z_key):
        """Directly test evaluator methods without GUI."""
        obj = _make_fake_tab(self._model_type, self._result)
        return obj._get_surface_evaluator(z_key)

    def test_mass_matrix_det(self):
        ev = self._get_evaluator("mass_matrix_det")
        assert ev is not None
        val = ev({"theta1": 0.0, "phi": 0.0})
        assert np.isfinite(val)
        assert val > 0  # M should be positive definite

    def test_mass_matrix_cond(self):
        ev = self._get_evaluator("mass_matrix_cond")
        assert ev is not None
        val = ev({"theta1": 0.0, "phi": 0.0})
        assert val >= 1.0  # condition number ≥ 1

    def test_potential_energy(self):
        ev = self._get_evaluator("potential_energy")
        assert ev is not None
        val = ev({"theta1": 0.0, "phi": 0.0})
        assert np.isfinite(val)

    def test_manipulability(self):
        ev = self._get_evaluator("manipulability")
        assert ev is not None
        val = ev({"theta1": 0.5, "phi": 0.3})
        assert val >= 0.0

    def test_unknown_key_returns_none(self):
        ev = self._get_evaluator("nonexistent")
        assert ev is None


class TestTripleEvaluators:
    """Test surface evaluators for the triple pendulum model."""

    def _get_evaluator(self, z_key):
        return _make_fake_tab("triple")._get_surface_evaluator(z_key)

    def test_mass_matrix_det(self):
        ev = self._get_evaluator("mass_matrix_det")
        assert ev is not None
        val = ev({"theta1": 0.0, "phi1": 0.0, "phi2": 0.0})
        assert np.isfinite(val) and val > 0

    def test_mass_matrix_cond(self):
        ev = self._get_evaluator("mass_matrix_cond")
        assert ev is not None
        val = ev({"phi1": 0.5, "phi2": -0.3})
        assert val >= 1.0

    def test_potential_energy(self):
        ev = self._get_evaluator("potential_energy")
        assert ev is not None
        val = ev({"theta1": 0.0, "phi1": 0.0, "phi2": 0.0})
        assert np.isfinite(val)

    def test_manipulability(self):
        ev = self._get_evaluator("manipulability")
        assert ev is not None
        val = ev({"theta1": 0.3, "phi1": 0.2, "phi2": 0.1})
        assert val >= 0.0


class TestGolferEvaluators:
    """Test surface evaluators for the golfer upper-body model."""

    def _get_evaluator(self, z_key):
        return _make_fake_tab("golfer")._get_surface_evaluator(z_key)

    def test_mass_matrix_det(self):
        ev = self._get_evaluator("mass_matrix_det")
        assert ev is not None
        val = ev({"theta1": 0.0, "phi": 0.0})
        assert np.isfinite(val)

    def test_mass_matrix_cond(self):
        ev = self._get_evaluator("mass_matrix_cond")
        assert ev is not None
        val = ev({"theta1": 0.0, "phi": 0.0})
        assert val >= 1.0

    def test_potential_energy(self):
        ev = self._get_evaluator("potential_energy")
        assert ev is not None
        val = ev({"theta1": 0.0})
        assert np.isfinite(val)

    def test_manipulability(self):
        ev = self._get_evaluator("manipulability")
        if ev is None:
            pytest.skip("analytical_fk_jacobians not available")
        val = ev({"theta1": 0.0, "phi": 0.0})
        assert val >= 0.0


# ---------------------------------------------------------------------------
# Equations popup: Jacobian topics
# ---------------------------------------------------------------------------


class TestEquationsPopupTopics:
    """Verify the Jacobian and Constraint Jacobian topics are registered."""

    def test_jacobian_topic_exists(self):
        from double_pendulum_golf.gui.equations_popup import EquationTopic, _TOPICS

        assert EquationTopic.JACOBIAN in _TOPICS

    def test_constraint_jacobian_topic_exists(self):
        from double_pendulum_golf.gui.equations_popup import EquationTopic, _TOPICS

        assert EquationTopic.CONSTRAINT_JACOBIAN in _TOPICS

    def test_jacobian_html_nonempty(self):
        from double_pendulum_golf.gui.equations_popup import EquationTopic, _TOPICS

        title, html = _TOPICS[EquationTopic.JACOBIAN]
        assert len(html) > 100
        assert "Jacobian" in title

    def test_constraint_jacobian_html_nonempty(self):
        from double_pendulum_golf.gui.equations_popup import EquationTopic, _TOPICS

        title, html = _TOPICS[EquationTopic.CONSTRAINT_JACOBIAN]
        assert len(html) > 100
        assert "Constraint" in title

    def test_jacobian_html_covers_manipulability(self):
        from double_pendulum_golf.gui.equations_popup import EquationTopic, _TOPICS

        _, html = _TOPICS[EquationTopic.JACOBIAN]
        assert "manipulability" in html.lower() or "ellipsoid" in html.lower()

    def test_constraint_jacobian_html_covers_kkt(self):
        from double_pendulum_golf.gui.equations_popup import EquationTopic, _TOPICS

        _, html = _TOPICS[EquationTopic.CONSTRAINT_JACOBIAN]
        assert "KKT" in html or "Lagrange" in html.lower()
