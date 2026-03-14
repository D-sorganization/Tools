"""
Unit tests for the PerturbationPanel GUI widget.

Uses pytest-qt (QApplication via conftest) with a minimal headless
environment.  Does NOT run a real simulation — all callbacks are stubs
so the tests remain fast and Qt-free of real ODE integration.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    from PyQt6.QtWidgets import QApplication

    _HAS_QT = True
except ImportError:
    _HAS_QT = False


pytestmark = pytest.mark.skipif(not _HAS_QT, reason="PyQt6 not available")


@pytest.fixture(scope="module")
def app():
    """Headless QApplication fixture (shared across module)."""
    import sys

    existing = QApplication.instance()
    if existing is not None:
        yield existing
        return
    a = QApplication(sys.argv[:1])
    yield a
    # Do NOT call a.quit() — other tests may still need it


@pytest.fixture
def panel(app):  # noqa: ARG001 — app fixture ensures QApplication exists
    from double_pendulum_golf.gui.perturbation_panel import PerturbationPanel

    return PerturbationPanel()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestPerturbationPanelConstruction:
    def test_creates_without_error(self, panel) -> None:
        assert panel is not None

    def test_run_button_initially_disabled(self, panel) -> None:
        assert not panel._run_btn.isEnabled()

    def test_cancel_button_initially_disabled(self, panel) -> None:
        assert not panel._cancel_btn.isEnabled()

    def test_noise_combo_has_three_items(self, panel) -> None:
        assert panel._noise_combo.count() == 3
        items = [panel._noise_combo.itemText(i) for i in range(3)]
        assert set(items) == {"white", "pink", "brown"}

    def test_default_trials(self, panel) -> None:
        assert panel._trials_spin.value() == 50

    def test_default_amplitude(self, panel) -> None:
        assert abs(panel._amp_spin.value() - 0.1) < 1e-9


# ---------------------------------------------------------------------------
# set_simulation_callbacks
# ---------------------------------------------------------------------------


class TestSetSimulationCallbacks:
    def test_enables_run_button(self, panel) -> None:
        panel.set_simulation_callbacks(
            lambda c: {},
            lambda r: {
                "tip_speed_final": 1.0,
                "tip_position_final": np.array([0.0, 0.0]),
            },
        )
        assert panel._run_btn.isEnabled()

    def test_rejects_none_simulate_fn(self, panel) -> None:
        with pytest.raises(AssertionError):
            panel.set_simulation_callbacks(None, lambda r: {})

    def test_rejects_none_extract_fn(self, panel) -> None:
        with pytest.raises(AssertionError):
            panel.set_simulation_callbacks(lambda c: {}, None)


# ---------------------------------------------------------------------------
# set_coeffs_source
# ---------------------------------------------------------------------------


class TestSetCoeffsSource:
    def test_accepts_callable(self, panel) -> None:
        panel.set_coeffs_source(lambda: [[1.0, 2.0]])
        assert panel._get_coeffs_fn is not None

    def test_rejects_non_callable(self, panel) -> None:
        with pytest.raises(AssertionError):
            panel.set_coeffs_source("not a function")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _display_summary
# ---------------------------------------------------------------------------


class TestDisplaySummary:
    def test_populates_labels(self, panel) -> None:
        summary = {
            "tip_speed_mean": 30.5,
            "tip_speed_std": 1.2,
            "tip_speed_cv": 0.039,
            "tip_speed_min": 28.1,
            "tip_speed_max": 32.9,
            "n_trials": 20,
        }
        panel._display_summary(summary)
        assert "30.500" in panel._result_labels["Mean"].text()
        assert "1.200" in panel._result_labels["Std"].text()
        assert "20" in panel._result_labels["Trials"].text()

    def test_clear_resets_to_dash(self, panel) -> None:
        panel._clear_results()
        for lbl in panel._result_labels.values():
            assert lbl.text() == "—"
