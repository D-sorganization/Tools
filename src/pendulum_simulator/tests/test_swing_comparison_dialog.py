"""
Unit tests for SwingComparisonDialog and PerturbationPanel.set_preset_source().

Uses pytest-qt (QApplication via conftest) with a minimal headless
environment.  Does NOT run a real simulation — all callbacks are stubs.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PRESET_NAMES = ["Preset A", "Preset B", "Preset C"]
_COEFFS: dict[str, list[list[float]]] = {
    "Preset A": [[-25.0, 10.0], [0.0]],
    "Preset B": [[-30.0, 12.0], [0.0]],
    "Preset C": [[0.0], [0.0]],
}


def _stub_simulate(coeffs):
    return object()


def _stub_extract(result):
    return {
        "tip_speed_final": 30.0,
        "tip_position_final": np.array([0.0, 0.0]),
    }


# ---------------------------------------------------------------------------
# PerturbationPanel.set_preset_source
# ---------------------------------------------------------------------------


class TestSetPresetSource:
    @pytest.fixture
    def panel(self, app):  # noqa: ARG002
        from double_pendulum_golf.gui.perturbation_panel import PerturbationPanel

        p = PerturbationPanel()
        p.set_simulation_callbacks(_stub_simulate, _stub_extract)
        return p

    def test_compare_btn_disabled_without_preset_source(self, panel) -> None:
        assert not panel._compare_btn.isEnabled()

    def test_compare_btn_enabled_after_set_preset_source(self, panel) -> None:
        panel.set_preset_source(
            lambda: _PRESET_NAMES,
            lambda name: _COEFFS.get(name, [[0.0]]),
        )
        assert panel._compare_btn.isEnabled()

    def test_rejects_non_callable_names_fn(self, panel) -> None:
        with pytest.raises(AssertionError):
            panel.set_preset_source("not_callable", lambda name: [])  # type: ignore[arg-type]

    def test_rejects_non_callable_coeffs_fn(self, panel) -> None:
        with pytest.raises(AssertionError):
            panel.set_preset_source(lambda: [], "not_callable")  # type: ignore[arg-type]

    def test_stores_preset_fns(self, panel) -> None:
        def names_fn() -> list[str]:
            return _PRESET_NAMES

        def coeffs_fn(name: str) -> list[list[float]]:
            return _COEFFS.get(name, [[0.0]])

        panel.set_preset_source(names_fn, coeffs_fn)
        assert panel._get_preset_names_fn is names_fn
        assert panel._get_coeffs_for_preset_fn is coeffs_fn


# ---------------------------------------------------------------------------
# SwingComparisonDialog construction
# ---------------------------------------------------------------------------


class TestSwingComparisonDialogConstruction:
    @pytest.fixture
    def dialog(self, app):  # noqa: ARG002
        from double_pendulum_golf.gui.swing_comparison_dialog import (
            SwingComparisonDialog,
        )

        return SwingComparisonDialog(
            preset_names=_PRESET_NAMES,
            get_coeffs_for_preset=lambda name: _COEFFS.get(name, [[0.0]]),
            simulate_fn=_stub_simulate,
            extract_fn=_stub_extract,
        )

    def test_creates_without_error(self, dialog) -> None:
        assert dialog is not None

    def test_preset_list_has_all_presets(self, dialog) -> None:
        assert dialog._preset_list.count() == len(_PRESET_NAMES)

    def test_first_two_presets_selected_by_default(self, dialog) -> None:
        selected = [
            dialog._preset_list.item(i).text()
            for i in range(dialog._preset_list.count())
            if dialog._preset_list.item(i) and dialog._preset_list.item(i).isSelected()
        ]
        assert len(selected) == 2
        assert selected[0] == _PRESET_NAMES[0]
        assert selected[1] == _PRESET_NAMES[1]

    def test_run_button_present(self, dialog) -> None:
        assert dialog._run_btn is not None

    def test_cancel_button_initially_disabled(self, dialog) -> None:
        assert not dialog._cancel_btn.isEnabled()

    def test_amp_spin_default(self, dialog) -> None:
        assert dialog._amp_spin.value() == 0.1

    def test_trials_spin_default(self, dialog) -> None:
        assert dialog._trials_spin.value() == 30

    def test_export_button_initially_disabled(self, dialog) -> None:
        assert not dialog._export_btn.isEnabled()


# ---------------------------------------------------------------------------
# SwingComparisonDialog Contracts
# ---------------------------------------------------------------------------


class TestSwingComparisonDialogContracts:
    def test_init_rejects_single_preset(self, app) -> None:  # noqa: ARG002
        from double_pendulum_golf.gui.swing_comparison_dialog import (
            SwingComparisonDialog,
        )

        with pytest.raises(AssertionError, match="Need at least 2 presets"):
            SwingComparisonDialog(
                preset_names=["Only One"],
                get_coeffs_for_preset=lambda name: [[0.0]],
                simulate_fn=_stub_simulate,
                extract_fn=_stub_extract,
            )
