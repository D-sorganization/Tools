"""Explicit PyQt paired-study controls, lifecycle, and archive loading."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from PyQt6.QtWidgets import QDialog  # noqa: E402

from rate_of_closure.model import ImpactScenario  # noqa: E402
from rate_of_closure.ui.pyqt6.variation_tab import VariationTab  # noqa: E402
from rate_of_closure.variation.localized_attribution_producer import (  # noqa: E402
    produce_localized_attribution,
)

from .test_localized_attribution_gui import _authority  # noqa: E402
from .test_localized_attribution_producer import _design  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture
def tab(qtbot):  # type: ignore[no-untyped-def]
    widget = VariationTab()
    qtbot.addWidget(widget)
    design = _design()
    widget.set_simulation_config(design.base_config)
    widget.load_plan(design.source_plan)
    yield widget
    widget.stop()


def test_separate_controls_are_explicit_accessible_and_available(
    tab: VariationTab,
) -> None:
    view = tab._localized_attribution

    assert view._configure_run.text() == "Configure & Run Separate Paired Study…"
    assert view._configure_run.isEnabled()
    assert "does not reuse Monte Carlo scatter" in view._configure_run.toolTip()
    assert view._cancel_study.accessibleName() == "Cancel separate paired study"
    assert not view._cancel_study.isEnabled()
    assert view._load_authority.isEnabled()


def test_normal_monte_carlo_never_constructs_paired_worker(
    qtbot, tab: VariationTab
) -> None:  # type: ignore[no-untyped-def]
    constructed: list[object] = []
    original = tab._attribution_worker_factory
    tab._attribution_worker_factory = lambda design: constructed.append(design)  # type: ignore[assignment,return-value]
    tab._sens_check.setChecked(False)

    tab._on_run()
    assert tab._worker is not None
    with qtbot.waitSignal(tab._worker.finished, timeout=60_000):
        pass

    tab._attribution_worker_factory = original
    assert constructed == []


def test_explicit_confirmed_dialog_runs_exact_pair_and_loads_live_authority(
    qtbot, tab: VariationTab, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    design = _design()

    class AcceptedDialog:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return int(QDialog.DialogCode.Accepted)

        def build_design(self):  # type: ignore[no-untyped-def]
            return design

    monkeypatch.setattr(
        "rate_of_closure.ui.pyqt6.variation_attribution_controller."
        "LocalizedAttributionRunDialog",
        AcceptedDialog,
    )

    tab._on_configure_attribution()
    worker = tab._attribution_worker
    assert worker is not None and worker.total_runs == 2
    with qtbot.waitSignal(worker.finished, timeout=60_000):
        pass

    assert tab.attribution_production() is not None
    assert tab._localized_attribution.authority() is not None
    assert tab._localized_attribution._study_progress.maximum() == 2
    assert tab._localized_attribution._study_progress.value() == 2
    assert "2 explicit trials" in tab._localized_attribution._study_status.text()
    assert "no causal inference" in tab._localized_attribution._study_status.text()


def test_stale_failure_and_cancel_callbacks_do_not_replace_prior_authority(
    tab: VariationTab,
) -> None:
    prior = _authority()
    tab._localized_attribution.set_authority(prior)
    generation = tab._attribution_generation

    tab._accept_attribution_failed(generation - 1, "stale")
    tab._accept_attribution_cancelled(generation)
    assert tab._localized_attribution.authority() == prior
    assert "Prior paired authority was not replaced" in (
        tab._localized_attribution._study_status.text()
    )


def test_stale_success_is_ignored_and_plan_or_config_change_invalidates(
    tab: VariationTab,
) -> None:
    prior = _authority()
    production = produce_localized_attribution(_design())
    tab._localized_attribution.set_authority(prior)
    generation = tab._attribution_generation

    tab._accept_attribution_succeeded(generation - 1, production)
    assert tab._localized_attribution.authority() == prior

    tab._runs_spin.setValue(tab._runs_spin.value() + 1)
    assert tab._localized_attribution.authority() is None

    tab._localized_attribution.set_authority(prior)
    tab.set_simulation_config(
        replace(tab._base_simulation_config, swing_duration_s=0.1)
    )
    assert tab._localized_attribution.authority() is None


def test_mode_or_unavailable_simulation_fail_closed_and_clear_authority(
    tab: VariationTab,
) -> None:
    tab._localized_attribution.set_authority(_authority())
    tab._mode_combo.setCurrentIndex(tab._mode_combo.findData("launch"))
    assert tab._localized_attribution.authority() is None
    assert not tab._localized_attribution._configure_run.isEnabled()

    tab._localized_attribution.set_authority(_authority())
    tab.set_simulation_unavailable("Simulation request is invalid.")
    assert tab._localized_attribution.authority() is None
    assert not tab._localized_attribution._configure_run.isEnabled()
    assert tab._localized_attribution._study_status.text() == (
        "Simulation request is invalid."
    )


def test_explorer_scenario_change_cancels_and_rejects_late_success(
    tab: VariationTab,
) -> None:
    prior = _authority()
    production = produce_localized_attribution(_design())
    tab._base_combo.setCurrentIndex(1)
    tab._localized_attribution.set_authority(prior)
    tab._attribution_production = production
    generation = tab._attribution_generation
    calls: list[str] = []

    class RunningWorker:
        def isRunning(self) -> bool:
            return True

        def cancel(self) -> None:
            calls.append("cancel")

        def wait(self, _timeout: int) -> bool:
            return True

    tab._attribution_worker = RunningWorker()  # type: ignore[assignment]
    tab.set_scenario(ImpactScenario(clubhead_speed_mph=113.0, impact_offset_toe_mm=1.0))

    assert calls == ["cancel"]
    assert tab._attribution_generation == generation + 1
    assert tab.attribution_production() is None
    assert tab._localized_attribution.authority() is None
    assert (
        "Explorer scenario changed" in tab._localized_attribution._study_status.text()
    )

    tab._accept_attribution_succeeded(generation, production)
    assert tab.attribution_production() is None
    assert tab._localized_attribution.authority() is None
    tab._attribution_worker = None

    tab._localized_attribution.set_authority(prior)
    tab._attribution_production = production
    tab.set_scenario(ImpactScenario(clubhead_speed_mph=113.0, impact_offset_toe_mm=2.0))
    assert tab.attribution_production() is None
    assert tab._localized_attribution.authority() is None


def test_scenario_change_does_not_invalidate_registry_based_plan(
    tab: VariationTab,
) -> None:
    prior = _authority()
    production = produce_localized_attribution(_design())
    assert tab._base_combo.currentIndex() == 0
    tab._localized_attribution.set_authority(prior)
    tab._attribution_production = production
    generation = tab._attribution_generation

    tab.set_scenario(ImpactScenario(clubhead_speed_mph=114.0))

    assert tab._attribution_generation == generation
    assert tab.attribution_production() is production
    assert tab._localized_attribution.authority() == prior


def test_explorer_scenario_change_ignores_fields_outside_plan(
    tab: VariationTab,
) -> None:
    prior = _authority()
    tab._base_combo.setCurrentIndex(1)
    tab._localized_attribution.set_authority(prior)
    generation = tab._attribution_generation

    tab.set_scenario(ImpactScenario(clubhead_speed_mph=113.0, omega_plane_dps=1800.0))

    assert tab._attribution_generation == generation
    assert tab._localized_attribution.authority() == prior


def test_stop_cancels_and_joins_separate_worker(tab: VariationTab) -> None:
    calls: list[str] = []

    class RunningWorker:
        def cancel(self) -> None:
            calls.append("cancel")

        def wait(self, timeout: int) -> bool:
            calls.append(f"wait:{timeout}")
            return True

    tab._attribution_worker = RunningWorker()  # type: ignore[assignment]
    tab.stop()

    assert calls == ["cancel", "wait:10000"]


def test_archive_load_is_atomic_and_visibly_unverified(
    tab: VariationTab, tmp_path: Path, monkeypatch
) -> None:
    prior = _authority()
    tab._localized_attribution.set_authority(prior)
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "rate_of_closure.ui.pyqt6.variation_attribution_controller."
        "QFileDialog.getOpenFileName",
        staticmethod(lambda *_args, **_kwargs: (str(invalid), "JSON (*.json)")),
    )

    tab._on_load_attribution_authority()
    assert tab._localized_attribution.authority() == prior
    assert "Cannot load paired authority" in (
        tab._localized_attribution._study_status.text()
    )

    valid = tmp_path / "valid.json"
    fixture = (
        Path(__file__).parent / "fixtures" / "localized_attribution_authority_v1.json"
    )
    valid.write_text(
        json.dumps(json.loads(fixture.read_text("utf-8"))),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "rate_of_closure.ui.pyqt6.variation_attribution_controller."
        "QFileDialog.getOpenFileName",
        staticmethod(lambda *_args, **_kwargs: (str(valid), "JSON (*.json)")),
    )
    tab._on_load_attribution_authority()

    assert tab._localized_attribution.authority() == prior
    assert tab.attribution_production() is None
    assert "not rerun or provenance-verified" in (
        tab._localized_attribution._study_status.text()
    )


def test_archive_save_failure_is_visible_and_preserves_authority(
    tab: VariationTab, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prior = _authority()
    tab._localized_attribution.set_authority(prior)
    destination = tmp_path / "authority.json"
    monkeypatch.setattr(
        "rate_of_closure.ui.pyqt6.variation_attribution_controller."
        "QFileDialog.getSaveFileName",
        staticmethod(lambda *_args, **_kwargs: (str(destination), "JSON (*.json)")),
    )
    monkeypatch.setattr(
        "rate_of_closure.ui.pyqt6.variation_attribution_controller."
        "write_authority_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("byte cap")),
    )

    tab._on_save_attribution_authority()

    assert tab._localized_attribution.authority() == prior
    assert tab._localized_attribution._study_status.text() == (
        "Cannot save paired authority: byte cap"
    )
    assert not destination.exists()
