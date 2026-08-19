"""Matched non-executing PyQt presentation contracts."""

from __future__ import annotations

import pytest

from rate_of_closure.application.regional_ground_execution_presentation import (
    RegionalGroundExecutionPresentation,
    RegionalGroundPresentationState,
)
from rate_of_closure.ui.pyqt6.regional_ground_execution_controller import (
    RegionalGroundExecutionController,
)
from rate_of_closure.ui.pyqt6.regional_ground_execution_presentation import (
    RegionalGroundExecutionPresentationPanel,
)
from rate_of_closure.variation.regional_ground_variation_control import (
    GroundRegionalVariationCancelled,
    GroundRegionalVariationFailed,
    GroundRegionalVariationFailureStage,
    GroundRegionalVariationProgress,
)
from rate_of_closure.web_authority.capability import DEFAULT_UNAVAILABLE_CAPABILITY
from tests.rate_of_closure.test_regional_ground_authority_jobs import _job, _result

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_summary_and_disabled_explanation_are_exact_and_compact() -> None:
    job = _job()
    presentation = RegionalGroundExecutionPresentation.initial(
        job, DEFAULT_UNAVAILABLE_CAPABILITY
    )

    assert presentation.summary.to_wire() == {
        "schema_version": "rate-of-closure/regional-ground-execution-job/v1",
        "model_id": job.flight.model_id,
        "model_version": job.flight.model_version,
        "producer": job.provenance.producer,
        "producer_version": job.provenance.producer_version,
        "source_revision": job.provenance.source_revision,
        "input_sha256": job.input_sha256,
    }
    assert presentation.execution_enabled is False
    assert presentation.disabled_reason_code == "execution_profile_unqualified"
    assert presentation.disabled_detail == DEFAULT_UNAVAILABLE_CAPABILITY.detail
    assert presentation.state is RegionalGroundPresentationState.IDLE


def test_progress_cancel_failure_and_result_states_preserve_exact_evidence() -> None:
    job = _job()
    base = RegionalGroundExecutionPresentation.initial(
        job, DEFAULT_UNAVAILABLE_CAPABILITY
    )
    running = base.with_progress(GroundRegionalVariationProgress(2, 4))
    requested = running.with_cancel_requested()
    cancelled = requested.with_cancelled(GroundRegionalVariationCancelled(2, 4))
    failed = running.with_failure(
        GroundRegionalVariationFailed(
            GroundRegionalVariationFailureStage.PREFLIGHT,
            2,
            4,
            RuntimeError("must not be presented"),
        )
    )
    succeeded = running.with_result(_result(job))

    assert running.to_wire()["status"] == "running"
    assert requested.to_wire()["status"] == "cancel_requested"
    assert cancelled.to_wire()["status"] == "cancelled"
    assert failed.to_wire()["failure_stage"] == "preflight"
    assert "must not be presented" not in str(failed.to_wire())
    assert succeeded.to_wire()["status"] == "succeeded"
    assert succeeded.to_wire()["result_schema_version"] == (
        "rate-of-closure/regional-ground-execution-result/v1"
    )
    assert succeeded.to_wire()["result_sha256"] == _result(job).dataset_sha256


def test_presentation_rejects_stale_or_cross_job_controller_evidence() -> None:
    job = _job()
    base = RegionalGroundExecutionPresentation.initial(
        job, DEFAULT_UNAVAILABLE_CAPABILITY
    )
    running = base.with_progress(GroundRegionalVariationProgress(2, 4))

    with pytest.raises(ValueError, match="monotonic"):
        running.with_progress(GroundRegionalVariationProgress(1, 4))
    with pytest.raises(ValueError, match="match the job"):
        running.with_cancelled(GroundRegionalVariationCancelled(2, 5))


def test_panel_is_read_only_disabled_and_presents_state(qtbot) -> None:
    job = _job()
    panel = RegionalGroundExecutionPresentationPanel(
        job, DEFAULT_UNAVAILABLE_CAPABILITY
    )
    qtbot.addWidget(panel)

    assert not panel.run_button.isEnabled()
    assert not panel.cancel_button.isEnabled()
    assert DEFAULT_UNAVAILABLE_CAPABILITY.detail in panel.disabled_label.text()
    assert job.flight.model_id in panel.summary_label.text()

    controller = RegionalGroundExecutionController(
        lambda submitted, _hooks: _result(submitted)
    )
    panel.bind_controller(controller)
    controller.progressed.emit(GroundRegionalVariationProgress(1, 4))
    assert panel.status_label.text() == "Running — 1 / 4 accepted trials"
    controller.failed.emit(
        GroundRegionalVariationFailed(
            GroundRegionalVariationFailureStage.PREFLIGHT,
            1,
            4,
            RuntimeError("private cause"),
        )
    )
    assert panel.status_label.text() == "Failed (preflight) — 1 / 4 accepted trials"
    assert "private cause" not in panel.status_label.text()
