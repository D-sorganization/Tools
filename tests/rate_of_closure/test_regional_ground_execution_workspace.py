"""Qt interaction contracts for the imported regional-ground Ground Study."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from rate_of_closure.ui.pyqt6.main_window import (  # noqa: E402
    RateOfClosureStandaloneMainWindow,
)
from rate_of_closure.ui.pyqt6.regional_ground_execution_workspace import (  # noqa: E402
    RegionalGroundExecutionWorkspace,
)
from rate_of_closure.variation.regional_ground_variation_control import (  # noqa: E402
    GroundRegionalVariationHooks,
)
from rate_of_closure.web_authority.capability import (  # noqa: E402
    QUALIFIED_EXECUTION_CAPABILITY,
)
from tests.rate_of_closure.test_regional_ground_execution_result import (  # noqa: E402
    _job,
    _result,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_default_workspace_is_injection_safe_and_accessible(qtbot) -> None:  # type: ignore[no-untyped-def]
    workspace = RegionalGroundExecutionWorkspace()
    qtbot.addWidget(workspace)

    assert not workspace.run_button.isEnabled()
    assert not workspace.cancel_button.isEnabled()
    assert not workspace.save_result_button.isEnabled()
    assert "qualified" in workspace.run_button.toolTip().lower()
    assert workspace.open_button.accessibleName() == "Open execution job JSON"
    assert workspace.status_label.objectName() == "regionalGroundExecutionStatus"


def test_accept_confirm_run_and_retain_exact_result(qtbot) -> None:  # type: ignore[no-untyped-def]
    confirmations: list[object] = []

    def confirm(job):  # type: ignore[no-untyped-def]
        confirmations.append(job)
        return True

    def submitter(job, _hooks: GroundRegionalVariationHooks):  # type: ignore[no-untyped-def]
        return _result()

    workspace = RegionalGroundExecutionWorkspace(
        capability=QUALIFIED_EXECUTION_CAPABILITY,
        submitter=submitter,
        confirmation=confirm,
    )
    qtbot.addWidget(workspace)
    workspace.accept_job(_job(), source_name="job.json")

    assert workspace.run_button.isEnabled()
    assert workspace.current_result is None
    assert _job().qualified_plan_sha256 in workspace.job_label.text()
    workspace.run_imported_job()
    qtbot.waitUntil(lambda: not workspace.is_running, timeout=5_000)

    assert confirmations == [_job()]
    assert workspace.current_result == _result()
    assert workspace.save_result_button.isEnabled()
    assert workspace.export_csv_button.isEnabled()
    assert workspace.status_label.property("state") == "success"
    assert _result().dataset_sha256 in workspace.status_label.text()


def test_declined_confirmation_does_not_submit(qtbot) -> None:  # type: ignore[no-untyped-def]
    submissions: list[object] = []

    def submitter(job, _hooks):  # type: ignore[no-untyped-def]
        submissions.append(job)
        return _result()

    workspace = RegionalGroundExecutionWorkspace(
        capability=QUALIFIED_EXECUTION_CAPABILITY,
        submitter=submitter,
        confirmation=lambda _job: False,
    )
    qtbot.addWidget(workspace)
    workspace.accept_job(_job(), source_name="job.json")
    workspace.run_imported_job()

    assert submissions == []
    assert "no physics executed" in workspace.status_label.text().lower()


def test_new_job_acceptance_clears_retained_result(qtbot) -> None:  # type: ignore[no-untyped-def]
    workspace = RegionalGroundExecutionWorkspace(
        capability=QUALIFIED_EXECUTION_CAPABILITY,
        submitter=lambda _job, _hooks: _result(),
        confirmation=lambda _job: True,
    )
    qtbot.addWidget(workspace)
    workspace.accept_job(_job(), source_name="first.json")
    workspace.run_imported_job()
    qtbot.waitUntil(lambda: not workspace.is_running, timeout=5_000)
    assert workspace.current_result is not None

    workspace.accept_job(_job(), source_name="second.json")
    assert workspace.current_result is None
    assert not workspace.save_result_button.isEnabled()


def test_failed_rerun_preserves_prior_complete_result(qtbot) -> None:  # type: ignore[no-untyped-def]
    calls = 0

    def submitter(_job, _hooks):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 1:
            return _result()
        raise RuntimeError("private failure detail")

    workspace = RegionalGroundExecutionWorkspace(
        capability=QUALIFIED_EXECUTION_CAPABILITY,
        submitter=submitter,
        confirmation=lambda _job: True,
    )
    qtbot.addWidget(workspace)
    workspace.accept_job(_job(), source_name="job.json")
    workspace.run_imported_job()
    qtbot.waitUntil(lambda: not workspace.is_running, timeout=5_000)
    previous = workspace.current_result

    workspace.run_imported_job()
    qtbot.waitUntil(lambda: not workspace.is_running, timeout=5_000)

    assert workspace.current_result is previous
    assert "private failure detail" not in workspace.status_label.text()
    assert "prior complete result was preserved" in workspace.status_label.text()


def test_same_immutable_job_can_be_submitted_again_after_worker_release(
    qtbot: Any,
) -> None:
    submitted_job_hashes: list[str] = []

    def submitter(job, _hooks):  # type: ignore[no-untyped-def]
        submitted_job_hashes.append(job.job_sha256)
        return _result()

    workspace = RegionalGroundExecutionWorkspace(
        capability=QUALIFIED_EXECUTION_CAPABILITY,
        submitter=submitter,
        confirmation=lambda _job: True,
    )
    qtbot.addWidget(workspace)
    workspace.accept_job(_job(), source_name="job.json")

    workspace.run_imported_job()
    qtbot.waitUntil(lambda: not workspace.is_running, timeout=5_000)
    workspace.run_imported_job()
    qtbot.waitUntil(lambda: not workspace.is_running, timeout=5_000)

    assert submitted_job_hashes == [_job().job_sha256, _job().job_sha256]
    assert workspace.current_result == _result()


def test_submit_failure_is_stable_and_does_not_escape_ui_slot(qtbot) -> None:  # type: ignore[no-untyped-def]
    workspace = RegionalGroundExecutionWorkspace(
        capability=QUALIFIED_EXECUTION_CAPABILITY,
        submitter=lambda _job, _hooks: _result(),
        confirmation=lambda _job: True,
    )
    qtbot.addWidget(workspace)
    workspace.accept_job(_job(), source_name="job.json")

    assert workspace._controller is not None  # noqa: SLF001 - interaction seam

    def rejected_submit(_job) -> None:  # type: ignore[no-untyped-def]
        raise RuntimeError("C:/private/secret/model-input.json")

    workspace._controller.submit = rejected_submit  # noqa: SLF001
    workspace.run_imported_job()

    assert (
        workspace.status_label.text() == "Submission failed before execution started."
    )
    assert "private" not in workspace.status_label.text()
    assert workspace.run_button.isEnabled()


def test_confirmation_failure_is_stable_and_does_not_submit(qtbot) -> None:  # type: ignore[no-untyped-def]
    submissions: list[object] = []

    def bad_confirmation(_job):  # type: ignore[no-untyped-def]
        raise RuntimeError("C:/private/confirmation.log")

    def record_submission(job, _hooks):  # type: ignore[no-untyped-def]
        submissions.append(job)
        return _result()

    workspace = RegionalGroundExecutionWorkspace(
        capability=QUALIFIED_EXECUTION_CAPABILITY,
        submitter=record_submission,
        confirmation=bad_confirmation,
    )
    qtbot.addWidget(workspace)
    workspace.accept_job(_job(), source_name="job.json")
    workspace.run_imported_job()

    assert submissions == []
    assert workspace.status_label.text() == (
        "Run unavailable: executable authority or confirmation failed."
    )
    assert "private" not in workspace.status_label.text()


def test_open_and_save_failures_do_not_render_private_causes(
    qtbot: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = RegionalGroundExecutionWorkspace()
    qtbot.addWidget(workspace)

    monkeypatch.setattr(
        "rate_of_closure.ui.pyqt6.regional_ground_execution_workspace."
        "QFileDialog.getOpenFileName",
        lambda *_args: ("C:/private/secret.json", "JSON files (*.json)"),
    )
    workspace.open_job()
    assert "C:/private" not in workspace.status_label.text()
    assert "valid bounded" in workspace.status_label.text()

    workspace.accept_job(_job(), source_name="job.json")
    monkeypatch.setattr(
        "rate_of_closure.ui.pyqt6.regional_ground_execution_workspace."
        "QFileDialog.getSaveFileName",
        lambda *_args: ("Z:/private/secret.json", "JSON files (*.json)"),
    )
    monkeypatch.setattr(
        "rate_of_closure.ui.pyqt6.regional_ground_execution_workspace."
        "write_regional_ground_execution_job_atomic",
        lambda *_args: (_ for _ in ()).throw(OSError("Z:/private/secret.json")),
    )
    workspace.save_job_as()
    assert "Z:/private" not in workspace.status_label.text()
    assert "written atomically" in workspace.status_label.text()


def test_capability_and_submitter_must_agree() -> None:
    with pytest.raises(ValueError, match="must agree"):
        RegionalGroundExecutionWorkspace(
            capability=QUALIFIED_EXECUTION_CAPABILITY,
            submitter=None,
        )


def test_source_standalone_window_injects_direct_qualified_runner(
    qtbot: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr("sys.frozen", raising=False)
    window = RateOfClosureStandaloneMainWindow()
    qtbot.addWidget(window)

    workspace = window._regional_ground_execution_tab
    assert workspace._capability.regional_ground_execution
    assert workspace._controller is not None


def test_source_standalone_window_executes_one_exact_profile_job(
    qtbot: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr("sys.frozen", raising=False)
    window = RateOfClosureStandaloneMainWindow()
    qtbot.addWidget(window)
    workspace = window._regional_ground_execution_tab
    workspace._confirmation = lambda _job: True
    workspace.accept_job(_job(), source_name="profile-qualified.json")

    workspace.run_imported_job()
    qtbot.waitUntil(lambda: not workspace.is_running, timeout=30_000)

    result = workspace.current_result
    assert result is not None
    result.assert_matches_job(_job())
    assert workspace.status_label.property("state") == "success"


def test_frozen_standalone_window_remains_explicitly_unavailable(
    qtbot: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.frozen", True, raising=False)
    window = RateOfClosureStandaloneMainWindow()
    qtbot.addWidget(window)

    workspace = window._regional_ground_execution_tab
    assert not workspace._capability.regional_ground_execution
    assert workspace._controller is None
