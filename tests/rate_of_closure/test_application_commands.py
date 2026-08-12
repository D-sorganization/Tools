"""UI-neutral application-command contract tests for Tools #4219."""

from __future__ import annotations

import pytest

from rate_of_closure.application.commands import (
    APP_COMMAND_IDS,
    AppCommandId,
    CommandAvailability,
    CommandUnavailableError,
)


def test_command_ids_are_stable_unique_wire_values() -> None:
    expected = {
        "file.new_workspace",
        "file.open_workspace",
        "file.open_recent_workspace",
        "file.save_workspace",
        "file.save_workspace_as",
        "file.import_workspace",
        "file.export_workspace",
        "file.close_workspace",
        "file.open_regional_ground_variation_request",
        "file.save_regional_ground_variation_request_as",
        "file.open_regional_ground_execution_job",
        "file.save_regional_ground_execution_job_as",
        "file.save_regional_ground_execution_result_as",
        "file.export_regional_ground_execution_rows_csv",
        "view.manage_modules",
        "view.restore_default_workspace",
        "view.show_impact",
        "view.show_swing",
        "view.show_flight",
        "global.open_glossary",
        "global.toggle_theme",
        "global.show_shortcuts",
        "global.open_current_module_help",
    }

    assert {command.value for command in APP_COMMAND_IDS} == expected
    assert len(APP_COMMAND_IDS) == len(set(APP_COMMAND_IDS))
    assert all(AppCommandId(command.value) is command for command in APP_COMMAND_IDS)


def test_available_command_has_no_disabled_reason() -> None:
    availability = CommandAvailability.available()

    assert availability.enabled is True
    assert availability.disabled_reason is None
    availability.require_enabled(AppCommandId.FILE_SAVE_WORKSPACE)


def test_disabled_command_requires_and_reports_a_reason() -> None:
    availability = CommandAvailability.disabled("Open or create a workspace first.")

    assert availability.enabled is False
    assert availability.disabled_reason == "Open or create a workspace first."
    with pytest.raises(CommandUnavailableError, match="file.save_workspace.*Open"):
        availability.require_enabled(AppCommandId.FILE_SAVE_WORKSPACE)


@pytest.mark.parametrize(
    ("enabled", "reason"),
    [(True, "unexpected"), (False, None), (False, ""), (False, "   ")],
)
def test_availability_rejects_ambiguous_states(
    enabled: bool, reason: str | None
) -> None:
    with pytest.raises(ValueError):
        CommandAvailability(enabled=enabled, disabled_reason=reason)
