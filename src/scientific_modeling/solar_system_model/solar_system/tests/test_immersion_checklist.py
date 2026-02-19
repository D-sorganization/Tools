"""Tests for immersive visualization checklist requirements."""

import pytest

from solar_system.ui.widgets import ImmersionChecklistPanel


@pytest.fixture()
def checklist_panel() -> ImmersionChecklistPanel:
    return ImmersionChecklistPanel()


def test_default_tasks_initialized(checklist_panel: ImmersionChecklistPanel) -> None:
    data = checklist_panel.get_render_data()
    completed, total = data["progress"]

    assert total >= 5
    assert completed == 0
    assert data["visible"] is True
    assert all("title" in task for task in data["tasks"])


def test_mark_complete_updates_progress(
    checklist_panel: ImmersionChecklistPanel,
) -> None:
    checklist_panel.mark_complete("select_body")
    checklist_panel.mark_complete("plan_transfer")
    completed, total = checklist_panel.get_progress()

    assert completed == 2
    assert total >= 5


def test_toggle_visibility(checklist_panel: ImmersionChecklistPanel) -> None:
    checklist_panel.toggle()
    assert checklist_panel.visible is False

    checklist_panel.toggle()
    assert checklist_panel.visible is True
