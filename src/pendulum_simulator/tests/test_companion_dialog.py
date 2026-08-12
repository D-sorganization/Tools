"""Headless PyQt contracts for the proximal-distal companion guide."""

from __future__ import annotations


def test_companion_dialog_exposes_workflow_tips_and_glossary(qapp) -> None:
    from double_pendulum_golf.gui.companion_dialog import CompanionGuideDialog

    dialog = CompanionGuideDialog()

    assert dialog.windowTitle() == "Proximal–Distal Companion Guide"
    assert dialog.experiment_list.count() >= 6
    assert "Workflow" in dialog.experiment_details.toPlainText()
    assert "Tips" in dialog.experiment_details.toPlainText()
    assert dialog.glossary_list.count() >= 12


def test_companion_dialog_filters_glossary(qapp) -> None:
    from double_pendulum_golf.gui.companion_dialog import CompanionGuideDialog

    dialog = CompanionGuideDialog()
    dialog.glossary_search.setText("ZTCF")

    assert dialog.glossary_list.count() >= 1
    assert "zero-torque" in dialog.glossary_details.toPlainText().lower()


def test_toolstrip_makes_companion_discoverable(qapp) -> None:
    from double_pendulum_golf.gui.toolstrip_widget import ToolStrip

    toolstrip = ToolStrip()

    assert toolstrip.btn_companion.text() == "Companion Guide"
    assert "glossary" in toolstrip.btn_companion.toolTip().lower()
