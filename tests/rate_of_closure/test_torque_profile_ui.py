"""PyQt contracts for prescribed-input profile authoring (issue #4136)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QPushButton  # noqa: E402

from rate_of_closure.ui.pyqt6.simulation_tab import SimulationTab  # noqa: E402
from rate_of_closure.ui.pyqt6.torque_profile_controller import (  # noqa: E402
    ProfileDraft,
    RunMode,
    TorqueProfileLibraryAdapter,
)
from rate_of_closure.ui.pyqt6.torque_profile_panel import (  # noqa: E402
    TorquePolynomialDialog,
    TorqueProfilePanel,
)
from shared.python.signal_toolkit.polynomial_generator import (  # noqa: E402
    PolynomialGeneratorWidget,
)
from shared.python.swing_sim.torque_profiles import (  # noqa: E402
    COEFFICIENT_ORDER,
    TORQUE_PROFILE_SCHEMA_VERSION,
    TORQUE_UNIT,
    PrescribedTorqueProfile,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture
def panel(qtbot):  # type: ignore[no-untyped-def]
    widget = TorqueProfilePanel()
    qtbot.addWidget(widget)
    return widget


def _draft() -> ProfileDraft:
    return ProfileDraft(
        profile_id="profile.rate_of_closure.driver.v1",
        model_id="model.double_pendulum.v1",
        name="Driver Release",
        description="Prescribed shoulder and wrist torque for a driver swing.",
        time_domain_s=(0.0, 1.25),
    )


class TestTorqueProfileLibraryAdapter:
    def test_assignment_builds_the_canonical_schema(self) -> None:
        library = TorqueProfileLibraryAdapter()
        profile = library.assign(_draft(), "joint.shoulder", [10.0, -2.0])

        payload = profile.to_json_dict()
        assert payload["schema_version"] == TORQUE_PROFILE_SCHEMA_VERSION
        assert payload["torque_unit"] == TORQUE_UNIT
        assert payload["coefficient_order"] == COEFFICIENT_ORDER
        assert profile.evaluate(0.5) == pytest.approx({"joint.shoulder": 9.0})

    def test_reassignment_preserves_other_joints_and_creation_time(self) -> None:
        library = TorqueProfileLibraryAdapter()
        first = library.assign(_draft(), "joint.shoulder", [10.0])
        second = library.assign(_draft(), "joint.wrist", [0.0, 3.0])

        assert second.created_at_utc == first.created_at_utc
        assert [item.joint_id for item in second.assignments] == [
            "joint.shoulder",
            "joint.wrist",
        ]

    def test_directory_library_and_single_profile_round_trip(
        self, tmp_path: Path
    ) -> None:
        library = TorqueProfileLibraryAdapter()
        original = library.assign(_draft(), "joint.shoulder", [10.0, -2.0])
        library_dir = tmp_path / "library"
        export_path = tmp_path / "driver.json"

        library.save_library(library_dir)
        library.export_profile(original.profile_id, export_path)

        loaded = TorqueProfileLibraryAdapter()
        assert loaded.load_library(library_dir) == 1
        imported = loaded.import_profile(export_path)
        assert imported == PrescribedTorqueProfile.loads(export_path.read_text())
        assert loaded.active_profile() == original


class TestTorqueProfilePanel:
    def test_run_mode_defaults_to_current_execution(self, panel) -> None:  # type: ignore[no-untyped-def]
        assert panel.selection().mode is RunMode.OPTIMIZED_DEFAULT
        assert panel._run_mode_combo.currentText() == "Default / Solver-Configured"
        assert "selected simulator" in panel._mode_description.text().lower()

        panel._run_mode_combo.setCurrentIndex(1)

        selection = panel.selection()
        assert selection.mode is RunMode.PRESCRIBED_TORQUE
        assert selection.execution_ready is False
        assert "dynamics kernel" in panel._mode_description.text().lower()
        assert "author or load" in selection.validation_message.lower()

    def test_joint_assignment_buttons_are_obvious(self, panel) -> None:  # type: ignore[no-untyped-def]
        assert set(panel.assignment_buttons()) == {"joint.shoulder", "joint.wrist"}
        for joint_id, button in panel.assignment_buttons().items():
            assert isinstance(button, QPushButton)
            assert "Assign" in button.text()
            assert joint_id in button.toolTip()
            assert button.cursor().shape() == Qt.CursorShape.PointingHandCursor

        panel._model_combo.setCurrentIndex(1)
        assert set(panel.assignment_buttons()) == {
            "joint.shoulder",
            "joint.wrist",
            "joint.club",
        }

    def test_generated_coefficients_update_summary(self, panel) -> None:  # type: ignore[no-untyped-def]
        panel._profile_id_edit.setText("profile.rate_of_closure.driver.v1")
        panel._name_edit.setText("Driver Release")
        panel._description_edit.setText("A prescribed driver torque profile.")

        panel.accept_polynomial("joint.shoulder", [10.0, -2.0])

        profile = panel.selection().profile
        assert profile is not None
        assert profile.assignments[0].polynomial.coefficients == (10.0, -2.0)
        assert "10" in panel.assignment_status("joint.shoulder")

    def test_library_actions_are_visible_clickable_and_described(self, panel) -> None:  # type: ignore[no-untyped-def]
        actions = panel.library_action_buttons()
        assert set(actions) == {"save", "load", "import", "export"}
        for button in actions.values():
            assert button.isEnabled()
            assert button.toolTip()
            assert button.cursor().shape() == Qt.CursorShape.PointingHandCursor


def test_editor_reuses_shared_polynomial_generator(qtbot) -> None:  # type: ignore[no-untyped-def]
    dialog = TorquePolynomialDialog("joint.shoulder")
    qtbot.addWidget(dialog)

    generators = dialog.findChildren(PolynomialGeneratorWidget)
    assert len(generators) == 1
    assert generators[0].joint_combo.currentText() == "joint.shoulder"


def test_complete_prescribed_profile_executes_in_simulation(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = SimulationTab()
    qtbot.addWidget(tab)
    panel = tab._torque_profile_panel
    panel.accept_polynomial("joint.shoulder", [5.0])
    panel.accept_polynomial("joint.wrist", [-1.0, 0.5])
    panel._run_mode_combo.setCurrentIndex(1)
    assert panel.selection().execution_ready is True
    try:
        with qtbot.waitSignal(tab.runCompleted, timeout=10_000):
            run = tab.run_now()
        assert run is not None
        assert run.config.source_kind == "double_pendulum"
        assert run.config.swing_run_config.prescribed_profile_id == (
            "profile.rate_of_closure.driver.v1"
        )
        assert "executed" in panel._status_label.text().lower()
    finally:
        tab.stop()
