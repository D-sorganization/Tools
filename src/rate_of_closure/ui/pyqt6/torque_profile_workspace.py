"""Workspace capture/apply mixin for the native torque-profile editor."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from rate_of_closure.application.workspace_torque_session import (
    TorqueWorkspaceState,
)
from rate_of_closure.ui.pyqt6.torque_profile_controller import RunMode
from shared.python.swing_sim.run_config import (
    DoublePendulumRunConfig,
    SwingRunMode,
)

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QCheckBox, QComboBox, QLabel

    from rate_of_closure.ui.pyqt6.torque_profile_controller import (
        TorqueProfileLibraryAdapter,
    )


class TorqueProfileWorkspaceMixin:
    """Persist the canonical library and selection without UI-derived data."""

    _library: TorqueProfileLibraryAdapter
    _run_mode_combo: QComboBox
    _joint_lock_checks: dict[str, QCheckBox]
    _status_label: QLabel

    if TYPE_CHECKING:

        def joint_locks(self): ...  # type: ignore[no-untyped-def]

        def _refresh_profiles(self, selected_id: str | None = None) -> None: ...

        def _display_profile(self, profile): ...  # type: ignore[no-untyped-def]

        def _rebuild_assignment_rows(self) -> None: ...

    def torque_workspace_state(self) -> TorqueWorkspaceState:
        """Capture the live library, stable selection, mode, and joint locks."""
        profile = self._library.active_profile()
        active_id = None if profile is None else profile.profile_id
        mode = self._run_mode_combo.currentData()
        if mode is RunMode.PRESCRIBED_TORQUE and active_id is None:
            raise ValueError(
                "prescribed torque mode requires an active library profile"
            )
        config = (
            DoublePendulumRunConfig.prescribed(
                cast(str, active_id),
                joint_locks=self.joint_locks(),
            )
            if mode is RunMode.PRESCRIBED_TORQUE
            else DoublePendulumRunConfig(joint_locks=self.joint_locks())
        )
        return TorqueWorkspaceState(
            profiles=self._library.profiles(),
            active_profile_id=active_id,
            run_config=config,
        )

    def apply_torque_workspace_state(self, state: TorqueWorkspaceState) -> None:
        """Atomically apply one fully validated workspace torque selection."""
        if not isinstance(state, TorqueWorkspaceState):
            raise TypeError("state must be a TorqueWorkspaceState")
        self._library.replace_library(state.profiles, state.active_profile_id)
        self._refresh_profiles(state.active_profile_id)
        profile = self._library.active_profile()
        if profile is not None:
            self._display_profile(profile)
        else:
            self._rebuild_assignment_rows()
        desired_mode = (
            RunMode.PRESCRIBED_TORQUE
            if state.run_config.mode is SwingRunMode.PRESCRIBED
            else RunMode.OPTIMIZED_DEFAULT
        )
        self._run_mode_combo.blockSignals(True)
        self._run_mode_combo.setCurrentIndex(
            self._run_mode_combo.findData(desired_mode)
        )
        self._run_mode_combo.blockSignals(False)
        locked = set(state.run_config.joint_locks.locked_joint_ids)
        for joint_id, checkbox in self._joint_lock_checks.items():
            checkbox.blockSignals(True)
            checkbox.setChecked(joint_id in locked)
            checkbox.blockSignals(False)
        selected = "none" if profile is None else profile.name
        self._status_label.setText(
            f"Workspace restored torque library; active profile: {selected}."
        )


__all__ = ["TorqueProfileWorkspaceMixin"]
