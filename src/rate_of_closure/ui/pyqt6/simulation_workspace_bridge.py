"""Whole-workspace bridge for simulation-local ball and target controls."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rate_of_closure.application.workspace_simulation_session import (
    SimulationWorkspaceState,
)

if TYPE_CHECKING:
    from rate_of_closure.ui.pyqt6.ball_setup_control import BallSetupControl
    from rate_of_closure.ui.pyqt6.spatial_target_panel import SpatialTargetPanel


class SimulationWorkspaceBridgeMixin:
    """Capture and apply the simulation fields owned by workspace files."""

    _ball_setup_control: BallSetupControl
    _spatial_target_panel: SpatialTargetPanel

    if TYPE_CHECKING:

        def _emit_config(self, *_args: object) -> None: ...

    def simulation_workspace_state(self) -> SimulationWorkspaceState:
        """Return the complete ball/target state committed by native editors."""
        return SimulationWorkspaceState(
            ball_setup=self._ball_setup_control.setup(),
            ball_setup_user_overridden=(
                not self._ball_setup_control.uses_club_default()
            ),
            spatial_target=self._spatial_target_panel.current_target(),
        )

    def apply_simulation_workspace_state(
        self,
        state: SimulationWorkspaceState,
    ) -> None:
        """Apply one already validated simulation slice without coercion."""
        if not isinstance(state, SimulationWorkspaceState):
            raise TypeError("state must be a SimulationWorkspaceState")
        self._ball_setup_control.set_persisted_setup(
            state.ball_setup,
            use_club_default=not state.ball_setup_user_overridden,
        )
        self._spatial_target_panel.set_target(state.spatial_target)
        self._emit_config()


__all__ = ["SimulationWorkspaceBridgeMixin"]
