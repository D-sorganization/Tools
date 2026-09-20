"""View-command methods kept separate from the Rate main-window shell."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QDialog

from rate_of_closure.view_workspace import ViewKind, ViewLayout

if TYPE_CHECKING:
    from rate_of_closure.ui.pyqt6.app_toolstrip import (
        ApplicationToolstrip,
        ModuleManagerDialog,
    )
    from rate_of_closure.ui.pyqt6.simulation_tab import SimulationTab


class MainWindowViewCommandsMixin:
    """Route UI-neutral view commands into the real simulation compositor."""

    _module_manager_dialog: ModuleManagerDialog | None
    _simulation_tab: SimulationTab
    _app_toolstrip: ApplicationToolstrip

    if TYPE_CHECKING:

        def show_primary_module(self, module_id: str) -> None: ...

    def show_compositor_view(self, view_id: str | ViewKind) -> None:
        """Show a named real viewport through the shared simulation compositor."""
        if isinstance(view_id, ViewKind):
            kind = view_id
        else:
            raw = view_id.removeprefix("view.show_")
            try:
                kind = ViewKind(raw)
            except ValueError as exc:
                raise ValueError(f"unsupported compositor view: {view_id!r}") from exc
        self.show_primary_module("simulation")
        self._simulation_tab.show_compositor_view(kind)

    def show_compositor_layout(self, layout: ViewLayout | str) -> None:
        """Apply a multi-view layout preset in the simulation compositor."""
        if isinstance(layout, str):
            try:
                layout = ViewLayout(layout)
            except ValueError as exc:
                raise ValueError(f"unsupported compositor layout: {layout!r}") from exc
        self.show_primary_module("simulation")
        self._simulation_tab.show_compositor_layout(layout)

    def module_manager_dialog(self) -> ModuleManagerDialog | None:
        """Return the current workspace module manager, if one was opened."""
        return self._module_manager_dialog

    def shortcut_help_dialog(self) -> QDialog | None:
        """Return the current keyboard-shortcut help dialog."""
        dialog = self._app_toolstrip.shortcut_dialog()
        return dialog if isinstance(dialog, QDialog) else None


__all__ = ["MainWindowViewCommandsMixin"]
