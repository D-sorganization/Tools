"""Main-window ownership boundary for combined regional-ground requests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rate_of_closure.application.regional_surface_plan import (
    editor_draft_from_regional_surface_plan_request,
)
from rate_of_closure.ui.pyqt6.regional_ground_variation_request_io import (
    RegionalGroundVariationRequestFileActions,
)
from rate_of_closure.variation.regional_ground_variation import (
    GroundRegionalVariationRequest,
    register_ground_variation_variables,
)

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QStatusBar

    from rate_of_closure.ui.pyqt6.app_toolstrip import ApplicationToolstrip
    from rate_of_closure.ui.pyqt6.regional_surface_plan_tab import (
        RegionalSurfacePlanTab,
    )
    from rate_of_closure.ui.pyqt6.variation_tab import VariationTab


class RegionalGroundVariationWindowMixin:
    """Coordinate native dialogs and the two request-owning editor tabs."""

    _app_toolstrip: ApplicationToolstrip
    _regional_surface_plan_tab: RegionalSurfacePlanTab
    _variation_tab: VariationTab
    _regional_ground_variation_files: RegionalGroundVariationRequestFileActions
    _loaded_regional_ground_variation_request: GroundRegionalVariationRequest | None

    def _prepare_regional_ground_variation_editors(self) -> None:
        """Register Rate-owned ground variables before constructing editor rows."""
        register_ground_variation_variables()

    def _initialize_regional_ground_variation_files(self) -> None:
        """Register extension variables and create the native file controller."""
        self._loaded_regional_ground_variation_request = None
        actions = RegionalGroundVariationRequestFileActions
        self._regional_ground_variation_files = actions(self, self)

    def _update_toolstrip_context(self, *_args: object) -> None:
        """Keep contextual File commands aligned with the active module."""
        self._app_toolstrip.set_active_module(self.current_primary_module_id())  # type: ignore[attr-defined]

    def _clear_loaded_regional_ground_variation_request(self) -> None:
        """Discard exact imported bytes after either owning editor changes."""
        self._loaded_regional_ground_variation_request = None

    def current_regional_ground_variation_request(
        self,
    ) -> GroundRegionalVariationRequest:
        """Return one validated combined snapshot without illustrative fallback."""
        loaded = self._loaded_regional_ground_variation_request
        if loaded is not None:
            return loaded
        return GroundRegionalVariationRequest(
            self._variation_tab.build_plan(),
            self._regional_surface_plan_tab.validated_request(),
            "regional-ground-variation-result",
            "rate-pyqt6-interactive-workspace",
            500,
            None,
        )

    def apply_regional_ground_variation_request(
        self, request: GroundRegionalVariationRequest
    ) -> None:
        """Apply one completely validated request to both editor owners."""
        if type(request) is not GroundRegionalVariationRequest:
            raise TypeError("request must be an exact GroundRegionalVariationRequest")
        self._variation_tab.require_plan_loadable(request.plan)
        editor_draft_from_regional_surface_plan_request(request.regional_plan)
        self._variation_tab.load_plan(request.plan)
        self._regional_surface_plan_tab.apply_imported_request(request.regional_plan)
        self._loaded_regional_ground_variation_request = request

    def show_regional_ground_variation_file_status(
        self, message: str, *, error: bool
    ) -> None:
        """Expose native file outcomes in the persistent application status bar."""
        status_bar: QStatusBar | None = self.statusBar()  # type: ignore[attr-defined]
        if status_bar is None:  # pragma: no cover - QMainWindow always owns one
            return
        prefix = "Regional-ground request error: " if error else ""
        status_bar.showMessage(f"{prefix}{message}")

    def open_regional_ground_variation_request(self) -> None:
        """Open the contextual combined request through a native dialog."""
        self._regional_ground_variation_files.open()

    def save_regional_ground_variation_request_as(self) -> None:
        """Save the contextual combined request through a native dialog."""
        self._regional_ground_variation_files.save_as()


__all__ = ["RegionalGroundVariationWindowMixin"]
