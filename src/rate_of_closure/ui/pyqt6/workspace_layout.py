"""Bounded persisted visual layout for the standalone PyQt shell."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rate_of_closure.club_camera import DEFAULT_CLUB_CAMERA
from rate_of_closure.visual_layout_preferences import (
    DEFAULT_SIDEBAR_FRACTION,
    MAX_SIDEBAR_FRACTION,
    MIN_SIDEBAR_FRACTION,
    VisualLayoutPreferences,
    load_visual_layout,
    save_visual_layout,
)

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QSplitter, QTabWidget

    from rate_of_closure.ui.pyqt6.club_view import Club3DView
    from rate_of_closure.ui.pyqt6.navigation_state import NavigationSettings

logger = logging.getLogger(__name__)


class WorkspaceLayoutMixin:
    """Restore and persist only presentation state with safe visual bounds."""

    if TYPE_CHECKING:
        _navigation_settings: NavigationSettings
        _shell_splitter: QSplitter
        _tabs: QTabWidget
        _club_view: Club3DView
        _visual_layout: VisualLayoutPreferences

    def _restore_visual_layout(self) -> None:
        preferences = load_visual_layout(self._navigation_settings)
        try:
            self._club_view.set_camera(preferences.club_camera)
        except Exception as error:  # noqa: BLE001 - optional preference restore
            logger.warning("Ignoring club camera layout restore failure: %s", error)
            preferences = VisualLayoutPreferences(
                DEFAULT_CLUB_CAMERA,
                preferences.module_help_open,
                preferences.shell_sidebar_fraction,
            )
        self._visual_layout = preferences
        self._apply_sidebar_fraction(preferences.shell_sidebar_fraction)

    def _connect_visual_layout_persistence(self) -> None:
        self._shell_splitter.splitterMoved.connect(self._persist_visual_layout)
        self._club_view.cameraChanged.connect(self._persist_visual_layout)

    def _apply_sidebar_fraction(self, fraction: float) -> None:
        bounded = min(MAX_SIDEBAR_FRACTION, max(MIN_SIDEBAR_FRACTION, fraction))
        available = self._shell_splitter.width() - self._shell_splitter.handleWidth()
        total = max(1_000, available)
        self._shell_splitter.setSizes(
            [round(bounded * total), round((1 - bounded) * total)]
        )

    def _reapply_visual_layout_geometry(self) -> None:
        """Apply the stored ratio after Qt assigns the shown shell width."""
        self._apply_sidebar_fraction(self._visual_layout.shell_sidebar_fraction)

    def _current_sidebar_fraction(self) -> float:
        sizes = self._shell_splitter.sizes()
        total = sum(sizes)
        if len(sizes) != 2 or total <= 0:
            return float(DEFAULT_SIDEBAR_FRACTION)
        return float(
            min(
                MAX_SIDEBAR_FRACTION,
                max(MIN_SIDEBAR_FRACTION, sizes[0] / total),
            )
        )

    def _persist_visual_layout(self, *_values: object) -> None:
        preferences = VisualLayoutPreferences(
            self._club_view.camera(),
            False,
            self._current_sidebar_fraction(),
        )
        if save_visual_layout(self._navigation_settings, preferences):
            self._visual_layout = preferences
