"""Transactional source, camera, and playback lifecycle for Club3DView."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from rate_of_closure.club_camera import (
    ClubCamera,
    ClubCameraAction,
    apply_club_camera_action,
)
from rate_of_closure.club_mesh_source import ClubMeshSource

if TYPE_CHECKING:
    from rate_of_closure.ui.pyqt6.club_view import Club3DView

logger = logging.getLogger(__name__)
TIMER_INTERVAL_MS = 200
ANIMATION_CYCLE_MS = 1920


class ClubViewLifecycleMixin:
    """State adoption methods; mixed into the concrete QWidget view."""

    def _apply_source(self: Club3DView, candidate: ClubMeshSource) -> None:
        prior = (self._source, self._mesh, self._hosel, self._cog)
        self._source = candidate
        self._mesh = candidate.mesh
        self._hosel = None if candidate.hosel is None else np.asarray(candidate.hosel)
        self._cog = (
            None
            if candidate.geometric_centroid is None
            else np.asarray(candidate.geometric_centroid)
        )
        try:
            self._draw()
        except Exception:
            self._source, self._mesh, self._hosel, self._cog = prior
            try:
                self._draw()
            except Exception as rollback_error:
                logger.error("club view rollback redraw failed: %s", rollback_error)
            raise
        self._reset_mesh_button.setEnabled(candidate.kind != "procedural")
        self._error.hide()
        self._error_kind = None
        self._update_status()

    def _set_error(self: Club3DView, text: str, kind: str = "render") -> None:
        self._error_kind = kind
        suffix = (
            "; prior head and camera remain displayed."
            if kind == "import"
            else "; prior source and camera remain selected; the image may be stale."
        )
        self._error.setText(text + suffix)
        self._error.show()

    def _apply_camera_action(
        self: Club3DView,
        action: ClubCameraAction,
    ) -> None:
        self._adopt_camera(apply_club_camera_action(self._camera, action))

    def _try_camera_action(self: Club3DView, action: ClubCameraAction) -> None:
        try:
            self._apply_camera_action(action)
        except Exception as error:
            self._set_error(f"Clubhead render failed: {str(error)[:512]}", "render")

    def _try_redraw(self: Club3DView) -> None:
        try:
            self._draw()
        except Exception as error:
            self._set_error(f"Clubhead render failed: {str(error)[:512]}", "render")
        else:
            if self._error_kind == "render":
                self._error.hide()
                self._error_kind = None

    def _adopt_camera(self: Club3DView, candidate: ClubCamera) -> None:
        prior = self._camera
        canvas_had_focus = self._canvas.hasFocus()
        self._camera, self._zoom = candidate, candidate.zoom
        try:
            self._draw()
        except Exception as error:
            self._camera, self._zoom = prior, prior.zoom
            try:
                self._draw()
            except Exception as rollback_error:
                logger.error("club camera rollback redraw failed: %s", rollback_error)
            raise error
        if canvas_had_focus:
            self._canvas.setFocus()
        if self._error_kind == "render":
            self._error.hide()
            self._error_kind = None
        self._update_status()

    def _advance(self: Club3DView) -> None:
        prior_phase = self._phase
        self._phase = (
            prior_phase + self._speed * TIMER_INTERVAL_MS / ANIMATION_CYCLE_MS
        ) % 1.0
        try:
            self._draw()
        except Exception as error:
            self._phase = prior_phase
            try:
                self._draw()
            except Exception as rollback_error:
                logger.error("club animation rollback failed: %s", rollback_error)
            self._timer.stop()
            self._play_button.setChecked(False)
            self._set_error(f"Clubhead render failed: {str(error)[:512]}", "render")
        else:
            if self._error_kind == "render":
                self._error.hide()
                self._error_kind = None
