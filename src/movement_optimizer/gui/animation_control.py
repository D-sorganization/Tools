# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Animation playback helpers extracted from MainWindow.

Provides play/pause toggle, step forward/back, rewind, and frame
advance as a mixin class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from ..models import BodyModel
from ..trajectory import OptimizationResult

if TYPE_CHECKING:
    from PyQt6.QtCore import QTimer

    from .controls_bar import PlaybackControls
    from .exercise_tab import ExerciseTab


def _require_body_for_animation(body: BodyModel | None) -> BodyModel:
    if body is None:
        raise ValueError("DbC Blocked: animation frame requires a body model.")
    return body


def _active_exercise_index(window: AnimationControlMixin) -> int | None:
    """Return the active barbell exercise index, or ``None`` for analysis tabs."""
    idx = window.tabs.currentIndex()
    if 0 <= idx < len(window.EXERCISE_CONFIGS):
        return idx
    return None


class AnimationControlMixin:
    """Mixin providing animation playback for MainWindow."""

    EXERCISE_CONFIGS: ClassVar[tuple[tuple[str, str], ...]]
    anim_timer: QTimer
    controls: PlaybackControls
    exercise_tabs: list[ExerciseTab]
    is_playing: bool
    tabs: Any

    if TYPE_CHECKING:

        def _snapshot_idx_state(
            self, idx: int
        ) -> tuple[OptimizationResult | None, int, BodyModel | None, Any]:
            """Return the currently published optimization state for an exercise."""
            raise NotImplementedError

        def _set_anim_frame(self, idx: int, frame: int) -> None:
            """Persist the current animation frame for an exercise."""
            raise NotImplementedError

    def _toggle_play(self) -> None:
        idx = _active_exercise_index(self)
        if idx is None:
            self._stop_anim()
            return
        r, _fi, _body, _dyn = self._snapshot_idx_state(idx)
        if r is None:
            return
        if self.is_playing:
            self._stop_anim()
        else:
            self.is_playing = True
            self.controls.set_playing(True)
            self._anim_step()

    def _stop_anim(self) -> None:
        self.is_playing = False
        self.anim_timer.stop()
        self.controls.set_playing(False)

    def _anim_step(self) -> None:
        if not self.is_playing:
            return
        idx = _active_exercise_index(self)
        if idx is None:
            self._stop_anim()
            return
        r, fi, body, dyn = self._snapshot_idx_state(idx)
        if r is None:
            self._stop_anim()
            return

        _, etype = self.EXERCISE_CONFIGS[idx]
        body = _require_body_for_animation(body)
        self.exercise_tabs[idx].draw_anim_frame(
            fi,
            r,
            dyn,
            body,
            etype,
        )

        n = len(r.t)
        next_frame = (fi + 1) % n
        self._set_anim_frame(idx, next_frame)
        speed = self.controls.speed_multiplier()
        self.controls.set_playback_status(fi + 1, n, speed)
        delay = max(15, int(40 / max(0.1, speed)))
        if next_frame == 0:
            delay = 700
        self.anim_timer.start(delay)

    def _step_fwd(self) -> None:
        idx = _active_exercise_index(self)
        if idx is None:
            self._stop_anim()
            return
        r, fi, body, dyn = self._snapshot_idx_state(idx)
        if r is None:
            return
        self._stop_anim()
        n = len(r.t)
        new_frame = (fi + 1) % n
        self._set_anim_frame(idx, new_frame)
        _, etype = self.EXERCISE_CONFIGS[idx]
        body = _require_body_for_animation(body)
        self.exercise_tabs[idx].draw_anim_frame(
            new_frame,
            r,
            dyn,
            body,
            etype,
        )
        self.controls.set_playback_status(
            new_frame + 1,
            n,
            self.controls.speed_multiplier(),
        )

    def _step_back(self) -> None:
        idx = _active_exercise_index(self)
        if idx is None:
            self._stop_anim()
            return
        r, fi, body, dyn = self._snapshot_idx_state(idx)
        if r is None:
            return
        self._stop_anim()
        n = len(r.t)
        new_frame = (fi - 1) % n
        self._set_anim_frame(idx, new_frame)
        _, etype = self.EXERCISE_CONFIGS[idx]
        body = _require_body_for_animation(body)
        self.exercise_tabs[idx].draw_anim_frame(
            new_frame,
            r,
            dyn,
            body,
            etype,
        )

    def _rewind(self) -> None:
        idx = _active_exercise_index(self)
        if idx is None:
            self._stop_anim()
            return
        r, _fi, body, dyn = self._snapshot_idx_state(idx)
        if r is None:
            return
        self._stop_anim()
        self._set_anim_frame(idx, 0)
        _, etype = self.EXERCISE_CONFIGS[idx]
        body = _require_body_for_animation(body)
        self.exercise_tabs[idx].draw_anim_frame(
            0,
            r,
            dyn,
            body,
            etype,
        )

    def _jump_to_end(self) -> None:
        idx = _active_exercise_index(self)
        if idx is None:
            self._stop_anim()
            return
        r, _fi, body, dyn = self._snapshot_idx_state(idx)
        if r is None:
            return
        self._stop_anim()
        n = len(r.t)
        last_frame = n - 1
        self._set_anim_frame(idx, last_frame)
        _, etype = self.EXERCISE_CONFIGS[idx]
        body = _require_body_for_animation(body)
        self.exercise_tabs[idx].draw_anim_frame(
            last_frame,
            r,
            dyn,
            body,
            etype,
        )
        self.controls.set_playback_status(
            last_frame + 1,
            n,
            self.controls.speed_multiplier(),
        )

    def _on_speed(self, speed: float) -> None:
        self.controls.set_speed_multiplier_text(speed)
