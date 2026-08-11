"""Shared accessible controls and state adapter for Matplotlib 3D cameras."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Protocol, runtime_checkable

from mpl_toolkits.mplot3d.axes3d import Axes3D
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.application.camera_commands import (
    CameraCommandId,
    CameraState,
    FaceOnSide,
    apply_camera_preset,
    apply_manual_override,
    recenter_camera,
    safe_tracking_zoom,
    set_tracking_enabled,
    update_tracking_target,
)

Vector3 = tuple[float, float, float]

_BUTTON_SPECS = (
    (
        CameraCommandId.VIEW_FACE_ON,
        "Face On",
        "App frame: lateral view toward the target line from the explicit side.",
    ),
    (
        CameraCommandId.VIEW_DOWN_THE_LINE,
        "Down the Line",
        "App frame: look from behind along +x downrange with +y vertical.",
    ),
    (
        CameraCommandId.VIEW_OVERHEAD,
        "Overhead",
        "App frame: look down along -y with +x downrange toward screen-up.",
    ),
    (
        CameraCommandId.VIEW_ISOMETRIC,
        "Reset View",
        "Restore the canonical isometric orientation without changing target or zoom.",
    ),
)

_ORTHOGRAPHIC_DEPTH_AXIS = {
    CameraCommandId.VIEW_FACE_ON: "x",
    CameraCommandId.VIEW_DOWN_THE_LINE: "y",
    CameraCommandId.VIEW_OVERHEAD: "z",
}


@runtime_checkable
class CameraViewport(Protocol):
    """Minimal viewport seam consumed by :class:`CameraViewportMixin`."""

    def _camera_subject_m(self) -> Vector3: ...

    def _camera_base_half_extent_m(self) -> float: ...

    def _camera_subject_radius_m(self) -> float: ...

    def _camera_state_changed(self) -> None: ...


class CameraControls(QWidget):
    """One DRY control bar used by swing/impact and flight viewports."""

    def __init__(
        self,
        subject_label: str,
        command: Callable[[CameraCommandId], None],
        face_side: Callable[[FaceOnSide], None],
        tracking: Callable[[bool], None],
        auto_fit: Callable[[bool], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._subject_label = subject_label
        self._command_widgets: dict[str, QWidget] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)
        view_layout = QHBoxLayout()
        view_layout.setSpacing(5)
        tracking_layout = QHBoxLayout()
        tracking_layout.setSpacing(5)
        status_layout = QHBoxLayout()
        layout.addLayout(view_layout)
        layout.addLayout(tracking_layout)
        layout.addLayout(status_layout)
        for command_id, label, tooltip in _BUTTON_SPECS:
            button = self._button(command_id, label, tooltip)
            button.clicked.connect(lambda _checked, value=command_id: command(value))
            view_layout.addWidget(button)
        self.face_side = QComboBox()
        self.face_side.setAccessibleName("Face-on Camera Side")
        self.face_side.addItem("Right of target", FaceOnSide.RIGHT)
        self.face_side.addItem("Left of target", FaceOnSide.LEFT)
        self.face_side.setToolTip(
            "Choose the physical side of the target line; handedness is never inferred."
        )
        self.face_side.currentIndexChanged.connect(
            lambda _index: face_side(FaceOnSide(self.face_side.currentData()))
        )
        view_layout.addStretch(1)
        tracking_layout.addWidget(self.face_side)
        self.track = QCheckBox(f"Track {subject_label}")
        self._configure_command_widget(
            self.track,
            CameraCommandId.TRACK_SUBJECT,
            f"Follow the {subject_label.lower()} with bounded focus updates.",
        )
        self.track.toggled.connect(tracking)
        tracking_layout.addWidget(self.track)
        self.auto_fit = QCheckBox("Auto Fit")
        self._configure_command_widget(
            self.auto_fit,
            CameraCommandId.AUTO_FIT,
            "Opt in to reducing unsafe zoom only when 16% subject "
            "clearance is violated.",
        )
        self.auto_fit.toggled.connect(auto_fit)
        tracking_layout.addWidget(self.auto_fit)
        self.recenter = self._button(
            CameraCommandId.RECENTER,
            "Re-center",
            f"Center on the current {subject_label.lower()} and resume tracking.",
        )
        self.recenter.clicked.connect(
            lambda _checked: command(CameraCommandId.RECENTER)
        )
        tracking_layout.addWidget(self.recenter)
        self.status = QLabel("Tracking off")
        self.status.setAccessibleName("Camera Tracking State")
        tracking_layout.addStretch(1)
        status_layout.addWidget(self.status)
        status_layout.addStretch(1)

    def _button(
        self, command_id: CameraCommandId, label: str, tooltip: str
    ) -> QPushButton:
        button = QPushButton(label)
        self._configure_command_widget(button, command_id, tooltip)
        return button

    def _configure_command_widget(
        self, widget: QWidget, command_id: CameraCommandId, tooltip: str
    ) -> None:
        widget.setProperty("cameraCommandId", command_id.value)
        widget.setToolTip(tooltip)
        widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._command_widgets[command_id.value] = widget

    def command_widgets(self) -> dict[str, QWidget]:
        """Return the stable command-to-widget map for parity inspection."""
        return dict(self._command_widgets)

    def sync(self, state: CameraState) -> None:
        """Refresh controls without feeding changes back into the viewport."""
        self.face_side.blockSignals(True)
        self.face_side.setCurrentIndex(
            0 if state.face_on_side is FaceOnSide.RIGHT else 1
        )
        self.face_side.blockSignals(False)
        self.track.blockSignals(True)
        self.track.setChecked(state.tracking_enabled)
        self.track.blockSignals(False)
        self.auto_fit.blockSignals(True)
        self.auto_fit.setChecked(state.auto_fit_enabled)
        self.auto_fit.blockSignals(False)
        if state.tracking_suspended:
            text = "Tracking suspended by manual orbit"
        elif state.tracking_enabled:
            text = f"Tracking {self._subject_label}"
        else:
            text = "Tracking off"
        self.status.setText(text)


class CameraViewportMixin:
    """Own one isolated camera state and delegate rendering to a viewport."""

    _camera_state: CameraState
    _camera_controls_widget: CameraControls

    def _camera_subject_m(self) -> Vector3:
        raise NotImplementedError

    def _camera_base_half_extent_m(self) -> float:
        raise NotImplementedError

    def _camera_subject_radius_m(self) -> float:
        raise NotImplementedError

    def _camera_state_changed(self) -> None:
        raise NotImplementedError

    def _initialize_camera(self, subject_label: str) -> CameraControls:
        self._camera_state = CameraState()
        self._camera_controls_widget = CameraControls(
            subject_label,
            self.apply_camera_command,
            self.set_face_on_side,
            self.set_camera_tracking,
            self.set_camera_auto_fit,
        )
        return self._camera_controls_widget

    def camera_controls(self) -> CameraControls:
        """Return this viewport's isolated, accessible camera controls."""
        return self._camera_controls_widget

    def camera_state(self) -> CameraState:
        """Return the immutable camera state snapshot."""
        return self._camera_state

    def camera_zoom(self) -> float:
        """Return the current dimensionless zoom factor."""
        return float(self._camera_state.zoom)

    def set_camera_zoom(self, zoom: float) -> None:
        """Set bounded zoom, applying opt-in subject clearance when enabled."""
        requested = max(0.25, min(8.0, float(zoom)))
        if self._camera_state.auto_fit_enabled:
            requested = safe_tracking_zoom(
                requested,
                self._camera_subject_radius_m(),
                self._camera_base_half_extent_m(),
            )
        self._camera_state = replace(self._camera_state, zoom=requested)
        self._notify_camera_state_changed()

    def set_camera_tracking(self, enabled: bool) -> None:
        """Toggle tracking and center immediately when enabling it."""
        self._camera_state = set_tracking_enabled(
            self._camera_state, bool(enabled), self._camera_subject_m()
        )
        self._notify_camera_state_changed()

    def set_camera_auto_fit(self, enabled: bool) -> None:
        """Opt in/out of clearance-constrained zoom and fit immediately."""
        self._camera_state = replace(self._camera_state, auto_fit_enabled=bool(enabled))
        if enabled:
            self._camera_state = replace(
                self._camera_state,
                zoom=safe_tracking_zoom(
                    self._camera_state.zoom,
                    self._camera_subject_radius_m(),
                    self._camera_base_half_extent_m(),
                ),
            )
        self._notify_camera_state_changed()

    def set_face_on_side(self, side: FaceOnSide) -> None:
        """Set the deliberate lateral side and refresh an active face-on view."""
        self._camera_state = replace(self._camera_state, face_on_side=side)
        if self._camera_state.preset_id is CameraCommandId.VIEW_FACE_ON:
            self._camera_state = apply_camera_preset(
                self._camera_state, CameraCommandId.VIEW_FACE_ON
            )
        self._notify_camera_state_changed()

    def apply_camera_command(self, command_id: CameraCommandId) -> None:
        """Apply a stable snap/reset/recenter command idempotently."""
        if command_id is CameraCommandId.RECENTER:
            self._camera_state = recenter_camera(
                self._camera_state, self._camera_subject_m()
            )
        elif command_id is CameraCommandId.AUTO_FIT:
            self.set_camera_auto_fit(True)
            return
        elif command_id is CameraCommandId.TRACK_SUBJECT:
            self.set_camera_tracking(not self._camera_state.tracking_enabled)
            return
        else:
            self._camera_state = apply_camera_preset(self._camera_state, command_id)
        self._notify_camera_state_changed()

    def suspend_camera_tracking(self) -> None:
        """Record an intentional manual orbit and pause focus following."""
        self._camera_state = apply_manual_override(self._camera_state)
        self._notify_camera_state_changed()

    def recenter_camera(self) -> None:
        """Public one-action recovery after a manual camera override."""
        self.apply_camera_command(CameraCommandId.RECENTER)

    def _advance_camera_tracking(self) -> None:
        maximum_step = max(0.05, self._camera_base_half_extent_m() * 0.5)
        self._camera_state = update_tracking_target(
            self._camera_state, self._camera_subject_m(), maximum_step
        )
        self._camera_controls_widget.sync(self._camera_state)

    def _apply_camera_axis_visibility(self, axes: Axes3D) -> None:
        """Hide only the depth axis for an exact orthographic preset.

        Matplotlib display axes are ``x=right``, ``y=downrange``, and
        ``z=up``. Isometric and manually orbited views restore every physical
        axis so a snap never leaves persistent presentation state behind.
        """
        preset_id = self._camera_state.preset_id
        hidden_axis = (
            None if preset_id is None else _ORTHOGRAPHIC_DEPTH_AXIS.get(preset_id)
        )
        for axis_name, axis in (
            ("x", axes.xaxis),
            ("y", axes.yaxis),
            ("z", axes.zaxis),
        ):
            axis.set_visible(axis_name != hidden_axis)

    def _notify_camera_state_changed(self) -> None:
        self._camera_controls_widget.sync(self._camera_state)
        viewport = self
        if not isinstance(viewport, CameraViewport):
            raise TypeError("CameraViewportMixin requires the CameraViewport contract")
        viewport._camera_state_changed()


__all__ = ["CameraControls", "CameraViewportMixin"]
