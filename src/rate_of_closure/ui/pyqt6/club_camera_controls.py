"""Accessible canonical camera controls for :class:`Club3DView`."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.application.camera_presets import (
    CAMERA_COMMAND_IDS,
    CameraCommandId,
    CameraState,
    CameraTrackingStateId,
    CameraViewId,
    FaceOnSide,
    apply_camera_view,
    apply_manual_camera_override,
    auto_fit_camera,
    camera_preset,
    enforce_tracking_clearance,
    matplotlib_angles,
    recenter_camera,
    set_auto_fit_fallback,
    set_camera_tracking,
    set_face_on_side,
    tracking_state_id,
    update_tracking_target,
    with_camera_zoom,
)

FitBounds = tuple[float, float]
Vector3 = tuple[float, float, float]

_VIEW_BUTTONS = (
    (
        CameraViewId.ISOMETRIC,
        "Isometric",
        "Canonical engineering isometric view; preserves target and zoom.",
    ),
    (
        CameraViewId.FACE_ON,
        "Face On",
        "Lateral view from the explicitly selected side of the target line.",
    ),
    (
        CameraViewId.DOWN_THE_LINE,
        "Down the Line",
        "Look from behind exactly along +x downrange with +y vertical.",
    ),
    (
        CameraViewId.OVERHEAD,
        "Overhead",
        "Look exactly down along -y with +x downrange toward screen-up.",
    ),
)


class ClubCameraControls(QWidget):
    """Own one immutable camera state and its compact command bar."""

    def __init__(
        self,
        changed: Callable[[CameraState, bool], None],
        fit_bounds: Callable[[], FitBounds],
        subject_m: Callable[[], Vector3],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._changed = changed
        self._fit_bounds = fit_bounds
        self._subject_m = subject_m
        self._state = CameraState()
        self._is_canonical_orientation = True
        self._command_widgets: dict[str, QWidget] = {}
        self._view_buttons: dict[CameraViewId, QPushButton] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 0)
        layout.setSpacing(2)
        view_row = QHBoxLayout()
        tracking_row = QHBoxLayout()
        layout.addLayout(view_row)
        layout.addLayout(tracking_row)
        self._view_group = QButtonGroup(self)
        self._view_group.setExclusive(True)
        for command_id, label, tooltip in _VIEW_BUTTONS:
            button = self._button(command_id.value, label, tooltip, checkable=True)
            button.clicked.connect(
                lambda _checked, value=command_id: self.apply_command(value)
            )
            self._view_group.addButton(button)
            self._view_buttons[command_id] = button
            view_row.addWidget(button)

        self._face_side = QComboBox()
        self._face_side.setAccessibleName("Face-on camera side")
        self._face_side.setProperty("cameraControlId", "camera.face_on_side")
        self._face_side.setToolTip(
            "Choose the physical viewing side; golfer handedness is never inferred."
        )
        self._face_side.addItem("Right of target", FaceOnSide.RIGHT)
        self._face_side.addItem("Left of target", FaceOnSide.LEFT)
        self._face_side.currentIndexChanged.connect(self._on_side_changed)
        view_row.addWidget(self._face_side)

        reset = self._button(
            CameraCommandId.RESET_VIEW.value,
            "Reset View",
            "Restore canonical isometric orientation without changing target or zoom.",
        )
        reset.clicked.connect(
            lambda _checked: self.apply_command(CameraCommandId.RESET_VIEW)
        )
        view_row.addWidget(reset)
        auto_fit = self._button(
            CameraCommandId.AUTO_FIT.value,
            "Auto Fit",
            "Fit the complete current clubhead and shaft with 16% clearance.",
        )
        auto_fit.clicked.connect(
            lambda _checked: self.apply_command(CameraCommandId.AUTO_FIT)
        )
        view_row.addWidget(auto_fit)
        view_row.addStretch(1)

        self._track = QCheckBox("Track Clubhead")
        self._configure_widget(
            self._track,
            CameraCommandId.TRACK_CLUBHEAD.value,
            "Follow the moving clubhead with bounded target updates; "
            "zoom is preserved.",
        )
        self._track.toggled.connect(self.set_tracking_enabled)
        tracking_row.addWidget(self._track)

        self._auto_fit_fallback = QCheckBox("Auto Fit fallback")
        self._auto_fit_fallback.setAccessibleName("Auto Fit fallback")
        self._auto_fit_fallback.setProperty(
            "cameraControlId", "camera.auto_fit_fallback"
        )
        self._auto_fit_fallback.setToolTip(
            "Opt in to reducing unsafe zoom only when 16% clubhead clearance "
            "would otherwise be violated."
        )
        self._auto_fit_fallback.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._auto_fit_fallback.toggled.connect(self.set_auto_fit_fallback)
        tracking_row.addWidget(self._auto_fit_fallback)

        recenter = self._button(
            CameraCommandId.RECENTER.value,
            "Re-center Clubhead",
            "Center on the current clubhead and resume tracking without changing zoom.",
        )
        recenter.clicked.connect(lambda _checked: self.recenter())
        tracking_row.addWidget(recenter)
        self._tracking_status = QLabel("Tracking off")
        self._tracking_status.setAccessibleName("Camera tracking state")
        tracking_row.addWidget(self._tracking_status)
        tracking_row.addStretch(1)
        self._sync()

    def _button(
        self, command_id: str, label: str, tooltip: str, *, checkable: bool = False
    ) -> QPushButton:
        button = QPushButton(label)
        self._configure_widget(button, command_id, tooltip)
        button.setCheckable(checkable)
        return button

    def _configure_widget(self, widget: QWidget, command_id: str, tooltip: str) -> None:
        """Register one accessible stable command widget."""
        accessible_name = (
            widget.text() if isinstance(widget, QAbstractButton) else command_id
        )
        widget.setAccessibleName(widget.accessibleName() or accessible_name)
        widget.setProperty("cameraCommandId", command_id)
        widget.setToolTip(tooltip)
        widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._command_widgets[command_id] = widget

    def command_widgets(self) -> dict[str, QWidget]:
        """Return the complete stable command/widget mapping."""
        assert set(self._command_widgets) == set(CAMERA_COMMAND_IDS)
        return dict(self._command_widgets)

    def state(self) -> CameraState:
        """Return the immutable current state."""
        return self._state

    def view_buttons(self) -> dict[CameraViewId, QPushButton]:
        """Return preset buttons for semantic-state inspection and testing."""
        return dict(self._view_buttons)

    def active_view_id(self) -> CameraViewId | None:
        """Return the exact active preset, or ``None`` after free orbit."""
        return self._state.preset_id if self._is_canonical_orientation else None

    def tracking_status_label(self) -> QLabel:
        """Return the visible, accessible tracking state label."""
        return self._tracking_status

    def mark_manual_orientation(self, target_m: Vector3 | None = None) -> None:
        """Clear preset selection and suspend tracking after manual movement."""
        self._is_canonical_orientation = False
        self._state = apply_manual_camera_override(self._state, target_m)
        self._notify(orientation_changed=False)

    def angles(self) -> tuple[float, float]:
        """Return the exact current Matplotlib elevation and azimuth."""
        angles: tuple[float, float] = matplotlib_angles(
            camera_preset(self._state.preset_id, self._state.face_on_side)
        )
        return angles

    def set_zoom(self, zoom: float) -> None:
        """Set bounded zoom without changing orientation or target."""
        self._state = with_camera_zoom(self._state, zoom)
        self._notify(orientation_changed=False)

    def set_tracking_enabled(self, enabled: bool) -> None:
        """Toggle tracking and center immediately when enabling it."""
        self._state = set_camera_tracking(self._state, enabled, self._subject_m())
        self._state = self._tracking_clearance(self._state)
        self._notify(orientation_changed=False)

    def set_auto_fit_fallback(self, enabled: bool) -> None:
        """Toggle reduction-only clearance protection and apply it immediately."""
        self._state = set_auto_fit_fallback(self._state, enabled)
        self._state = self._tracking_clearance(self._state)
        self._notify(orientation_changed=False)

    def recenter(self) -> None:
        """Center on the current clubhead and resume an enabled tracker."""
        self._state = recenter_camera(self._state, self._subject_m())
        self._notify(orientation_changed=False)

    def advance_tracking(
        self, subject_m: Vector3, *, recenter_on_wrap: bool = False
    ) -> None:
        """Advance per-frame state without recursively requesting a redraw."""
        self._state = (
            recenter_camera(self._state, subject_m)
            if recenter_on_wrap
            and self._state.tracking_enabled
            and not self._state.tracking_suspended
            else update_tracking_target(self._state, subject_m)
        )
        self._sync()

    def enforce_clearance(self) -> None:
        """Apply the opt-in clearance fallback without requesting a redraw."""
        self._state = self._tracking_clearance(self._state)
        self._sync()

    def tracking_state_id(self) -> CameraTrackingStateId:
        """Return the stable visible tracking-state identifier."""
        return tracking_state_id(self._state)

    def _tracking_clearance(self, state: CameraState) -> CameraState:
        subject_radius, base_half_extent = self._fit_bounds()
        return enforce_tracking_clearance(state, subject_radius, base_half_extent)

    def set_face_on_side(self, side: FaceOnSide | str) -> None:
        """Set the explicit lateral side and update an active Face-On view."""
        self._state = set_face_on_side(self._state, side)
        face_on_active = self._state.preset_id is CameraViewId.FACE_ON
        if face_on_active:
            self._is_canonical_orientation = True
        self._notify(orientation_changed=face_on_active)

    def apply_command(self, command_id: CameraViewId | CameraCommandId | str) -> None:
        """Apply one strict view/reset/fit command."""
        if isinstance(command_id, CameraViewId) or command_id in {
            view.value for view in CameraViewId
        }:
            self._state = apply_camera_view(self._state, str(command_id))
            self._is_canonical_orientation = True
            orientation_changed = True
        else:
            try:
                action = (
                    command_id
                    if isinstance(command_id, CameraCommandId)
                    else CameraCommandId(command_id)
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"unknown camera command: {command_id!r}") from exc
            if action is CameraCommandId.RESET_VIEW:
                self._state = apply_camera_view(self._state, CameraViewId.ISOMETRIC)
                self._is_canonical_orientation = True
                orientation_changed = True
            elif action is CameraCommandId.AUTO_FIT:
                subject_radius, base_half_extent = self._fit_bounds()
                self._state = auto_fit_camera(
                    self._state, subject_radius, base_half_extent
                )
                orientation_changed = False
            elif action is CameraCommandId.TRACK_CLUBHEAD:
                self.set_tracking_enabled(not self._state.tracking_enabled)
                return
            else:
                self.recenter()
                return
        self._notify(orientation_changed=orientation_changed)

    def _on_side_changed(self, _index: int) -> None:
        self.set_face_on_side(FaceOnSide(self._face_side.currentData()))

    def _sync(self) -> None:
        self._face_side.blockSignals(True)
        self._face_side.setCurrentIndex(
            0 if self._state.face_on_side is FaceOnSide.RIGHT else 1
        )
        self._face_side.blockSignals(False)
        self._track.blockSignals(True)
        self._track.setChecked(self._state.tracking_enabled)
        self._track.blockSignals(False)
        self._auto_fit_fallback.blockSignals(True)
        self._auto_fit_fallback.setChecked(self._state.auto_fit_fallback_enabled)
        self._auto_fit_fallback.blockSignals(False)
        state_id = tracking_state_id(self._state)
        status = {
            CameraTrackingStateId.OFF: "Tracking off",
            CameraTrackingStateId.ACTIVE: "Tracking Clubhead",
            CameraTrackingStateId.SUSPENDED: "Tracking suspended by manual camera",
        }[state_id]
        self._tracking_status.setText(status)
        self._tracking_status.setProperty("cameraTrackingStateId", state_id.value)
        self._view_group.setExclusive(False)
        for view_id, button in self._view_buttons.items():
            button.setChecked(
                self._is_canonical_orientation and view_id is self._state.preset_id
            )
        self._view_group.setExclusive(True)

    def _notify(self, *, orientation_changed: bool) -> None:
        self._sync()
        self._changed(self._state, orientation_changed)


__all__ = ["ClubCameraControls"]
