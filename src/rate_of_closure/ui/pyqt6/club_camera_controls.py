"""Accessible canonical camera controls for :class:`Club3DView`."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QHBoxLayout,
    QPushButton,
    QWidget,
)

from rate_of_closure.application.camera_presets import (
    CAMERA_COMMAND_IDS,
    CameraCommandId,
    CameraState,
    CameraViewId,
    FaceOnSide,
    apply_camera_view,
    auto_fit_camera,
    camera_preset,
    matplotlib_angles,
    set_face_on_side,
    with_camera_zoom,
)

FitBounds = tuple[float, float]

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
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._changed = changed
        self._fit_bounds = fit_bounds
        self._state = CameraState()
        self._is_canonical_orientation = True
        self._command_widgets: dict[str, QWidget] = {}
        self._view_buttons: dict[CameraViewId, QPushButton] = {}

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 0)
        layout.setSpacing(5)
        self._view_group = QButtonGroup(self)
        self._view_group.setExclusive(True)
        for command_id, label, tooltip in _VIEW_BUTTONS:
            button = self._button(command_id.value, label, tooltip, checkable=True)
            button.clicked.connect(
                lambda _checked, value=command_id: self.apply_command(value)
            )
            self._view_group.addButton(button)
            self._view_buttons[command_id] = button
            layout.addWidget(button)

        self._face_side = QComboBox()
        self._face_side.setAccessibleName("Face-on camera side")
        self._face_side.setProperty("cameraControlId", "camera.face_on_side")
        self._face_side.setToolTip(
            "Choose the physical viewing side; golfer handedness is never inferred."
        )
        self._face_side.addItem("Right of target", FaceOnSide.RIGHT)
        self._face_side.addItem("Left of target", FaceOnSide.LEFT)
        self._face_side.currentIndexChanged.connect(self._on_side_changed)
        layout.addWidget(self._face_side)

        reset = self._button(
            CameraCommandId.RESET_VIEW.value,
            "Reset View",
            "Restore canonical isometric orientation without changing target or zoom.",
        )
        reset.clicked.connect(
            lambda _checked: self.apply_command(CameraCommandId.RESET_VIEW)
        )
        layout.addWidget(reset)
        auto_fit = self._button(
            CameraCommandId.AUTO_FIT.value,
            "Auto Fit",
            "Fit the complete current clubhead and shaft with 16% clearance.",
        )
        auto_fit.clicked.connect(
            lambda _checked: self.apply_command(CameraCommandId.AUTO_FIT)
        )
        layout.addWidget(auto_fit)
        layout.addStretch(1)
        self._sync()

    def _button(
        self, command_id: str, label: str, tooltip: str, *, checkable: bool = False
    ) -> QPushButton:
        button = QPushButton(label)
        button.setAccessibleName(label)
        button.setProperty("cameraCommandId", command_id)
        button.setToolTip(tooltip)
        button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        button.setCheckable(checkable)
        self._command_widgets[command_id] = button
        return button

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

    def mark_manual_orientation(self) -> None:
        """Clear preset selection after a free orbit without losing camera state."""
        self._is_canonical_orientation = False
        self._sync()

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
            else:
                subject_radius, base_half_extent = self._fit_bounds()
                self._state = auto_fit_camera(
                    self._state, subject_radius, base_half_extent
                )
                orientation_changed = False
        self._notify(orientation_changed=orientation_changed)

    def _on_side_changed(self, _index: int) -> None:
        self.set_face_on_side(FaceOnSide(self._face_side.currentData()))

    def _sync(self) -> None:
        self._face_side.blockSignals(True)
        self._face_side.setCurrentIndex(
            0 if self._state.face_on_side is FaceOnSide.RIGHT else 1
        )
        self._face_side.blockSignals(False)
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
