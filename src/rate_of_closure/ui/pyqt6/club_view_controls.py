"""Construction helper for Club3DView controls."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
)

from rate_of_closure.club_camera import ClubCameraAction
from rate_of_closure.ui.pyqt6.club_view_render import VIEW_MODES
from rate_of_closure.units import FIELD_GUIDANCE

if TYPE_CHECKING:
    from rate_of_closure.ui.pyqt6.club_view import Club3DView


def build_playback_bar(view: Club3DView) -> QHBoxLayout:
    """Build and bind the playback and camera control row."""
    bar = QHBoxLayout()
    bar.setContentsMargins(4, 4, 4, 0)
    view._play_button = QPushButton("Play")
    view._play_button.setToolTip("Play or pause the bounded clubhead animation.")
    view._play_button.setCheckable(True)
    view._play_button.setFixedWidth(72)
    view._play_button.toggled.connect(view._on_play_toggled)
    bar.addWidget(view._play_button)
    bar.addWidget(QLabel("Playback Speed"))
    view._speed_slider = QSlider(Qt.Orientation.Horizontal)
    view._speed_slider.setRange(10, 300)
    view._speed_slider.setValue(100)
    view._speed_slider.setAccessibleName("Playback speed multiplier")
    view._speed_slider.setToolTip("Animation speed: 0.1x to 3.0x")
    view._speed_slider.valueChanged.connect(view._on_speed_changed)
    bar.addWidget(view._speed_slider, stretch=1)
    view._speed_label = QLabel("1.0x")
    view._speed_label.setFixedWidth(40)
    bar.addWidget(view._speed_label)
    bar.addWidget(QLabel("Display"))
    view._mode_combo = QComboBox()
    view._mode_combo.addItems(VIEW_MODES)
    view._mode_combo.setCurrentIndex(1)
    view._mode_combo.setAccessibleName("Clubhead display mode")
    view._mode_combo.setToolTip(
        "Fixed rotates in place; Moving also translates along the target line."
    )
    view._mode_combo.currentTextChanged.connect(lambda _text: view._try_redraw())
    bar.addWidget(view._mode_combo)
    view._load_mesh_button = QPushButton("Load Clubhead STL…")
    view._load_mesh_button.setToolTip(
        "Read one bounded local STL without upload and display-normalize it; "
        "STL units and physical registration are unknown."
    )
    view._load_mesh_button.clicked.connect(view._on_load_mesh_clicked)
    bar.addWidget(view._load_mesh_button)
    view._reset_mesh_button = QPushButton("Procedural Head")
    view._reset_mesh_button.setToolTip("Return to the authored procedural wireframe.")
    view._reset_mesh_button.setEnabled(False)
    view._reset_mesh_button.clicked.connect(view.try_clear_mesh)
    bar.addWidget(view._reset_mesh_button)
    view._show_cg_check = QCheckBox("Show reference marker")
    view._show_cg_check.setChecked(True)
    view._show_cg_check.setToolTip(FIELD_GUIDANCE["show_cg_marker"])
    view._show_cg_check.toggled.connect(lambda _checked: view._try_redraw())
    bar.addWidget(view._show_cg_check)
    view._reset_view_button = QPushButton("Reset View")
    view._reset_view_button.setToolTip("Restore the canonical clubhead camera view.")
    view._reset_view_button.clicked.connect(
        lambda: view._try_camera_action(ClubCameraAction.HOME)
    )
    bar.addWidget(view._reset_view_button)
    return bar
