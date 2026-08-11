"""Compact, accessible controls for the PyQt swing scene."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.ui.pyqt6.impact_layer_controls import ImpactLayerControls
from rate_of_closure.ui.pyqt6.simulation_specs import RATE_PRESETS
from rate_of_closure.units import FIELD_GUIDANCE


class SimulationViewControlsMixin:
    """Build responsive controls without owning scene or playback behavior."""

    _impact_layer_controls: ImpactLayerControls
    _ball_check: QCheckBox
    _ground_check: QCheckBox
    _course_check: QCheckBox
    _screw_check: QCheckBox
    _impact_check: QCheckBox
    _kinetics_check: QCheckBox
    _flight_check: QCheckBox
    _impact_layer_checks: dict[str, QCheckBox]

    if TYPE_CHECKING:

        def _apply_impact_view(self, index: int) -> None: ...

        def _draw(self) -> None: ...

        def _on_export_impact(self) -> None: ...

        def _on_play_toggled(self, playing: bool) -> None: ...

        def _on_slider_moved(self, value: int) -> None: ...

        def jump_to_inspection_event(self) -> None: ...

        def step_frames(self, frames: int) -> None: ...

    def _build_playback_controls(self) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setContentsMargins(4, 4, 4, 0)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(4)

        self._play_button = QPushButton("Play")
        self._play_button.setCheckable(True)
        self._play_button.setMinimumWidth(60)
        self._play_button.setToolTip("Play or pause swing and flight playback.")
        self._play_button.toggled.connect(self._on_play_toggled)

        self._step_back_button = QPushButton("−1")
        self._step_back_button.setToolTip("Step one sampled frame backward.")
        self._step_back_button.clicked.connect(lambda: self.step_frames(-1))
        self._step_forward_button = QPushButton("+1")
        self._step_forward_button.setToolTip("Step one sampled frame forward.")
        self._step_forward_button.clicked.connect(lambda: self.step_frames(1))
        self._inspection_button = QPushButton("Jump to Impact")
        self._inspection_button.setEnabled(False)
        self._inspection_button.setToolTip(
            "Jump to exact impact or the explicitly labelled closest approach."
        )
        self._inspection_button.clicked.connect(self.jump_to_inspection_event)
        self._time_label = QLabel("0.000 s")
        self._time_label.setMinimumWidth(68)

        layout.addWidget(self._play_button, 0, 0)
        layout.addWidget(self._step_back_button, 0, 1)
        layout.addWidget(self._step_forward_button, 0, 2)
        layout.addWidget(self._inspection_button, 0, 3)
        layout.addWidget(self._time_label, 0, 4)

        self._position_slider = QSlider(Qt.Orientation.Horizontal)
        self._position_slider.setRange(0, 1000)
        self._position_slider.setToolTip(
            "Scrub the playback instant across the swing and flight timeline."
        )
        self._position_slider.valueChanged.connect(self._on_slider_moved)
        self._loop_check = QCheckBox("Loop")
        self._loop_check.setToolTip("Restart playback when the timeline ends.")
        self._rate_combo = QComboBox()
        self._rate_combo.addItems([name for name, _rate in RATE_PRESETS])
        self._rate_combo.setCurrentIndex(3)
        self._rate_combo.setToolTip(
            "Playback rate; 1× maps animation time to simulated time."
        )
        layout.addWidget(self._position_slider, 1, 0, 1, 3)
        layout.addWidget(self._loop_check, 1, 3)
        rate_row = QHBoxLayout()
        rate_row.setContentsMargins(0, 0, 0, 0)
        rate_row.addWidget(QLabel("Rate"))
        rate_row.addWidget(self._rate_combo)
        layout.addLayout(rate_row, 1, 4)
        layout.setColumnStretch(3, 1)
        return panel

    def _build_layers_control(self) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(3)
        self._layers_button = QToolButton()
        self._layers_button.setText("Display & Layers")
        self._layers_button.setCheckable(True)
        self._layers_button.setArrowType(Qt.ArrowType.RightArrow)
        self._layers_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self._layers_button.setToolTip(
            "Show or hide scene layers, camera, legend, and export controls."
        )
        self._layers_button.toggled.connect(self._set_layers_expanded)
        layout.addWidget(self._layers_button)

        self._layers_panel = QWidget()
        grid = QGridLayout(self._layers_panel)
        grid.setContentsMargins(18, 2, 2, 4)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(3)
        self._create_scene_checks()
        for index, check in enumerate(self._layer_checkboxes()):
            grid.addWidget(check, index // 3, index % 3)

        row = (len(self._layer_checkboxes()) + 2) // 3
        self._screw_entity = QComboBox()
        self._screw_entity.addItem("Club", "club")
        self._screw_entity.setToolTip(
            "Choose the rigid body or modeled joint used for screw-axis analysis."
        )
        self._screw_entity.currentIndexChanged.connect(lambda _index: self._draw())
        self._impact_view = QComboBox()
        self._impact_view.setAccessibleName("Impact Camera View")
        self._impact_view.addItem("Isometric", (30.0, -60.0))
        self._impact_view.addItem("Face-On", (0.0, -90.0))
        self._impact_view.addItem("Down-the-Line", (10.0, 0.0))
        self._impact_view.setToolTip(
            "Choose a named engineering camera; the plot remains rotatable."
        )
        self._impact_view.currentIndexChanged.connect(self._apply_impact_view)
        self._impact_export_button = QPushButton("Export Impact…")
        self._impact_export_button.setEnabled(False)
        self._impact_export_button.setToolTip(
            "Export the impact still as PNG or SVG, or scene data as JSON."
        )
        self._impact_export_button.clicked.connect(self._on_export_impact)
        grid.addWidget(self._screw_entity, row, 0)
        grid.addWidget(self._impact_view, row, 1)
        grid.addWidget(self._impact_export_button, row, 2)

        self._legend_check = QCheckBox("Legend")
        self._legend_check.setChecked(True)
        self._legend_check.setToolTip("Show or hide the plot legend.")
        self._legend_check.toggled.connect(lambda _checked: self._draw())
        self._legend_position = QComboBox()
        self._legend_position.addItem("Outside right", "outside_right")
        self._legend_position.addItem("Inside upper right", "inside_upper_right")
        self._legend_position.addItem("Inside lower right", "inside_lower_right")
        self._legend_position.addItem("Inside lower left", "inside_lower_left")
        self._legend_position.setToolTip(
            "Place the legend outside the data region or in a selected corner."
        )
        self._legend_position.currentIndexChanged.connect(lambda _index: self._draw())
        grid.addWidget(self._legend_check, row + 1, 0)
        grid.addWidget(self._legend_position, row + 1, 1, 1, 2)
        self._layers_panel.setVisible(False)
        layout.addWidget(self._layers_panel)
        return wrapper

    def _create_scene_checks(self) -> None:
        specs = (
            ("_ball_check", "Ball", True, FIELD_GUIDANCE["ball_visible"]),
            ("_ground_check", "Ground", True, FIELD_GUIDANCE["ground_visible"]),
            ("_course_check", "Course", True, FIELD_GUIDANCE["course_visible"]),
            ("_screw_check", "Screw Axis", False, FIELD_GUIDANCE["screw_axis_visible"]),
            (
                "_impact_check",
                "Impact Inspector",
                True,
                "Show exact event-time club, contact, D-plane, and screw geometry.",
            ),
            ("_kinetics_check", "Kinetics", False, FIELD_GUIDANCE["kinetics_visible"]),
            (
                "_flight_check",
                "Ball Flight",
                False,
                "Show flight-scale geometry; this makes the swing appear much smaller. "
                + FIELD_GUIDANCE["swing_flight_toggle"],
            ),
        )
        for name, label, checked, tooltip in specs:
            check = QCheckBox(label)
            check.setChecked(checked)
            check.setToolTip(tooltip)
            check.toggled.connect(lambda _checked: self._draw())
            setattr(self, name, check)

    def _layer_checkboxes(self) -> tuple[QCheckBox, ...]:
        """Return scene checkboxes laid out without clipped text."""
        return (
            self._ball_check,
            self._ground_check,
            self._course_check,
            self._screw_check,
            self._impact_check,
            self._kinetics_check,
            self._flight_check,
            *self._impact_layer_checks.values(),
        )

    def _build_engineering_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(3)
        self._impact_summary = QLabel("Run a simulation to inspect key impact metrics.")
        self._impact_summary.setWordWrap(True)
        self._impact_summary.setAccessibleName("Key Impact Metrics")
        self._impact_summary.setToolTip(
            "A compact summary of the current calculation; expand details for "
            "provenance."
        )
        layout.addWidget(self._impact_summary)
        self._details_button = QToolButton()
        self._details_button.setText("Engineering Details")
        self._details_button.setCheckable(True)
        self._details_button.setArrowType(Qt.ArrowType.RightArrow)
        self._details_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self._details_button.setToolTip(
            "Show detailed impact, D-plane, screw-axis, and ground-clearance metrics."
        )
        self._details_button.toggled.connect(self._set_details_expanded)
        layout.addWidget(self._details_button)

        content = QWidget()
        details = QVBoxLayout(content)
        details.setContentsMargins(2, 2, 2, 2)
        self._screw_readout = QLabel()
        self._screw_readout.setWordWrap(True)
        self._screw_readout.setVisible(False)
        self._screw_readout.setToolTip(
            "Screw-motion readout in app frame x target, y up, z right."
        )
        details.addWidget(self._screw_readout)
        self._impact_kinematics_readout = QLabel(
            "Run a simulation to inspect impact kinematics."
        )
        self._impact_kinematics_readout.setWordWrap(True)
        self._impact_kinematics_readout.setTextFormat(Qt.TextFormat.RichText)
        self._impact_kinematics_readout.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._impact_kinematics_readout.setAccessibleName(
            "Impact and Wedge Engineering Readout"
        )
        self._impact_kinematics_readout.setToolTip(
            "Detailed frame-explicit metrics, provenance, and model boundaries."
        )
        details.addWidget(self._impact_kinematics_readout)
        details.addStretch(1)

        self._impact_details_scroll = QScrollArea()
        self._impact_details_scroll.setWidgetResizable(True)
        self._impact_details_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._impact_details_scroll.setMaximumHeight(150)
        self._impact_details_scroll.setWidget(content)
        self._impact_details_scroll.setVisible(False)
        self._impact_details_scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        layout.addWidget(self._impact_details_scroll)
        return panel

    def _set_layers_expanded(self, expanded: bool) -> None:
        self._layers_panel.setVisible(expanded)
        self._layers_button.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )

    def _set_details_expanded(self, expanded: bool) -> None:
        self._impact_details_scroll.setVisible(expanded)
        self._details_button.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )

    def impact_visible_layers(self) -> frozenset[str]:
        """Return independently toggleable persisted impact layers."""
        layers: frozenset[str] = self._impact_layer_controls.visible_layers()
        return layers

    def legend_visible(self) -> bool:
        """Return whether the plot legend is enabled."""
        return bool(self._legend_check.isChecked())

    def legend_location(self) -> str:
        """Return the stable legend-placement key."""
        return str(self._legend_position.currentData())


__all__ = ["SimulationViewControlsMixin"]
