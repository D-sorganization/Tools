"""Results display widget for Club Tester and Heavy Hit coupling (C6, H4)."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.ui.pyqt6.club_tester_models import ClubTesterExecutionResult

__all__ = ["ClubTesterResultsPanel"]

_METRIC_ROWS: tuple[tuple[str, str, str], ...] = (
    ("Delivered Loft", "delivered_loft_deg", "°"),
    ("Face Angle (+ Open)", "face_angle_deg", "°"),
    ("Lie Toe-Down", "lie_toe_down_deg", "°"),
    ("Clubhead Speed", "clubhead_speed_mps", " m/s"),
    ("Ball Exit Speed", "ball_speed_mps", " m/s"),
    ("Launch Angle", "launch_angle_deg", "°"),
    ("Backspin Rate", "backspin_rpm", " rpm"),
    ("Carry Distance", "carry_m", " m"),
    ("Max Apex Height", "max_height_m", " m"),
    ("Flight Time", "flight_time_s", " s"),
    ("Lateral Deviation", "lateral_m", " m"),
)


class ClubTesterResultsPanel(QWidget):
    """Side-by-side comparison table, shaft deltas, and heavy-hit coupling readout."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        layout.addWidget(self._build_table_box())
        layout.addWidget(self._build_shaft_box())
        layout.addWidget(self._build_coupling_box())
        layout.addWidget(self._build_json_box())

    def _build_table_box(self) -> QGroupBox:
        box = QGroupBox("Outcome Comparison (Baseline vs Counterfactual)")
        layout = QVBoxLayout(box)

        self._table = QTableWidget(len(_METRIC_ROWS), 4)
        self._table.setAccessibleName("Club Tester Outcome Comparison Table")
        self._table.setToolTip(
            "Side-by-side simulation outcomes for baseline club vs counterfactual."
        )
        self._table.setHorizontalHeaderLabels(
            ["Metric", "Baseline", "Counterfactual", "Delta"]
        )
        header = self._table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        vertical_header = self._table.verticalHeader()
        if vertical_header is not None:
            vertical_header.setVisible(False)
        self._table.setAlternatingRowColors(True)

        align = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        for row_idx, (label, _attr, _suffix) in enumerate(_METRIC_ROWS):
            item = QTableWidgetItem(label)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row_idx, 0, item)
            for col_idx in (1, 2, 3):
                val_item = QTableWidgetItem("—")
                val_item.setTextAlignment(align)
                val_item.setFlags(val_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._table.setItem(row_idx, col_idx, val_item)

        layout.addWidget(self._table)
        return box

    def _build_shaft_box(self) -> QGroupBox:
        box = QGroupBox("Delivered Shaft Dynamics")
        layout = QHBoxLayout(box)

        self._dynamic_loft_lbl = QLabel("Dynamic Loft Add: —")
        self._dynamic_loft_lbl.setAccessibleName("Shaft Dynamic Loft Add")
        layout.addWidget(self._dynamic_loft_lbl)

        self._face_closure_lbl = QLabel("Face Closure: —")
        self._face_closure_lbl.setAccessibleName("Shaft Face Closure")
        layout.addWidget(self._face_closure_lbl)

        self._kick_speed_lbl = QLabel("Kick Speed: —")
        self._kick_speed_lbl.setAccessibleName("Shaft Kick Speed")
        layout.addWidget(self._kick_speed_lbl)

        self._first_mode_lbl = QLabel("1st Mode: —")
        self._first_mode_lbl.setAccessibleName("Shaft First Bending Mode")
        layout.addWidget(self._first_mode_lbl)

        return box

    def _build_coupling_box(self) -> QGroupBox:
        box = QGroupBox("Heavy Hit Impact Coupling Analysis")
        layout = QVBoxLayout(box)

        top_row = QHBoxLayout()
        self._decoupling_lbl = QLabel("Decoupling Fraction: —")
        self._decoupling_lbl.setStyleSheet(
            "font-weight: bold; font-size: 13px; color: #10b981;"
        )
        self._decoupling_lbl.setAccessibleName("Decoupling Fraction")
        top_row.addWidget(self._decoupling_lbl)

        self._rigid_bound_lbl = QLabel("Rigid-Shaft Upper Bound: —")
        self._rigid_bound_lbl.setAccessibleName("Rigid Shaft Ball Speed Upper Bound")
        top_row.addWidget(self._rigid_bound_lbl)
        layout.addLayout(top_row)

        details_row = QHBoxLayout()
        self._speeds_lbl = QLabel("Coupled Ball Speed: — | Free-Head: —")
        self._speeds_lbl.setAccessibleName("Coupled vs Free-Head Speeds")
        details_row.addWidget(self._speeds_lbl)

        self._contact_force_lbl = QLabel("Peak Force: — | Contact Time: —")
        self._contact_force_lbl.setAccessibleName("Contact Force and Duration")
        details_row.addWidget(self._contact_force_lbl)
        layout.addLayout(details_row)

        self._provenance_lbl = QLabel("Golfer Boundary: —")
        self._provenance_lbl.setAccessibleName("Golfer Boundary Provenance")
        self._provenance_lbl.setStyleSheet("color: #64748b; font-size: 11px;")
        layout.addWidget(self._provenance_lbl)

        return box

    def _build_json_box(self) -> QGroupBox:
        box = QGroupBox("Report Export / Wire Summary")
        layout = QVBoxLayout(box)

        self._json_view = QTextEdit()
        self._json_view.setReadOnly(True)
        self._json_view.setAccessibleName("Fitting Report JSON Preview")
        self._json_view.setToolTip(
            "Deterministic JSON representation of the latest fitting analysis."
        )
        self._json_view.setFixedHeight(120)
        layout.addWidget(self._json_view)
        return box

    def display_results(
        self, result: ClubTesterExecutionResult, json_text: str = ""
    ) -> None:
        """Populate table and summary cards from execution results."""
        baseline = result.report.baseline
        cf = (
            result.report.counterfactuals[0]
            if result.report.counterfactuals
            else baseline
        )

        for row_idx, (_label, attr, suffix) in enumerate(_METRIC_ROWS):
            b_val = getattr(baseline, attr)
            c_val = getattr(cf, attr)
            delta = c_val - b_val

            fmt = (
                "{:.2f}" if "mps" in attr or "m" in attr or "deg" in attr else "{:.1f}"
            )
            item_b = self._table.item(row_idx, 1)
            if item_b is not None:
                item_b.setText(f"{fmt.format(b_val)}{suffix}")
            item_c = self._table.item(row_idx, 2)
            if item_c is not None:
                item_c.setText(f"{fmt.format(c_val)}{suffix}")
            delta_str = f"{'+' if delta > 0 else ''}{fmt.format(delta)}{suffix}"
            item_d = self._table.item(row_idx, 3)
            if item_d is not None:
                item_d.setText(delta_str)

        # Shaft dynamics
        shaft = cf.shaft
        self._dynamic_loft_lbl.setText(
            f"Dynamic Loft: +{shaft.dynamic_loft_add_deg:.2f}°"
        )
        self._face_closure_lbl.setText(f"Face Closure: {shaft.face_closure_deg:.2f}°")
        self._kick_speed_lbl.setText(f"Kick Speed: +{shaft.kick_speed_mps:.2f} m/s")
        self._first_mode_lbl.setText(f"1st Mode: {shaft.first_mode_hz:.1f} Hz")

        # Heavy hit
        if result.coupled_result:
            cr = result.coupled_result
            pct = cr.decoupling_fraction * 100.0
            self._decoupling_lbl.setText(f"Decoupling Fraction: {pct:.2f}% (Decoupled)")
            self._speeds_lbl.setText(
                f"Coupled: {cr.ball_speed_mps:.2f} m/s | "
                f"Free: {cr.free_head_ball_speed_mps:.2f} m/s"
            )
            force_kn = cr.peak_contact_force_n / 1000.0
            time_us = cr.contact_time_s * 1e6
            self._contact_force_lbl.setText(
                f"Peak Force: {force_kn:.2f} kN | Contact Time: {time_us:.1f} µs"
            )
            self._provenance_lbl.setText(f"Golfer Boundary: {cr.grip_provenance}")

            if result.rigid_shaft_ball_speed_mps:
                self._rigid_bound_lbl.setText(
                    f"Rigid-Shaft Bound: {result.rigid_shaft_ball_speed_mps:.2f} m/s"
                )
        else:
            self._decoupling_lbl.setText("Decoupling Fraction: N/A (Coupling Disabled)")
            self._speeds_lbl.setText("Coupled Ball Speed: — | Free-Head: —")
            self._contact_force_lbl.setText("Peak Force: — | Contact Time: —")
            self._provenance_lbl.setText("Golfer Boundary: —")
            self._rigid_bound_lbl.setText("Rigid-Shaft Bound: —")

        if json_text:
            self._json_view.setPlainText(json_text)
