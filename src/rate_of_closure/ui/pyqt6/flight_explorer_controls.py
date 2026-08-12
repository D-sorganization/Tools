"""Field definitions and small control factories for the flight explorer."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QWhatsThis,
    QWidget,
)

from rate_of_closure.model import MPH_PER_MPS

#: Entry modes, in combo order.
ENTRY_MODES: tuple[str, ...] = ("Direct Launch Conditions", "Impact Delivery")

#: Speed display units: label -> factor from displayed to m/s.
SPEED_UNITS: dict[str, float] = {"mph": 1.0 / MPH_PER_MPS, "m/s": 1.0}

#: (metric key, Title Case label, unit suffix) result rows in display order.
EXPLORER_ROWS: tuple[tuple[str, str, str], ...] = (
    ("carry_m", "Carry Distance", " m"),
    ("max_height_m", "Apex Height", " m"),
    ("flight_time_s", "Flight Time", " s"),
    ("landing_angle_deg", "Landing Angle", "°"),
    ("lateral_m", "Lateral Landing Offset", " m"),
)

#: Rows following the user's distance display unit. Apex stays in metres.
DISTANCE_ROWS: frozenset[str] = frozenset({"carry_m", "lateral_m"})

#: Field tuple: attr, label, guidance key, low, high, default, decimals, suffix.
DIRECT_FIELDS: tuple[tuple[str, str, str, float, float, float, int, str], ...] = (
    ("launch_angle_deg", "Launch Angle", "fx_launch_angle", -89.0, 89.0, 10.9, 1, "°"),
    (
        "launch_direction_deg",
        "Launch Direction",
        "fx_launch_direction",
        -45.0,
        45.0,
        0.0,
        1,
        "°",
    ),
    ("spin_rpm", "Total Spin", "fx_spin_rpm", 0.0, 15000.0, 2686.0, 0, " rpm"),
    (
        "spin_axis_tilt_deg",
        "Spin-Axis Tilt",
        "fx_spin_axis_tilt",
        -60.0,
        60.0,
        0.0,
        1,
        "°",
    ),
)

#: Delivery-mode fields in the same format as :data:`DIRECT_FIELDS`.
DELIVERY_FIELDS: tuple[tuple[str, str, str, float, float, float, int, str], ...] = (
    ("club_path_deg", "Club Path", "fx_club_path", -45.0, 45.0, 0.0, 1, "°"),
    ("face_angle_deg", "Face Angle", "fx_face_angle", -45.0, 45.0, 0.0, 1, "°"),
    ("attack_angle_deg", "Attack Angle", "fx_attack_angle", -20.0, 20.0, -1.0, 1, "°"),
    ("dynamic_loft_deg", "Dynamic Loft", "fx_dynamic_loft", 0.0, 70.0, 12.0, 1, "°"),
    (
        "impact_offset_toe_mm",
        "Impact Toward Toe",
        "impact_offset_toe_mm",
        -30.0,
        30.0,
        0.0,
        1,
        " mm",
    ),
    (
        "impact_offset_high_mm",
        "Impact Above Center",
        "impact_offset_high_mm",
        -30.0,
        30.0,
        0.0,
        1,
        " mm",
    ),
)


def make_spin(
    low: float, high: float, default: float, decimals: int, suffix: str, tooltip: str
) -> QDoubleSpinBox:
    """Return a typed entry spin box in the explorer's house style."""
    spin = QDoubleSpinBox()
    spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
    spin.setKeyboardTracking(False)
    spin.setDecimals(decimals)
    spin.setRange(low, high)
    spin.setValue(default)
    spin.setSuffix(suffix)
    spin.setToolTip(tooltip)
    spin.setMinimumWidth(84)
    return spin


def field_label(label: str, attr: str, guidance: str) -> QWidget:
    """Return a visibly clickable field label with non-modal guidance."""
    container = QWidget()
    row = QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)
    row.addWidget(QLabel(label))
    button = QToolButton()
    button.setText("Details")
    button.setAutoRaise(True)
    button.setObjectName(f"{attr.removesuffix('_deg')}_info")
    button.setAccessibleName(f"Explain {label}")
    button.setAccessibleDescription(guidance)
    button.setToolTip(guidance)
    button.clicked.connect(
        lambda _checked=False: QWhatsThis.showText(
            button.mapToGlobal(button.rect().bottomLeft()), guidance, button
        )
    )
    row.addWidget(button)
    return container
