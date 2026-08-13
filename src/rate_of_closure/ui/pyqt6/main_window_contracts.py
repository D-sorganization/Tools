"""Immutable presentation tables used by the PyQt main window."""

from rate_of_closure.ui.pyqt6.workspace_navigation import _DEFAULT_TAB_IDS

#: (result field, Title Case label) in display order. Every field must have an
#: entry in RESULT_EXPLANATIONS; the help-content tests enforce that contract.
_RESULT_ROWS: tuple[tuple[str, str], ...] = (
    ("path_deviation_deg", "Impact-Point Path vs Reference"),
    ("aoa_deviation_deg", "Attack-Angle Change"),
    ("tangential_speed_mph", "Rotation-Induced Velocity"),
    ("speed_delta_mph", "Delivered Speed Change"),
    ("closure_rate_dps", "Closure Rate (CCV)"),
    ("normalized_closure_deg_per_ft", "Normalized Closure"),
    ("closure_during_contact_deg", "Face Closure During Contact"),
    ("loft_gain_during_contact_deg", "Dynamic Loft Gained During Contact"),
)

_METRIC_ROWS: tuple[tuple[str, str], ...] = (
    ("ccv_dps", "Club Closure Velocity (CCV)"),
    ("closure_deg_per_ft", "Closure per Foot of Travel"),
    ("closure_deg_per_inch", "Closure per Inch of Travel"),
    ("closure_deg_per_ms", "Closure per Millisecond"),
    ("r_isa_ft", "Distance to Screw Axis (R_ISA)"),
    ("r_isa_m", "Distance to Screw Axis (Metric)"),
    ("time_to_square_from_1deg_open_ms", "Time to Square From 1° Open"),
    ("toe_heel_speed_delta_mph", "Toe vs Heel Speed Difference"),
)

#: Fixed suffixes; fields in _QUANTITY_ROWS use the selected display unit.
_UNITS: dict[str, str] = {
    "path_deviation_deg": "°",
    "aoa_deviation_deg": "°",
    "normalized_closure_deg_per_ft": " °/ft",
    "closure_during_contact_deg": "°",
    "loft_gain_during_contact_deg": "°",
    "closure_deg_per_ft": " °/ft",
    "closure_deg_per_inch": " °/in",
    "closure_deg_per_ms": " °/ms",
    "r_isa_ft": " ft",
    "r_isa_m": " m",
    "time_to_square_from_1deg_open_ms": " ms",
}

_QUANTITY_ROWS: dict[str, str] = {
    "tangential_speed_mph": "speed",
    "speed_delta_mph": "speed",
    "toe_heel_speed_delta_mph": "speed",
    "closure_rate_dps": "rotation",
    "ccv_dps": "rotation",
}

#: Compatibility export used by the help-content contract tests.
_TAB_HELP_KEYS = _DEFAULT_TAB_IDS

__all__ = [
    "_METRIC_ROWS",
    "_QUANTITY_ROWS",
    "_RESULT_ROWS",
    "_TAB_HELP_KEYS",
    "_UNITS",
]
