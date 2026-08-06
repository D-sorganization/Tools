"""Data catalog: every plottable variable of a SimulationRun.

The registry maps a stable namespaced key (``category.name``) to a
:class:`VariableSpec` describing the variable — Title Case label, unit,
category, an extractor callable ``run -> np.ndarray | float``, and an
axis-scale hint. Array-valued categories (Swing Sample, Flight) yield
per-sample series; the scalar categories (Input, Impact, Launch,
Metric) yield one number per run and become plottable series through
the sweep pipeline (:mod:`rate_of_closure.plotting.render`).

The key list is pinned by the contract test and mirrored key-for-key by
the web catalog (``web/src/model/plotcatalog.ts``) through the exported
parity fixture.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from rate_of_closure._contracts import ensure, require
from rate_of_closure.model import solve
from rate_of_closure.plotting import optional_values
from rate_of_closure.simulation.kinetics import KineticsSeries, kinetics_for_run
from rate_of_closure.simulation.session import SimulationRun

__all__ = [
    "CATALOG",
    "CATEGORIES",
    "VariableSpec",
    "catalog_keys",
    "extract",
    "variables_by_category",
]

#: Catalog categories in display order. Array-valued (per-sample)
#: categories first, scalar (per-run) categories after.
CATEGORIES: tuple[str, ...] = (
    "Input",
    "Swing Sample",
    "Kinetics",
    "Impact",
    "Launch",
    "Flight",
    "Metric",
)

#: Categories whose extractors return per-sample arrays.
SERIES_CATEGORIES: frozenset[str] = frozenset({"Swing Sample", "Kinetics", "Flight"})

#: Axis-scale hints accepted by :class:`VariableSpec`.
SCALE_HINTS: tuple[str, ...] = ("linear", "log")

Extractor = Callable[[SimulationRun], "np.ndarray | float"]


@dataclass(frozen=True)
class VariableSpec:
    """One plottable variable.

    Args:
        key: Stable namespaced identifier, ``category.name``.
        label: Title Case display label.
        unit: Display unit string ("" for dimensionless).
        category: One of :data:`CATEGORIES`.
        extractor: ``run -> np.ndarray | float`` in the app frame / SI
            or stated display unit.
        scale: Axis-scale hint, ``"linear"`` or ``"log"``.
    """

    key: str
    label: str
    unit: str
    category: str
    extractor: Extractor
    scale: str = "linear"

    def __post_init__(self) -> None:
        require(
            bool(self.key) and "." in self.key,
            "key must be namespaced as 'category.name'",
            self.key,
        )
        require(bool(self.label), "label must be non-empty", self.label)
        require(self.category in CATEGORIES, "unknown category", self.category)
        require(self.scale in SCALE_HINTS, "unknown scale hint", self.scale)
        require(callable(self.extractor), "extractor must be callable")

    @property
    def is_series(self) -> bool:
        """True when the extractor yields a per-sample array."""
        return self.category in SERIES_CATEGORIES

    @property
    def axis_label(self) -> str:
        """Axis label: ``Label [unit]`` (unit omitted when empty)."""
        return f"{self.label} [{self.unit}]" if self.unit else self.label


#: Ball-flight distance variables that follow the user's Distance
#: display unit (#4125 H6 — yards default). Heights (flight.y_m,
#: metric.max_height_m) and swing-scale positions stay in metres.
DISTANCE_KEYS: frozenset[str] = frozenset(
    {
        "flight.x_m",
        "flight.z_m",
        "metric.carry_m",
        "putting.path_x",
        "putting.path_y",
        "putting.rollout",
        "putting.skid_distance",
        "putting.break",
    }
)


def _speed_series(vectors: np.ndarray) -> np.ndarray:
    return np.asarray(np.linalg.norm(vectors, axis=1), dtype=float)


def _kinetics_series(picker: Callable[[KineticsSeries], np.ndarray]) -> Extractor:
    """Kinetics extractor factory: joint kinetics need the pendulum
    joint states, so sources without them (manual, triple pendulum)
    yield an all-NaN series of matching length (#4125 H2 — the plots
    render empty rather than lying)."""

    def _extract(run: SimulationRun) -> np.ndarray:
        series = kinetics_for_run(run)
        if series is None:
            return np.full(run.swing_times.shape[0], np.nan)
        return np.asarray(picker(series), dtype=float)

    return _extract


def _entries() -> list[VariableSpec]:
    """Build the full registry (one literal list, easy to review)."""
    inputs: list[tuple[str, str, str, Extractor]] = [
        (
            "clubhead_speed_mph",
            "Clubhead Speed",
            "mph",
            lambda r: float(r.config.scenario.clubhead_speed_mph),
        ),
        (
            "omega_plane_dps",
            "In-Plane Rotation (SPV)",
            "deg/s",
            lambda r: float(r.config.scenario.omega_plane_dps),
        ),
        (
            "omega_shaft_dps",
            "About-Shaft Rotation (HTV)",
            "deg/s",
            lambda r: float(r.config.scenario.omega_shaft_dps),
        ),
        (
            "lie_angle_deg",
            "Shaft Lie at Impact",
            "deg",
            lambda r: float(r.config.scenario.lie_angle_deg),
        ),
        (
            "com_to_face_mm",
            "GC to Face Center",
            "mm",
            lambda r: float(r.config.scenario.com_to_face_mm),
        ),
        (
            "impact_offset_toe_mm",
            "Impact Toward Toe",
            "mm",
            lambda r: float(r.config.scenario.impact_offset_toe_mm),
        ),
        (
            "impact_offset_high_mm",
            "Impact Above Center",
            "mm",
            lambda r: float(r.config.scenario.impact_offset_high_mm),
        ),
        (
            "contact_duration_us",
            "Contact Duration",
            "µs",
            lambda r: float(r.config.scenario.contact_duration_us),
        ),
        ("plane_yaw_deg", "Plane Yaw", "deg", lambda r: float(r.config.plane.yaw_deg)),
        (
            "plane_side_tilt_deg",
            "Plane Side Tilt",
            "deg",
            lambda r: float(r.config.plane.side_tilt_deg),
        ),
        (
            "plane_forward_tilt_deg",
            "Plane Forward Tilt",
            "deg",
            lambda r: float(r.config.plane.forward_tilt_deg),
        ),
        (
            "impact_time_s",
            "Impact Time (τ)",
            "s",
            lambda r: optional_values.optional_float(r.impact_time_s),
        ),
    ]
    swing: list[tuple[str, str, str, Extractor]] = [
        ("time_s", "Swing Time", "s", lambda r: np.asarray(r.swing_times, float)),
        (
            "x_m",
            "Clubhead X (Target Line)",
            "m",
            lambda r: np.asarray(r.swing_positions[:, 0], float),
        ),
        (
            "y_m",
            "Clubhead Y (Up)",
            "m",
            lambda r: np.asarray(r.swing_positions[:, 1], float),
        ),
        (
            "z_m",
            "Clubhead Z (Right)",
            "m",
            lambda r: np.asarray(r.swing_positions[:, 2], float),
        ),
        (
            "speed_mps",
            "Clubhead Speed",
            "m/s",
            lambda r: _speed_series(r.swing_twists[:, 3:]),
        ),
        (
            "angular_speed_dps",
            "Clubhead Angular Speed",
            "deg/s",
            lambda r: np.degrees(_speed_series(r.swing_twists[:, :3])),
        ),
    ]
    kinetics: list[tuple[str, str, str, Extractor]] = [
        (
            "shoulder_torque_nm",
            "Shoulder Net Torque",
            "N·m",
            _kinetics_series(lambda k: k.torque_inertial_nm[:, 0]),
        ),
        (
            "wrist_torque_nm",
            "Wrist Net Torque",
            "N·m",
            _kinetics_series(lambda k: k.torque_inertial_nm[:, 1]),
        ),
        (
            "shoulder_gravity_torque_nm",
            "Shoulder Gravity Torque",
            "N·m",
            _kinetics_series(lambda k: k.torque_gravity_nm[:, 0]),
        ),
        (
            "wrist_gravity_torque_nm",
            "Wrist Gravity Torque",
            "N·m",
            _kinetics_series(lambda k: k.torque_gravity_nm[:, 1]),
        ),
        (
            "shoulder_damping_torque_nm",
            "Shoulder Damping Torque",
            "N·m",
            _kinetics_series(lambda k: k.torque_damping_nm[:, 0]),
        ),
        (
            "wrist_damping_torque_nm",
            "Wrist Damping Torque",
            "N·m",
            _kinetics_series(lambda k: k.torque_damping_nm[:, 1]),
        ),
        (
            "shoulder_ztcf_torque_nm",
            "Shoulder ZTCF Inertial Torque",
            "N·m",
            _kinetics_series(lambda k: k.ztcf_inertial_torque_nm[:, 0]),
        ),
        (
            "wrist_ztcf_torque_nm",
            "Wrist ZTCF Inertial Torque",
            "N·m",
            _kinetics_series(lambda k: k.ztcf_inertial_torque_nm[:, 1]),
        ),
        (
            "shoulder_power_w",
            "Shoulder Power",
            "W",
            _kinetics_series(lambda k: k.power_w[:, 0]),
        ),
        (
            "wrist_power_w",
            "Wrist Power",
            "W",
            _kinetics_series(lambda k: k.power_w[:, 1]),
        ),
        (
            "shoulder_force_n",
            "Shoulder Reaction Force",
            "N",
            _kinetics_series(lambda k: k.force_magnitude_n("shoulder")),
        ),
        (
            "wrist_force_n",
            "Wrist Reaction Force",
            "N",
            _kinetics_series(lambda k: k.force_magnitude_n("wrist")),
        ),
        (
            "clubhead_force_n",
            "Clubhead Force",
            "N",
            _kinetics_series(lambda k: k.force_magnitude_n("clubhead")),
        ),
        (
            "shoulder_ztcf_force_n",
            "Shoulder ZTCF Reaction Force",
            "N",
            _kinetics_series(lambda k: k.ztcf_force_magnitude_n("shoulder")),
        ),
        (
            "wrist_ztcf_force_n",
            "Wrist ZTCF Reaction Force",
            "N",
            _kinetics_series(lambda k: k.ztcf_force_magnitude_n("wrist")),
        ),
        (
            "clubhead_ztcf_force_n",
            "Clubhead ZTCF Force",
            "N",
            _kinetics_series(lambda k: k.ztcf_force_magnitude_n("clubhead")),
        ),
    ]
    impact: list[tuple[str, str, str, Extractor]] = [
        (
            "clubhead_speed_mps",
            "Delivered Clubhead Speed",
            "m/s",
            optional_values.delivered_speed,
        ),
        ("club_path_deg", "Club Path", "deg", optional_values.path_deg),
        (
            "attack_angle_deg",
            "Attack Angle",
            "deg",
            optional_values.attack_angle_deg,
        ),
        (
            "spin_loft_deg",
            "Spin Loft",
            "deg",
            lambda r: optional_values.delivery_scalar(r, "spin_loft_deg"),
        ),
        (
            "face_to_path_deg",
            "Face to Path",
            "deg",
            lambda r: optional_values.delivery_scalar(r, "face_to_path_deg"),
        ),
        (
            "spin_axis_tilt_deg",
            "Spin Axis Tilt",
            "deg",
            lambda r: optional_values.delivery_scalar(r, "spin_axis_tilt_deg"),
        ),
        (
            "energy_transfer_j",
            "Impact Energy Transfer",
            "J",
            optional_values.impact_energy,
        ),
    ]
    launch: list[tuple[str, str, str, Extractor]] = [
        (
            "ball_speed_mph",
            "Ball Speed",
            "mph",
            lambda r: optional_values.launch_scalar(r, "ball_speed_mph"),
        ),
        (
            "launch_angle_deg",
            "Launch Angle",
            "deg",
            lambda r: optional_values.launch_scalar(r, "launch_angle_deg"),
        ),
        (
            "launch_azimuth_deg",
            "Launch Direction",
            "deg",
            lambda r: optional_values.launch_scalar(r, "launch_azimuth_deg"),
        ),
        (
            "spin_rpm",
            "Total Spin",
            "rpm",
            lambda r: optional_values.launch_scalar(r, "spin_rpm"),
        ),
    ]
    flight: list[tuple[str, str, str, Extractor]] = [
        ("time_s", "Flight Time", "s", lambda r: np.asarray(r.flight_times, float)),
        (
            "x_m",
            "Downrange Distance",
            "m",
            lambda r: np.asarray(r.flight_positions[:, 0], float),
        ),
        ("y_m", "Height", "m", lambda r: np.asarray(r.flight_positions[:, 1], float)),
        (
            "z_m",
            "Lateral (Right of Target)",
            "m",
            lambda r: np.asarray(r.flight_positions[:, 2], float),
        ),
        (
            "speed_mps",
            "Ball Speed",
            "m/s",
            lambda r: _speed_series(r.flight_velocities),
        ),
    ]
    metric: list[tuple[str, str, str, Extractor]] = [
        (
            "carry_m",
            "Carry Distance",
            "m",
            lambda r: optional_values.launch_scalar(r, "carry_m"),
        ),
        (
            "max_height_m",
            "Apex Height",
            "m",
            lambda r: optional_values.launch_scalar(r, "max_height_m"),
        ),
        (
            "flight_time_s",
            "Flight Time",
            "s",
            lambda r: optional_values.launch_scalar(r, "flight_time_s"),
        ),
        (
            "landing_angle_deg",
            "Landing Angle",
            "deg",
            lambda r: optional_values.launch_scalar(r, "landing_angle_deg"),
        ),
        (
            "path_deviation_deg",
            "Impact-Point Path Deviation",
            "deg",
            lambda r: float(solve(r.config.scenario).path_deviation_deg),
        ),
        (
            "closure_rate_dps",
            "Closure Rate (CCV)",
            "deg/s",
            lambda r: float(solve(r.config.scenario).closure_rate_dps),
        ),
    ]
    groups: list[tuple[str, str, list[tuple[str, str, str, Extractor]]]] = [
        ("input", "Input", inputs),
        ("swing", "Swing Sample", swing),
        ("kinetics", "Kinetics", kinetics),
        ("impact", "Impact", impact),
        ("launch", "Launch", launch),
        ("flight", "Flight", flight),
        ("metric", "Metric", metric),
    ]
    return [
        VariableSpec(
            key=f"{prefix}.{name}",
            label=label,
            unit=unit,
            category=category,
            extractor=extractor,
        )
        for prefix, category, rows in groups
        for name, label, unit, extractor in rows
    ]


#: The registry, keyed by namespaced variable key, in display order.
CATALOG: dict[str, VariableSpec] = {spec.key: spec for spec in _entries()}
ensure(len(CATALOG) == len(_entries()), "catalog keys must be unique")


def catalog_keys() -> tuple[str, ...]:
    """All catalog keys in display order (pinned by the contract test)."""
    return tuple(CATALOG)


def variables_by_category(category: str) -> tuple[VariableSpec, ...]:
    """The catalog entries of one category, in display order.

    Args:
        category: One of :data:`CATEGORIES`.

    Returns:
        The matching :class:`VariableSpec` entries.
    """
    require(category in CATEGORIES, "unknown category", category)
    return tuple(s for s in CATALOG.values() if s.category == category)


def extract(run: SimulationRun, key: str) -> np.ndarray | float:
    """Extract one variable from a run.

    Args:
        run: The simulation run.
        key: A catalog key.

    Returns:
        A float for scalar categories, a 1-D array for series
        categories.
    """
    require(isinstance(run, SimulationRun), "run must be a SimulationRun", run)
    require(key in CATALOG, f"unknown catalog key {key!r}", key)
    spec = CATALOG[key]
    value = spec.extractor(run)
    if spec.is_series:
        array = np.asarray(value, dtype=float)
        ensure(array.ndim == 1, f"{key} extractor must yield a 1-D array")
        return array
    ensure(isinstance(value, float), f"{key} extractor must yield a float")
    return value
