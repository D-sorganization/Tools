"""Editor specs for the Solver panel (epic #4103, #4109 / #4110).

Declarative tables driving the goal editor and the variable-partition
editor of :class:`~rate_of_closure.ui.pyqt6.solver_panel.SolverPanel`:
one row per solver goal quantity and one row per solver variable, each
with a Title Case label, display unit, spin ranges, defaults, and
sourced hover guidance in the FIELD_GUIDANCE style ("Suggested range
... Source: ...").

Names match ``shared.python.swing_sim.solver.goals`` exactly
(test-enforced): the goal table covers ``GOAL_QUANTITIES`` and the
variable table covers ``DELIVERY_VARIABLE_DEFAULTS`` plus
``SWING_VARIABLE_DEFAULTS``.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["GOAL_SPECS", "VARIABLE_SPECS", "GoalSpec", "VariableSpec"]


@dataclass(frozen=True)
class GoalSpec:
    """One goal-quantity row: label, unit, editor ranges, guidance."""

    name: str
    label: str
    unit: str
    default_target: float
    spin_range: tuple[float, float]
    guidance: str


@dataclass(frozen=True)
class VariableSpec:
    """One variable row: label, unit, default value/bounds, guidance."""

    name: str
    label: str
    unit: str
    default_value: float
    default_bounds: tuple[float, float]
    spin_range: tuple[float, float]
    guidance: str
    decimals: int = 1
    swing_only: bool = False


GOAL_SPECS: tuple[GoalSpec, ...] = (
    GoalSpec(
        "club_path_deg",
        "Club Path",
        "°",
        0.0,
        (-45.0, 45.0),
        "Suggested range: -8 to +8 deg (positive = in-to-out); tour "
        "driver means sit within a few degrees of zero. Source: openly "
        "published tour launch-monitor averages.",
    ),
    GoalSpec(
        "face_angle_deg",
        "Face Angle",
        "°",
        0.0,
        (-45.0, 45.0),
        "Suggested range: -5 to +5 deg at impact (positive = open, "
        "right of target for a right-handed player). Source: standard "
        "launch-monitor sign conventions and published tour data.",
    ),
    GoalSpec(
        "attack_angle_deg",
        "Attack Angle",
        "°",
        -1.0,
        (-30.0, 30.0),
        "Suggested range: -5 to +5 deg for drivers (tour mean near "
        "-1 deg; distance-optimised swings hit up to +5). Source: "
        "openly published tour launch-monitor averages.",
    ),
    GoalSpec(
        "dynamic_loft_deg",
        "Dynamic Loft",
        "°",
        12.0,
        (0.0, 60.0),
        "Suggested range: 8-16 deg delivered loft for drivers, up to "
        "the static loft plus shaft lean for irons and wedges. Source: "
        "published launch-monitor fitting references.",
    ),
    GoalSpec(
        "ball_speed_mph",
        "Ball Speed",
        " mph",
        150.0,
        (0.0, 250.0),
        "Suggested range: 120-185 mph driver ball speed (tour average "
        "near 170; strong amateurs 140-160). Source: openly published "
        "tour launch-monitor averages.",
    ),
    GoalSpec(
        "launch_angle_deg",
        "Launch Angle",
        "°",
        12.0,
        (-20.0, 60.0),
        "Suggested range: 9-16 deg for driver carry optimisation at "
        "tour speeds; higher for slower ball speeds. Source: published "
        "launch-optimisation charts from launch-monitor vendors.",
    ),
    GoalSpec(
        "launch_azimuth_deg",
        "Launch Azimuth",
        "°",
        0.0,
        (-45.0, 45.0),
        "Suggested range: within +/-5 deg of the target line (positive "
        "= right of target); the ball starts close to the face "
        "direction. Source: standard launch-monitor D-plane material.",
    ),
    GoalSpec(
        "spin_rpm",
        "Total Spin",
        " rpm",
        2600.0,
        (0.0, 20000.0),
        "Suggested range: 2,000-3,200 rpm for drivers (tour mean near "
        "2,500), 4,000-7,000 mid irons, up to 10,000+ for wedges. "
        "Source: openly published tour launch-monitor averages.",
    ),
    GoalSpec(
        "spin_axis_deg",
        "Spin-Axis Tilt",
        "°",
        0.0,
        (-90.0, 90.0),
        "Suggested range: within +/-10 deg for reasonably straight "
        "shots (positive = fade/slice side); +/-20 deg is a strong "
        "curve. Source: standard launch-monitor D-plane material.",
    ),
    GoalSpec(
        "carry_m",
        "Carry Distance",
        " m",
        230.0,
        (0.0, 400.0),
        "Suggested range: 200-290 m driver carry at tour ball speeds; "
        "scale with ball speed for other clubs. Source: openly "
        "published tour launch-monitor averages.",
    ),
)

VARIABLE_SPECS: tuple[VariableSpec, ...] = (
    VariableSpec(
        "clubhead_speed_mps",
        "Clubhead Speed",
        " m/s",
        45.0,
        (30.0, 60.0),
        (1.0, 100.0),
        "Suggested range: 36-58 m/s (80-130 mph) driver clubhead speed "
        "(tour average near 50.5 m/s). Source: openly published tour "
        "launch-monitor averages.",
    ),
    VariableSpec(
        "club_path_deg",
        "Club Path",
        "°",
        0.0,
        (-15.0, 15.0),
        (-45.0, 45.0),
        "Suggested range: -8 to +8 deg (positive = in-to-out). Source: "
        "openly published tour launch-monitor averages.",
    ),
    VariableSpec(
        "face_angle_deg",
        "Face Angle",
        "°",
        0.0,
        (-15.0, 15.0),
        (-45.0, 45.0),
        "Suggested range: -5 to +5 deg at impact (positive = open). "
        "Source: standard launch-monitor sign conventions and published "
        "tour data.",
    ),
    VariableSpec(
        "attack_angle_deg",
        "Attack Angle",
        "°",
        0.0,
        (-10.0, 10.0),
        (-30.0, 30.0),
        "Suggested range: -5 to +5 deg for drivers (tour mean near "
        "-1 deg). Source: openly published tour launch-monitor "
        "averages.",
    ),
    VariableSpec(
        "dynamic_loft_deg",
        "Dynamic Loft",
        "°",
        10.5,
        (5.0, 25.0),
        (0.0, 60.0),
        "Suggested range: 8-16 deg delivered loft for drivers; the "
        "10.5 deg default mirrors the desktop default driver. Source: "
        "published launch-monitor fitting references.",
    ),
    VariableSpec(
        "lie_deg",
        "Residual Lie Rotation",
        "°",
        0.0,
        (-10.0, 10.0),
        (-45.0, 45.0),
        "Suggested range: within +/-5 deg of square; 0 keeps the "
        "impact offsets aligned with the face's toe/high axes. Source: "
        "AffineDrift launch-monitor frame conventions.",
    ),
    VariableSpec(
        "impact_offset_toe_mm",
        "Impact Toward Toe",
        " mm",
        0.0,
        (-20.0, 20.0),
        (-40.0, 40.0),
        "Suggested range: within +/-15 mm of face center for "
        "reasonable strikes; gear-effect studies use up to +/-20 mm. "
        "Source: published robot-test impact maps.",
    ),
    VariableSpec(
        "impact_offset_high_mm",
        "Impact Above Center",
        " mm",
        0.0,
        (-15.0, 15.0),
        (-40.0, 40.0),
        "Suggested range: within +/-10 mm of face center vertically. "
        "Source: published robot-test impact maps.",
    ),
    VariableSpec(
        "swing_yaw_deg",
        "Swing-Plane Yaw",
        "°",
        0.0,
        (-30.0, 30.0),
        (-90.0, 90.0),
        "Suggested range: within +/-20 deg; rotates the whole swing "
        "plane about vertical, steering the delivered path. Source: "
        "3-D motion-capture swing-plane studies collected in the "
        "AffineDrift dossier.",
        swing_only=True,
    ),
    VariableSpec(
        "swing_side_tilt_deg",
        "Swing-Plane Side Tilt",
        "°",
        -45.0,
        (-70.0, -20.0),
        (-89.0, 89.0),
        "Suggested range: -30 to -60 deg for driver swings (0 = "
        "vertical cartwheel plane; the tour driver plane leans roughly "
        "45 deg). Source: 3-D motion-capture swing-plane studies "
        "collected in the AffineDrift dossier.",
        swing_only=True,
    ),
    VariableSpec(
        "swing_forward_tilt_deg",
        "Swing-Plane Forward Tilt",
        "°",
        0.0,
        (-30.0, 30.0),
        (-89.0, 89.0),
        "Suggested range: within +/-15 deg; tips the plane toward or "
        "away from the target, trading attack angle against low-point "
        "position. Source: 3-D motion-capture swing-plane studies "
        "collected in the AffineDrift dossier.",
        swing_only=True,
    ),
    VariableSpec(
        "swing_impact_time_offset_s",
        "Impact-Time Offset",
        " s",
        0.0,
        (-0.05, 0.05),
        (-0.5, 0.5),
        "Suggested range: within +/-0.05 s of the peak-clubhead-speed "
        "instant; early impacts arrive on a descending, in-to-out arc, "
        "late ones the reverse. Source: pendulum swing-model timing in "
        "the shared swing_sim package.",
        decimals=4,
        swing_only=True,
    ),
    VariableSpec(
        "swing_damping_shoulder",
        "Shoulder Damping",
        " N·m·s",
        0.4,
        (0.0, 2.0),
        (0.0, 10.0),
        "Suggested range: 0-2 N·m·s viscous damping on the shoulder "
        "joint (0.4 is the shared golf default). Source: double-"
        "pendulum golf-swing literature parameters used by swing_sim.",
        decimals=2,
        swing_only=True,
    ),
    VariableSpec(
        "swing_damping_wrist",
        "Wrist Damping",
        " N·m·s",
        0.25,
        (0.0, 2.0),
        (0.0, 10.0),
        "Suggested range: 0-2 N·m·s viscous damping on the wrist hinge "
        "(0.25 is the shared golf default). Source: double-pendulum "
        "golf-swing literature parameters used by swing_sim.",
        decimals=2,
        swing_only=True,
    ),
)
