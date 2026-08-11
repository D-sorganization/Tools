"""Goal and variable-partition types for the impact-parameter solver.

Epic #4103, issue #4109. Scaffolding modeled on UpstreamDrift's
``movement_optimizer`` (typed goal/constraint dataclasses feeding a pure
objective; see ``trajectory/optimizer_constraints.py``), with golf-impact
semantics: goals target launch-monitor quantities, and the optimizer
controls a user-chosen partition of delivery / swing variables.

Goal quantities (all optional, each with an optional weight)
------------------------------------------------------------
============================ ======= =====================================
name                         unit    sign convention
============================ ======= =====================================
``club_path_deg``            deg     + = in-to-out
``face_angle_deg``           deg     + = open (right of target, RH player)
``attack_angle_deg``         deg     + = hitting up
``dynamic_loft_deg``         deg     + = lofted upward
``ball_speed_mph``           mph     >= 0
``launch_angle_deg``         deg     + = above horizontal
``launch_azimuth_deg``       deg     + = right of the target line
``spin_rpm``                 RPM     total spin magnitude, >= 0
``spin_axis_deg``            deg     + = fade/slice side (tilt vs backspin)
``carry_m``                  m       carry distance, >= 0
============================ ======= =====================================

Variables (delivery mode)
-------------------------
The delivery front-end parameters of
:class:`shared.python.swing_sim.impact.delivery.DeliveryParameters`:
``clubhead_speed_mps``, ``club_path_deg``, ``face_angle_deg``,
``attack_angle_deg``, ``dynamic_loft_deg``, ``lie_deg``, plus the impact
offsets ``impact_offset_toe_mm`` / ``impact_offset_high_mm``.

Variables (swing-source mode, ``use_swing_source=True``)
--------------------------------------------------------
The clubhead delivery (speed / path / attack angle) is *derived* from a
:class:`shared.python.swing_sim.swing_source.DoublePendulumSwing` sampled
near peak clubhead speed, so those three names are rejected; instead the
pendulum variables become available: the three sequential swing-plane
tilts ``swing_yaw_deg`` / ``swing_side_tilt_deg`` /
``swing_forward_tilt_deg``, the impact-time offset
``swing_impact_time_offset_s`` (relative to the peak-speed instant), and
the pendulum damping parameters ``swing_damping_shoulder`` /
``swing_damping_wrist``. Face orientation and impact offsets remain
delivery variables in both modes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType

import numpy as np

from shared.python.contracts import require

from .targets import TargetRegion

GOAL_QUANTITIES: tuple[str, ...] = (
    "club_path_deg",
    "face_angle_deg",
    "attack_angle_deg",
    "dynamic_loft_deg",
    "ball_speed_mph",
    "launch_angle_deg",
    "launch_azimuth_deg",
    "spin_rpm",
    "spin_axis_deg",
    "carry_m",
)
"""Canonical, ordered goal-quantity names (defines residual ordering)."""

DELIVERY_VARIABLE_DEFAULTS: Mapping[str, float] = MappingProxyType(
    {
        "clubhead_speed_mps": 45.0,
        "club_path_deg": 0.0,
        "face_angle_deg": 0.0,
        "attack_angle_deg": 0.0,
        "dynamic_loft_deg": 10.5,
        "lie_deg": 0.0,
        "impact_offset_toe_mm": 0.0,
        "impact_offset_high_mm": 0.0,
    }
)
"""Delivery variables and their defaults when neither free nor fixed."""

SWING_VARIABLE_DEFAULTS: Mapping[str, float] = MappingProxyType(
    {
        "swing_yaw_deg": 0.0,
        "swing_side_tilt_deg": 0.0,
        "swing_forward_tilt_deg": 0.0,
        "swing_impact_time_offset_s": 0.0,
        "swing_damping_shoulder": 0.4,
        "swing_damping_wrist": 0.25,
    }
)
"""Swing-source variables (pendulum + plane tilts) and their defaults."""

SWING_DERIVED_VARIABLES: tuple[str, ...] = (
    "clubhead_speed_mps",
    "club_path_deg",
    "attack_angle_deg",
)
"""Delivery variables derived from the swing sample in swing-source mode."""


@dataclass(frozen=True)
class GoalTerm:
    """One goal target with a positive weight (dimensionless multiplier)."""

    target: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        require(math.isfinite(self.target), "target must be finite", self.target)
        require(
            math.isfinite(self.weight) and self.weight > 0.0,
            "weight must be finite and > 0",
            self.weight,
        )


def _as_term(value: GoalTerm | float | tuple[float, float]) -> GoalTerm:
    """Coerce ``float`` / ``(target, weight)`` / :class:`GoalTerm` inputs."""
    if isinstance(value, GoalTerm):
        return value
    if isinstance(value, tuple):
        target, weight = value
        return GoalTerm(float(target), float(weight))
    return GoalTerm(float(value))


@dataclass(frozen=True)
class ImpactGoal:
    """Targets over any subset of the goal quantities, each optionally weighted.

    Build directly with :class:`GoalTerm` fields, or via :meth:`of` which
    also accepts plain floats and ``(target, weight)`` tuples::

        goal = ImpactGoal.of(ball_speed_mph=163.0, spin_rpm=(2600.0, 0.5))

    Invariant: at least one quantity is targeted.
    """

    club_path_deg: GoalTerm | None = None
    face_angle_deg: GoalTerm | None = None
    attack_angle_deg: GoalTerm | None = None
    dynamic_loft_deg: GoalTerm | None = None
    ball_speed_mph: GoalTerm | None = None
    launch_angle_deg: GoalTerm | None = None
    launch_azimuth_deg: GoalTerm | None = None
    spin_rpm: GoalTerm | None = None
    spin_axis_deg: GoalTerm | None = None
    carry_m: GoalTerm | None = None
    #: Landing target region (#4125 H7b) — additive: any quantity goals
    #: still apply; the region contributes one extra residual (distance
    #: outside the region, 0 inside, plus a small centering term).
    target_region: TargetRegion | None = None
    target_region_weight: float = 1.0

    def __post_init__(self) -> None:
        require(
            self.target_region is not None
            or any(getattr(self, name) is not None for name in GOAL_QUANTITIES),
            "ImpactGoal must target at least one quantity or a region",
            None,
        )
        for spec in fields(self):
            if spec.name in ("target_region", "target_region_weight"):
                continue
            value = getattr(self, spec.name)
            require(
                value is None or isinstance(value, GoalTerm),
                f"{spec.name} must be a GoalTerm or None (use ImpactGoal.of)",
                value,
            )
        require(
            self.target_region is None or isinstance(self.target_region, TargetRegion),
            "target_region must be a TargetRegion or None",
            self.target_region,
        )
        require(
            math.isfinite(self.target_region_weight)
            and self.target_region_weight > 0.0,
            "target_region_weight must be finite and > 0",
            self.target_region_weight,
        )

    @classmethod
    def of(
        cls,
        target_region: TargetRegion | None = None,
        target_region_weight: float = 1.0,
        **targets: GoalTerm | float | tuple[float, float],
    ) -> ImpactGoal:
        """Build a goal from floats, ``(target, weight)`` tuples, or terms."""
        return cls.from_mapping(
            targets,
            target_region=target_region,
            target_region_weight=target_region_weight,
        )

    @classmethod
    def from_mapping(
        cls,
        targets: Mapping[str, GoalTerm | float | tuple[float, float]],
        *,
        target_region: TargetRegion | None = None,
        target_region_weight: float = 1.0,
    ) -> ImpactGoal:
        """Build a goal from a dynamic mapping without unsafe ``**dict`` typing."""
        unknown = set(targets) - set(GOAL_QUANTITIES)
        require(not unknown, "unknown goal quantities", sorted(unknown))
        normalized = {name: _as_term(value) for name, value in targets.items()}
        return cls(
            club_path_deg=normalized.get("club_path_deg"),
            face_angle_deg=normalized.get("face_angle_deg"),
            attack_angle_deg=normalized.get("attack_angle_deg"),
            dynamic_loft_deg=normalized.get("dynamic_loft_deg"),
            ball_speed_mph=normalized.get("ball_speed_mph"),
            launch_angle_deg=normalized.get("launch_angle_deg"),
            launch_azimuth_deg=normalized.get("launch_azimuth_deg"),
            spin_rpm=normalized.get("spin_rpm"),
            spin_axis_deg=normalized.get("spin_axis_deg"),
            carry_m=normalized.get("carry_m"),
            target_region=target_region,
            target_region_weight=target_region_weight,
        )

    def items(self) -> tuple[tuple[str, GoalTerm], ...]:
        """Targeted ``(quantity, term)`` pairs in canonical order."""
        return tuple(
            (name, term)
            for name in GOAL_QUANTITIES
            if (term := getattr(self, name)) is not None
        )

    @property
    def needs_flight(self) -> bool:
        """True when a ball-flight simulation is required."""
        return self.carry_m is not None or self.target_region is not None

    @property
    def needs_launch(self) -> bool:
        """True when the impact -> launch derivation is required."""
        return self.target_region is not None or any(
            getattr(self, name) is not None
            for name in (
                "ball_speed_mph",
                "launch_angle_deg",
                "launch_azimuth_deg",
                "spin_rpm",
                "spin_axis_deg",
                "carry_m",
            )
        )


@dataclass(frozen=True)
class VariablePartition:
    """Which variables the optimizer controls (with bounds) vs user-fixed.

    Attributes:
        free: Mapping of variable name -> ``(lower, upper)`` bounds. These
            are the optimizer's decision variables.
        fixed: Mapping of variable name -> value held constant.
        use_swing_source: When True, candidate evaluation runs a
            double-pendulum swing source and derives clubhead speed /
            path / attack angle from it (see module docstring); the
            ``swing_*`` variables become available and the derived
            delivery names are rejected.

    Variables in neither mapping take the documented defaults.

    Invariants (DbC, validated at construction):
        - every name is a known variable for the selected mode;
        - ``free`` and ``fixed`` are disjoint;
        - bounds are finite with ``lower < upper``; fixed values finite;
        - in swing-source mode no derived delivery variable appears; in
          delivery mode no ``swing_*`` variable appears.
    """

    free: Mapping[str, tuple[float, float]]
    fixed: Mapping[str, float] = field(default_factory=dict)
    use_swing_source: bool = False

    def __post_init__(self) -> None:
        known = set(DELIVERY_VARIABLE_DEFAULTS)
        if self.use_swing_source:
            known = (known - set(SWING_DERIVED_VARIABLES)) | set(
                SWING_VARIABLE_DEFAULTS
            )
        free = dict(self.free)
        fixed = dict(self.fixed)
        unknown = (set(free) | set(fixed)) - known
        require(
            not unknown,
            "unknown variables for this mode (swing-derived delivery names "
            "are rejected when use_swing_source=True, swing_* names when "
            "False)",
            sorted(unknown),
        )
        overlap = set(free) & set(fixed)
        require(
            not overlap,
            "variables cannot be both free and fixed",
            sorted(overlap),
        )
        for name, bounds in free.items():
            lo, hi = float(bounds[0]), float(bounds[1])
            require(
                math.isfinite(lo) and math.isfinite(hi) and lo < hi,
                f"bounds for {name!r} must be finite with lower < upper",
                (lo, hi),
            )
            free[name] = (lo, hi)
        for name, value in fixed.items():
            require(
                math.isfinite(float(value)),
                f"fixed value for {name!r} must be finite",
                value,
            )
            fixed[name] = float(value)
        object.__setattr__(self, "free", MappingProxyType(free))
        object.__setattr__(self, "fixed", MappingProxyType(fixed))

    @property
    def free_names(self) -> tuple[str, ...]:
        """Free-variable names in insertion order (defines vector layout)."""
        return tuple(self.free)

    def bounds_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Lower/upper bound vectors aligned with :attr:`free_names`."""
        require(len(self.free) > 0, "partition has no free variables", None)
        lo = np.array([self.free[name][0] for name in self.free_names])
        hi = np.array([self.free[name][1] for name in self.free_names])
        return lo, hi

    def assemble(self, x: np.ndarray) -> dict[str, float]:
        """Full variable dict from a free-variable vector (defaults filled).

        Preconditions: ``x`` is finite with one entry per free variable.
        """
        arr = np.asarray(x, dtype=float)
        require(
            arr.shape == (len(self.free),),
            "x must have one entry per free variable",
            arr.shape,
        )
        require(bool(np.all(np.isfinite(arr))), "x must be finite", arr)
        variables: dict[str, float] = dict(DELIVERY_VARIABLE_DEFAULTS)
        if self.use_swing_source:
            for name in SWING_DERIVED_VARIABLES:
                variables.pop(name)
            variables.update(SWING_VARIABLE_DEFAULTS)
        variables.update(self.fixed)
        variables.update(zip(self.free_names, arr.tolist(), strict=True))
        return variables


__all__ = [
    "DELIVERY_VARIABLE_DEFAULTS",
    "GOAL_QUANTITIES",
    "SWING_DERIVED_VARIABLES",
    "SWING_VARIABLE_DEFAULTS",
    "GoalTerm",
    "ImpactGoal",
    "TargetRegion",
    "VariablePartition",
]
