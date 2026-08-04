"""Built-in advanced plots, each a PlotSpec factory (epic #4120 V1).

Every builtin renders through the one pipeline in
:mod:`rate_of_closure.plotting.render`. Factories take the reference
run (optional) so range defaults adapt to it — the τ sweep spans the
actual swing duration, the closure sweep brackets the scenario's
about-shaft rate.

Deviation note (documented in SPEC.md): the swing time series plots
clubhead speed and clubhead angular speed rather than pendulum joint
angles θ/ω — SimulationRun stores clubhead poses/twists, not joint
states, and the catalog only describes run data.
"""

from __future__ import annotations

from collections.abc import Callable

from rate_of_closure._contracts import require
from rate_of_closure.plotting.spec import PlotSpec
from rate_of_closure.simulation.session import SimulationRun

__all__ = ["BUILTIN_PLOTS", "builtin_spec"]

BuiltinFactory = Callable[[SimulationRun | None], PlotSpec]


def _closure_sweep(run: SimulationRun | None) -> PlotSpec:
    """The migrated Closure Sweep: path deviation vs about-shaft rate."""
    return PlotSpec(
        kind="sweep",
        x_key="input.omega_shaft_dps",
        y_keys=("metric.path_deviation_deg",),
        title="Impact-Point Path Deviation vs About-Shaft Rotation Rate",
        x_start=0.0,
        x_stop=4000.0,
        x_count=41,
    )


def _tau_range(run: SimulationRun | None) -> tuple[float, float]:
    """A τ window inside the swing (10-90 % of the sampled duration)."""
    duration = float(run.swing_times[-1]) if run is not None else 0.06
    return 0.1 * duration, 0.9 * duration


def _delivery_vs_tau(run: SimulationRun | None) -> PlotSpec:
    start, stop = _tau_range(run)
    return PlotSpec(
        kind="sweep",
        x_key="input.impact_time_s",
        y_keys=(
            "impact.club_path_deg",
            "impact.attack_angle_deg",
            "impact.face_to_path_deg",
        ),
        title="Delivery vs Impact-Time Offset (τ)",
        x_start=start,
        x_stop=stop,
        x_count=21,
    )


def _launch_vs_toe(run: SimulationRun | None) -> PlotSpec:
    return PlotSpec(
        kind="sweep",
        x_key="input.impact_offset_toe_mm",
        y_keys=("launch.ball_speed_mph", "launch.spin_rpm"),
        title="Launch vs Toe Impact Offset",
        x_start=-20.0,
        x_stop=20.0,
        x_count=21,
    )


def _launch_vs_high(run: SimulationRun | None) -> PlotSpec:
    return PlotSpec(
        kind="sweep",
        x_key="input.impact_offset_high_mm",
        y_keys=("launch.ball_speed_mph", "launch.spin_rpm"),
        title="Launch vs Vertical Impact Offset",
        x_start=-10.0,
        x_stop=10.0,
        x_count=21,
    )


def _swing_time_series(run: SimulationRun | None) -> PlotSpec:
    return PlotSpec(
        kind="line",
        x_key="swing.time_s",
        y_keys=("swing.speed_mps", "swing.angular_speed_dps"),
        title="Swing Time Series (Clubhead Speed and Angular Speed)",
    )


def _flight_profile_side(run: SimulationRun | None) -> PlotSpec:
    return PlotSpec(
        kind="line",
        x_key="flight.x_m",
        y_keys=("flight.y_m",),
        title="Flight Profile — Height vs Downrange Distance",
    )


def _flight_profile_top(run: SimulationRun | None) -> PlotSpec:
    return PlotSpec(
        kind="line",
        x_key="flight.x_m",
        y_keys=("flight.z_m",),
        title="Flight Profile — Top-Down Lateral vs Downrange Distance",
    )


#: Builtin name -> (Title Case label, factory), in picker order.
BUILTIN_PLOTS: dict[str, tuple[str, BuiltinFactory]] = {
    "closure_sweep": ("Closure Sweep", _closure_sweep),
    "delivery_vs_tau": ("Delivery vs τ Sweep", _delivery_vs_tau),
    "launch_vs_toe_offset": ("Launch vs Toe Offset", _launch_vs_toe),
    "launch_vs_high_offset": ("Launch vs High Offset", _launch_vs_high),
    "swing_time_series": ("Swing Time Series", _swing_time_series),
    "flight_profile_side": ("Flight Profile (Side)", _flight_profile_side),
    "flight_profile_top": ("Flight Profile (Top-Down)", _flight_profile_top),
}


def builtin_spec(name: str, run: SimulationRun | None = None) -> PlotSpec:
    """The PlotSpec of one builtin, adapted to the reference run.

    Args:
        name: A :data:`BUILTIN_PLOTS` key.
        run: Optional reference run used for adaptive ranges.

    Returns:
        The validated plot definition.
    """
    require(name in BUILTIN_PLOTS, f"unknown builtin plot {name!r}", name)
    _label, factory = BUILTIN_PLOTS[name]
    return factory(run)
