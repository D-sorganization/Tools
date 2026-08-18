"""Shared builders for variation plot-definition contract tests."""

from rate_of_closure.variation.plot_definition import PlotDefinition


def complete_geometric_definition(**overrides: object) -> PlotDefinition:
    """Build one complete, valid geometric plot definition."""
    values: dict[str, object] = {
        "result_id": "ensemble-contract",
        "plot_type": "swing_arc_overlay",
        "coordinate_frame": "app_frame:x_target,y_up,z_right",
        "point_id": "swing.clubhead.reference",
        "position_unit": "m",
        "alignment_basis": "common_simulation_time_s",
        "dispersion_metric": "rms-radius",
        "dispersion_unit": "m",
        "quiet_threshold": 0.005,
        "confidence_level": None,
        "min_quiet_duration_s": 0.0,
        "min_quiet_samples": 1,
        "selected_trial_index": 0,
        "camera_yaw_deg": -37.0,
        "camera_pitch_deg": 22.0,
        "camera_zoom": 1.2,
        "outcome_filter": "evaluated_hit",
        "phase_end_fraction": 0.75,
        "perturbation_source_key": "swing_sim.swing.yaw_deg",
        "perturbation_band": "Upper Third",
        "show_confidence_ellipsoids": False,
    }
    values.update(overrides)
    return PlotDefinition(**values)  # type: ignore[arg-type]


def scatter_definition(**overrides: object) -> PlotDefinition:
    """Build one complete, valid scalar-scatter definition."""
    values: dict[str, object] = {
        "result_id": "scatter-contract",
        "plot_type": "scalar_scatter",
        "x_variable_key": "input:swing.speed",
        "y_variable_key": "output:carry_m",
        "selected_trial_index": 1,
    }
    values.update(overrides)
    return PlotDefinition(**values)  # type: ignore[arg-type]


def matrix_definition(**overrides: object) -> PlotDefinition:
    """Build one complete, valid distribution-matrix definition."""
    values: dict[str, object] = {
        "result_id": "matrix-contract",
        "plot_type": "distribution_matrix",
        "variable_keys": ("input:swing.speed", "output:carry_m"),
    }
    values.update(overrides)
    return PlotDefinition(**values)  # type: ignore[arg-type]


__all__ = [
    "complete_geometric_definition",
    "matrix_definition",
    "scatter_definition",
]
