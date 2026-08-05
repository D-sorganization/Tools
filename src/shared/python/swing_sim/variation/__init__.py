"""Shared variation / Monte-Carlo engine (epic #4120, phase V3).

One 'how parameters vary' vocabulary for the whole repo: a namespaced
variable registry (:mod:`.spec`), a seeded parallel N-run executor
(:mod:`.engine`), dispersion/sensitivity analysis (:mod:`.analysis`),
and CSV/JSON dataset IO (:mod:`.dataset_io`).

Deliberately NOT re-exported from ``shared.python.swing_sim`` — import
from this subpackage directly (same policy as ``swing_sim.solver``).
"""

from __future__ import annotations

from .analysis import (
    DispersionEllipse,
    OutputStats,
    SensitivityResult,
    dispersion_ellipse,
    one_at_a_time_sensitivity,
    spearman_matrix,
    summary_stats,
)
from .engine import (
    DELIVERY_OUTPUTS,
    FLIGHT_OUTPUTS,
    LAUNCH_OUTPUTS,
    CancelledError,
    ProgressReport,
    VariationDataset,
    evaluate_run,
    outputs_for_mode,
    run_variation,
    sample_inputs,
)
from .ensemble_geometry import (
    EnsemblePositionTraces,
    LowVariabilityCriteria,
    LowVariabilityInterval,
    PositionDispersion,
    compute_position_dispersion,
    find_low_variability_intervals,
)
from .spec import (
    CATEGORY_CLUB,
    CATEGORY_DELIVERY,
    CATEGORY_LAUNCH,
    CATEGORY_SWING,
    DISTRIBUTIONS,
    MODE_CATEGORIES,
    MODES,
    SWING_DERIVED_KEYS,
    NoiseSpec,
    VariableDef,
    VariationPlan,
    keys_for_mode,
    register_variable,
    variable_registry,
    variables_in_category,
)

__all__ = [
    "CATEGORY_CLUB",
    "CATEGORY_DELIVERY",
    "CATEGORY_LAUNCH",
    "CATEGORY_SWING",
    "DELIVERY_OUTPUTS",
    "DISTRIBUTIONS",
    "FLIGHT_OUTPUTS",
    "LAUNCH_OUTPUTS",
    "MODES",
    "MODE_CATEGORIES",
    "SWING_DERIVED_KEYS",
    "CancelledError",
    "DispersionEllipse",
    "EnsemblePositionTraces",
    "LowVariabilityCriteria",
    "LowVariabilityInterval",
    "NoiseSpec",
    "OutputStats",
    "ProgressReport",
    "PositionDispersion",
    "SensitivityResult",
    "VariableDef",
    "VariationDataset",
    "VariationPlan",
    "dispersion_ellipse",
    "compute_position_dispersion",
    "evaluate_run",
    "keys_for_mode",
    "find_low_variability_intervals",
    "one_at_a_time_sensitivity",
    "outputs_for_mode",
    "register_variable",
    "run_variation",
    "sample_inputs",
    "spearman_matrix",
    "summary_stats",
    "variable_registry",
    "variables_in_category",
]
