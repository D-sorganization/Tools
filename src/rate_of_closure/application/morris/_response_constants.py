"""Private exact field and vocabulary constants for Morris responses."""

import math

CAPABILITY_SCHEMA_ID = "rate-of-closure/morris-authority-capability"
REPORT_SCHEMA_ID = "swing-sim/morris-global-sensitivity-report"
API_PREFIX = "/api/rate-of-closure/v1"
CAPABILITY_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "available",
        "api_prefix",
        "request_schema_id",
        "job_schema_id",
    }
)
JOB_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "job_id",
        "request_id",
        "status",
        "completed_samples",
        "total_samples",
        "cancel_requested",
        "report",
        "error",
    }
)
REPORT_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "method",
        "design",
        "assumptions",
        "interaction_caveat",
        "estimates",
    }
)
DESIGN_FIELDS = frozenset(
    {"trajectories", "levels", "seed", "total_samples", "normalized_step"}
)
ESTIMATE_FIELDS = frozenset(
    {"source", "target", "effects", "availability", "sample_adequacy", "denominator"}
)
SOURCE_FIELDS = frozenset(
    {"spec_id", "variable_key", "unit", "bounds", "time_window_s", "point_ids"}
)
TARGET_FIELDS = frozenset(
    {"name", "unit", "kind", "time_s", "point_id", "coordinate_frame"}
)
EFFECT_FIELDS = frozenset({"mu", "mu_star", "mu_star_standard_error", "sigma"})
DENOMINATOR_FIELDS = frozenset(
    {
        "total_pairs",
        "valid_pairs",
        "typed_no_impact_pairs",
        "no_impact_unavailable_pairs",
        "failed_pairs",
        "nonfinite_pairs",
    }
)
AVAILABILITIES = frozenset({"available", "constant-output", "insufficient-data"})
ADEQUACIES = frozenset({"adequate", "limited", "insufficient"})
TARGET_KINDS = frozenset({"scalar", "state-point", "impact", "shot-outcome"})
PRODUCER_CLAMP_MULTIPLIER = 64.0
IDENTITY_EPSILON_MULTIPLIER = 32.0
MAX_SAFELY_SQUARED_METRIC = math.sqrt(float.fromhex("0x1.fffffffffffffp+1023"))
