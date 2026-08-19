"""Schema identifiers and field sets for variation execution metadata.

Split out of ``execution_metadata`` to keep that module under the 500-LOC
per-file budget (``scripts/check_file_size_budget.py``). These are declarations
only -- no behaviour -- so the split is a pure move; ``execution_metadata``
re-exports every name, leaving existing importers unaffected.
"""

from __future__ import annotations

import re

EXECUTION_DOCUMENT_SCHEMA_ID = "rate-of-closure/variation-execution-document"
EXECUTION_DOCUMENT_SCHEMA_VERSION = 2
EXECUTION_METADATA_SCHEMA_ID = "rate-of-closure/variation-execution-metadata"
EXECUTION_METADATA_SCHEMA_VERSION = 2
VARIABLE_REGISTRY_SCHEMA_ID = "swing-sim/variation-variable-registry"
VARIABLE_REGISTRY_SCHEMA_VERSION = 1
LEGACY_CURRENT_REGISTRY_WARNING = (
    "Legacy plan has no historical execution sidecar; resolved against the "
    "current variable registry. This is not evidence of historical reproducibility."
)
LEGACY_EXECUTION_DOCUMENT_MIGRATION_ERROR = (
    "Execution document schema @1 lacks RNG and solver identity; load its raw "
    "plan and resolve a fresh @2 sidecar. Historical replay remains unproven."
)

_DOCUMENT_FIELDS = frozenset({"schema_id", "schema_version", "plan", "metadata"})
_METADATA_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "plan_sha256",
        "mode",
        "flight_model",
        "registry_schema_id",
        "registry_schema_version",
        "registry_sha256",
        "resolved_variables",
        "rng_identity",
        "implementation_identity",
    }
)
_VARIABLE_FIELDS = frozenset({"variable_key", "value", "unit", "dimension"})
_RNG_FIELDS = frozenset(
    {
        "algorithm_id",
        "algorithm_version",
        "stream_derivation_id",
        "stream_derivation_version",
    }
)
_IMPLEMENTATION_FIELDS = frozenset(
    {
        "runtime_id",
        "runtime_version",
        "executor_id",
        "executor_version",
        "solver_id",
        "solver_version",
    }
)
_PLAN_FIELDS = frozenset(
    {
        "schema_version",
        "mode",
        "base_variables",
        "noise",
        "n_runs",
        "seed",
        "flight_model",
        "groups",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
