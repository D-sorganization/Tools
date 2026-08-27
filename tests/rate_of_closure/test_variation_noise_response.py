"""Integration contracts for the governed R12.3 noise-response field."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from rate_of_closure.variation.simulation_adapter import spatial_source_layouts
from shared.python.swing_sim.variation.noise_response import (
    ADEQUACY_ESTIMABLE,
    compute_position_noise_response_field,
)
from shared.python.swing_sim.variation.noise_response_plot import (
    iter_position_noise_response_plot_rows,
)
from shared.python.swing_sim.variation.tests.noise_response_test_support import (
    build_response_inputs,
    default_fixture_config,
)

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "docs/audits/rate_of_closure_r12_3_noise_response_capabilities.v1.json"
R11_3_AUDIT = (
    ROOT / "docs/audits/rate_of_closure_r11_3_trace_resampling_capabilities.v1.json"
)
GUIDE = ROOT / "docs/specs/GEOMETRIC_NOISE_RESPONSE_FIELD.md"


def _load(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def test_plot_rows_expose_paired_metrics_denominators_and_limits() -> None:
    field = compute_position_noise_response_field(
        build_response_inputs(default_fixture_config())
    )

    rows = tuple(iter_position_noise_response_plot_rows(field))

    assert len(rows) == 8
    first = rows[0]
    assert first.input_id == "input.ball-speed"
    assert first.input_unit == "mph"
    assert first.point_id == "swing.wrist"
    assert first.coordinate_frame == "swing.world"
    assert first.time_s == 0.0
    assert first.adequacy == ADEQUACY_ESTIMABLE
    assert first.availability_count == 4
    assert first.all_eligible_count == 4
    assert len(first.signed_response) == 3
    assert np.isfinite(first.response_magnitude)
    assert np.isfinite(first.matched_absolute_rms_scatter_m)
    assert np.isfinite(first.all_eligible_absolute_rms_scatter_m)
    assert first.method_id == "paired-oat-linear-through-origin/v1"
    assert first.normalization_id == "declared-distribution-standard-deviation/v1"
    assert "not causal anatomy" in first.scientific_boundary
    assert len(first.field_sha256) == 64


def test_capability_manifest_is_exhaustive_and_inherits_resampling_policy() -> None:
    audit = _load(AUDIT)
    resampling = _load(R11_3_AUDIT)
    layouts = spatial_source_layouts()

    assert audit["schema_version"] == "tools-r12.3-noise-response-capabilities/v1"
    assert audit["requirement_id"] == "R12.3"
    assert audit["qualified_base_revision"] == (
        "4ddec9175814451fdc3d1a94b45f1190e7503bca"
    )
    assert audit["implementation_issue"] == 4765
    assert audit["resampling_policy_id"] == resampling["policy_id"]
    assert tuple(audit["source_layouts"]) == tuple(layouts)
    assert audit["adapter_cells"] == resampling["adapter_cells"]
    assert audit["adapter_status_counts"] == {
        "verified": 2,
        "explicitly_unavailable": 10,
    }
    assert audit["field_schema"] == {
        "schema_id": "swing-sim/position-noise-response-field",
        "schema_version": 1,
    }
    assert audit["estimator"] == {
        "method_id": "paired-oat-linear-through-origin/v1",
        "normalization_id": "declared-distribution-standard-deviation/v1",
    }
    assert {item["status"] for item in audit["input_designs"]} == {
        "verified",
        "explicitly_unsupported",
    }
    assert set(resampling["adverse_cases"]) <= set(audit["adverse_cases"])
    assert "not causal anatomy" in audit["scientific_boundary"]


def test_scientific_guide_separates_estimands_and_publishes_exact_gates() -> None:
    guide = GUIDE.read_text(encoding="utf-8")

    for proposition in (
        "Quiet geometry",
        "noise responsiveness",
        "causal control",
        "joint work",
        "outcome robustness",
        "not substitutes for governed participant data",
    ):
        assert proposition in guide
    for command in (
        "python -m pytest src/shared/python/swing_sim/variation/tests "
        '-k "response or dispersion or resampling" -n 0 -q',
        "python -m pytest tests/rate_of_closure "
        '-k "response or dispersion or variation_geometry" -n 0 -q',
        "python -m scripts.check_design_manual_governance",
        "python -m scripts.build_tools_module_inventory --check",
    ):
        assert command in guide
