"""Contracts for the requirement-level evidence authority for Tools #4142."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from rate_of_closure.variation.simulation_adapter import spatial_source_layouts

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/audits/rate_of_closure_epic_4142_evidence.v1.json"
R10_4_REQUALIFICATION = (
    ROOT / "docs/audits/rate_of_closure_r10_4_requalification.v1.json"
)
R10_3_CAPABILITY_AUDIT = (
    ROOT / "docs/audits/rate_of_closure_r10_3_execution_capabilities.v1.json"
)
R11_1_CAPABILITY_AUDIT = (
    ROOT / "docs/audits/rate_of_closure_r11_1_complete_trial_capabilities.v1.json"
)
R11_3_CAPABILITY_AUDIT = (
    ROOT / "docs/audits/rate_of_closure_r11_3_trace_resampling_capabilities.v1.json"
)
PUBLIC_GUIDE = ROOT / "docs/rate_of_closure/variation_ensemble_reproducibility_guide.md"
EXPECTED_REQUIREMENTS = tuple(
    [f"R10.{index}" for index in range(1, 7)]
    + [f"R11.{index}" for index in range(1, 6)]
    + [f"R12.{index}" for index in range(1, 6)]
    + [f"R13.{index}" for index in range(1, 6)]
    + [f"R14.{index}" for index in range(1, 7)]
    + [f"R15.{index}" for index in range(1, 5)]
)
ALLOWED_STATUSES = {"verified", "partial", "unverified", "external_blocked"}
NONAUTHORITATIVE_PREFIXES = (
    "AGENT_HANDOFF.md",
    "SPEC.md",
    "docs/agent_handoff_archive/",
)
UPSTREAM_VARIATION_PR = "https://github.com/D-sorganization/UpstreamDrift/pull/9039"
PINNED_TOOLS_REVISION = (
    "17474249b9267d0e73a779c1d72f231e7b8de39c"  # pragma: allowlist secret
)


def _load() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(EVIDENCE.read_text(encoding="utf-8")))


def test_epic_4142_evidence_covers_every_requirement_exactly_once() -> None:
    evidence = _load()

    assert evidence["schema_version"] == "tools-epic-requirement-evidence/v1"
    assert evidence["repository"] == "D-sorganization/Tools"
    assert evidence["epic"] == 4142
    assert evidence["audit_base_revision"] == (
        "eebdddf8c6e366722be40c25278cf34a0392f256"  # pragma: allowlist secret
    )
    assert tuple(item["requirement_id"] for item in evidence["requirements"]) == (
        EXPECTED_REQUIREMENTS
    )
    assert len({item["requirement_id"] for item in evidence["requirements"]}) == len(
        EXPECTED_REQUIREMENTS
    )


def test_epic_4142_evidence_is_fail_closed_and_locally_traceable() -> None:
    evidence = _load()
    observed_counts = {status: 0 for status in ALLOWED_STATUSES}

    for item in evidence["requirements"]:
        status = item["status"]
        observed_counts[status] += 1
        assert status in ALLOWED_STATUSES
        assert item["requirement"].strip()
        assert item["rationale"].strip()
        assert item["validation_commands"]
        assert all(command.strip() for command in item["validation_commands"])
        assert item["evidence_files"]
        assert all("*" not in path for path in item["evidence_files"])
        for relative in item["evidence_files"]:
            assert (ROOT / relative).is_file(), relative

        if status == "verified":
            assert item["gaps"] == []
            assert any(
                path.startswith(("src/", "tests/"))
                and not path.startswith(NONAUTHORITATIVE_PREFIXES)
                for path in item["evidence_files"]
            )
            assert any(path.startswith("tests/") for path in item["evidence_files"])
        else:
            assert item["gaps"]
            assert all(gap.strip() for gap in item["gaps"])

    assert evidence["status_counts"] == observed_counts
    assert evidence["epic_closeable"] is False
    assert any(
        count for status, count in observed_counts.items() if status != "verified"
    )


def test_epic_4142_remote_evidence_is_immutable_and_reviewable() -> None:
    evidence = _load()

    for item in evidence["requirements"]:
        for remote in item["remote_evidence"]:
            assert remote.startswith("https://github.com/D-sorganization/")
            assert (
                "/actions/runs/" in remote or "/pull/" in remote or "/issues/" in remote
            )
            assert "/main/" not in remote


def test_r10_4_is_verified_by_revision_bound_current_main_requalification() -> None:
    """R10.4 requires current behavior and adjudicated historical failures."""
    evidence = _load()
    requirements = {item["requirement_id"]: item for item in evidence["requirements"]}
    r10_4 = requirements["R10.4"]
    audit = cast(
        dict[str, Any],
        json.loads(R10_4_REQUALIFICATION.read_text(encoding="utf-8")),
    )

    assert evidence["status_counts"] == {
        "verified": 26,
        "partial": 5,
        "unverified": 0,
        "external_blocked": 0,
    }
    assert r10_4["status"] == "verified"
    assert r10_4["gaps"] == []
    assert (
        str(R10_4_REQUALIFICATION.relative_to(ROOT)).replace("\\", "/")
        in (r10_4["evidence_files"])
    )

    assert audit["schema_version"] == "tools-r10.4-requalification/v1"
    assert audit["requirement_id"] == "R10.4"
    assert audit["qualified_base_revision"] == (
        "cff2909f1585273e10fa49165bfab8521e889da1"  # pragma: allowlist secret
    )
    assert audit["implementation_pull_request"] == 4669
    assert audit["implementation_head_revision"] == (
        "36f4b1add2bc72cea87bd9f87d36b232db76d50b"  # pragma: allowlist secret
    )
    assert audit["implementation_merge_revision"] == (
        "f9730033fd279ba8b4abe03bab2aadd950400b47"  # pragma: allowlist secret
    )
    assert audit["current_main_results"] == {
        "python_tests_passed": 138,
        "web_tests_passed": 270,
        "upstream_provider_contracts_passed": 3,
    }
    assert audit["historical_failures"] == [
        {
            "job": "ground-tee-playwright",
            "classification": "runner_provisioning_contention",
            "behavioral_test_execution": False,
        },
        {
            "job": "upstream-downstream-consumer-contract",
            "classification": "superseded_consumer_isolation_defect",
            "behavioral_test_execution": True,
        },
    ]
    assert audit["scientific_boundary"] == (
        "Repository verification of execution-document provenance and "
        "persistence does not validate a human swing mechanism, identify a "
        "participant, or support universal coaching advice."
    )


def test_r10_3_is_verified_by_exhaustive_cross_runtime_capabilities() -> None:
    """R10.3 requires declared semantics for every registered input locus."""
    evidence = _load()
    requirements = {item["requirement_id"]: item for item in evidence["requirements"]}
    r10_3 = requirements["R10.3"]
    audit = cast(
        dict[str, Any],
        json.loads(R10_3_CAPABILITY_AUDIT.read_text(encoding="utf-8")),
    )

    assert evidence["status_counts"] == {
        "verified": 26,
        "partial": 5,
        "unverified": 0,
        "external_blocked": 0,
    }
    assert r10_3["status"] == "verified"
    assert r10_3["gaps"] == []
    assert (
        str(R10_3_CAPABILITY_AUDIT.relative_to(ROOT)).replace("\\", "/")
        in r10_3["evidence_files"]
    )
    assert audit == {
        "schema_version": "tools-r10.3-execution-capabilities/v1",
        "requirement_id": "R10.3",
        "qualified_base_revision": (
            "9fe87f0eec9f341fdfc50fc2a116c601b94781d5"  # pragma: allowlist secret
        ),
        "implementation_issue": 4756,
        "authority": "src/rate_of_closure/locus_execution_capabilities.v1.json",
        "standalone_web_mirror": (
            "src/rate_of_closure/web/src/vendored/locus_execution_capabilities.v1.json"
        ),
        "mirror_drift_gates": [
            "src/rate_of_closure/web/src/vendored/vendoredSync.test.ts",
            "tests/rate_of_closure/test_web_vendored_sync.py",
        ],
        "registered_variable_count": 31,
        "supported_by_adapter": {
            "global_simulation_value/v1": 11,
            "localized_joint_torque_offset/v1": 2,
            "regional_ground_value/v1": 2,
            "turf_profile_value/v1": 4,
        },
        "explicitly_unsupported_count": 12,
        "time_window_semantics": "half_open_seconds",
        "point_id_semantics": ("topological_control_loci_not_spatial_trace_points"),
        "negative_gates": [
            "registry_coverage_drift",
            "duplicate_variable_key",
            "malformed_capability_record",
            "forbidden_locus_metadata",
            "missing_or_inexact_topological_locus",
            "out_of_run_time_window",
        ],
        "scientific_boundary": (
            "Execution-locus coverage proves deterministic adapter semantics; "
            "it does not identify anatomical force sources, validate a human "
            "swing mechanism, or support universal coaching advice."
        ),
    }


def test_r11_1_capability_matrix_is_exhaustive_and_fail_closed() -> None:
    """Every current source/adapter cell must be verified or unavailable."""
    requirements = {item["requirement_id"]: item for item in _load()["requirements"]}
    r11_1 = requirements["R11.1"]
    audit = cast(
        dict[str, Any], json.loads(R11_1_CAPABILITY_AUDIT.read_text(encoding="utf-8"))
    )
    sources = tuple(audit["source_kinds"])
    adapters = tuple(audit["adapter_ids"])
    cells = audit["cells"]

    assert audit["schema_version"] == "tools-r11.1-complete-trial-capabilities/v1"
    assert audit["requirement_id"] == "R11.1"
    assert audit["qualified_base_revision"] == (
        "55805fe4de1b0afc3710efce4ed516d59e685717"  # pragma: allowlist secret
    )
    assert audit["implementation_issue"] == 4758
    assert audit["record_schema"] == "rate-complete-trial/v1"
    assert audit["durable_schema_version"] == 3
    assert r11_1["status"] == "verified"
    assert r11_1["gaps"] == []
    assert (
        "https://github.com/D-sorganization/Tools/pull/4762" in r11_1["remote_evidence"]
    )
    assert (
        str(R11_1_CAPABILITY_AUDIT.relative_to(ROOT)).replace("\\", "/")
        in r11_1["evidence_files"]
    )
    assert len(cells) == len(sources) * len(adapters) == 12
    assert {(cell["source_kind"], cell["adapter_id"]) for cell in cells} == {
        (source, adapter) for source in sources for adapter in adapters
    }
    assert {cell["status"] for cell in cells} == {
        "verified",
        "explicitly_unavailable",
    }
    assert sum(cell["status"] == "verified" for cell in cells) == 2
    assert sum(cell["status"] == "explicitly_unavailable" for cell in cells) == 10
    assert audit["status_counts"] == {
        "verified": 2,
        "explicitly_unavailable": 10,
    }
    for cell in cells:
        assert (ROOT / cell["evidence"]).is_file()
        assert (cell["reason"] is None) == (cell["status"] == "verified")


def test_r11_3_resampling_matrix_and_adverse_cases_are_exhaustive() -> None:
    """Every source layout, adapter cell, and missing-data class is qualified."""
    requirements = {item["requirement_id"]: item for item in _load()["requirements"]}
    r11_3 = requirements["R11.3"]
    audit = cast(
        dict[str, Any], json.loads(R11_3_CAPABILITY_AUDIT.read_text(encoding="utf-8"))
    )
    r11_1 = cast(
        dict[str, Any], json.loads(R11_1_CAPABILITY_AUDIT.read_text(encoding="utf-8"))
    )

    assert audit["schema_version"] == "tools-r11.3-trace-resampling-capabilities/v1"
    assert audit["requirement_id"] == "R11.3"
    assert audit["qualified_base_revision"] == (
        "66b1cb4d16d8ea36fa7c3f4eb0c4f3725ae03734"  # pragma: allowlist secret
    )
    assert audit["implementation_issue"] == 4763
    assert audit["policy_id"] == "swing-trace-time-linear-contiguous/v1"
    assert audit["adapter_cell_status_authority"] == str(
        R11_1_CAPABILITY_AUDIT.relative_to(ROOT)
    ).replace("\\", "/")
    assert r11_3["status"] == "verified"
    assert r11_3["gaps"] == []
    assert (
        str(R11_3_CAPABILITY_AUDIT.relative_to(ROOT)).replace("\\", "/")
        in r11_3["evidence_files"]
    )

    layouts = audit["source_layouts"]
    declared_layouts = spatial_source_layouts()
    assert tuple(item["source_kind"] for item in layouts) == tuple(declared_layouts)
    assert tuple(item["source_kind"] for item in layouts) == tuple(
        r11_1["source_kinds"]
    )
    assert all(item["status"] == "verified" for item in layouts)
    assert {item["source_kind"]: tuple(item["point_ids"]) for item in layouts} == dict(
        declared_layouts
    )
    assert all(item["point_count"] == len(item["point_ids"]) for item in layouts)

    cells = audit["adapter_cells"]
    assert len(cells) == len(r11_1["cells"]) == 12
    assert [
        (item["source_kind"], item["adapter_id"], item["status"]) for item in cells
    ] == [
        (item["source_kind"], item["adapter_id"], item["status"])
        for item in r11_1["cells"]
    ]
    assert audit["adapter_status_counts"] == r11_1["status_counts"]
    assert set(audit["adverse_cases"]) == {
        "exact_grid_identity",
        "exact_grid_subset",
        "off_grid_affine_interpolation",
        "leading_missing",
        "trailing_missing",
        "interior_gap",
        "single_sample_island",
        "all_invalid_failure",
        "no_impact",
        "impact_lower_tie",
        "impact_without_valid_target",
        "outside_domain",
        "invalid_target_grid",
        "immutability_and_aliasing",
        "serial_chunk_equivalence",
        "source_layout_registry_drift",
        "adapter_matrix_drift",
    }
    assert audit["scientific_boundary"] == (
        "Trace-grid equivalence qualifies software alignment for model outputs; "
        "it does not validate anatomical force attribution, a human swing "
        "mechanism, or universal coaching advice."
    )


def test_r15_upstream_consumption_evidence_is_verified_and_revision_bound() -> None:
    """The merged consumer must bind R15.1--R15.3 to one immutable authority."""
    requirements = {item["requirement_id"]: item for item in _load()["requirements"]}

    for requirement_id in ("R15.1", "R15.2", "R15.3"):
        requirement = requirements[requirement_id]
        assert requirement["status"] == "verified"
        assert requirement["gaps"] == []
        assert UPSTREAM_VARIATION_PR in requirement["remote_evidence"]

    assert PINNED_TOOLS_REVISION in requirements["R15.1"]["rationale"]


def test_r15_4_public_guide_is_complete_neutral_and_reproducible() -> None:
    """The public guide must retain scientific limits and executable entry points."""
    guide = PUBLIC_GUIDE.read_text(encoding="utf-8")
    requirements = {item["requirement_id"]: item for item in _load()["requirements"]}
    r15_4 = requirements["R15.4"]

    for heading in (
        "# Reproducible Ensemble Variation and Sensitivity Analysis",
        "## Scope and Evidence Boundary",
        "## Mechanical and Statistical Interpretation",
        "## Data and Schema Contracts",
        "## Methods and Assumptions",
        "## Quick Start",
        "## Reproducible Verification",
        "## Performance and Scaling Evidence",
        "## Review and Falsification Workflow",
        "## Limitations and Unsupported Inferences",
    ):
        assert heading in guide

    for boundary in (
        "model-scenario evidence",
        "does not establish human validity",
        "does not justify universal coaching advice",
        "No-impact is retained",
        "correlation is not causation",
    ):
        assert boundary in guide

    for command in (
        "$env:PYTHONPATH = (Resolve-Path src).Path",
        "assert serial.success.shape == (plan.n_runs,)",
        "output_available = np.isfinite(serial.outputs)",
        "python -m pytest src/shared/python/swing_sim/variation/tests",
        "python -m pytest tests/rate_of_closure",
        "npm test -- --run variation",
        "python -m ruff check src/shared/python/swing_sim/variation",
    ):
        assert command in guide

    for relative in (
        "docs/specs/VARIATION_PLAN_PERSISTENCE.md",
        "docs/rate_of_closure/variation_visualization_performance.md",
        "docs/rate_of_closure/ensemble_stream_scaling.v1.json",
        "docs/audits/rate_of_closure_epic_4142_evidence.v1.json",
    ):
        assert relative in guide
        assert (ROOT / relative).is_file()

    assert r15_4["status"] == "verified"
    assert r15_4["gaps"] == []
    assert (
        str(PUBLIC_GUIDE.relative_to(ROOT)).replace("\\", "/")
        in r15_4["evidence_files"]
    )
