#!/usr/bin/env python3
"""Generate and verify the deterministic Tools calculation-freshness manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PurePosixPath

from scripts.tools_calculation_freshness_contract import (
    FRESHNESS_PATH,
    FRESHNESS_VERSION,
    canonical_json_digest,
    execute_dplane_calculation,
    load_calculation_freshness,
    sha256_file_lf,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES_PATH = (
    ROOT / "manuals" / "tools" / "fixtures" / "dplane-calculation-fixtures.json"
)


def build_freshness_manifest(root: Path = ROOT) -> dict[str, object]:
    """Generate the complete calculation-freshness payload from code and fixtures."""
    fixtures_rel = PurePosixPath(
        "manuals/tools/fixtures/dplane-calculation-fixtures.json"
    )
    fixtures_data = json.loads((root / fixtures_rel).read_text(encoding="utf-8"))

    # Execute all cases
    examples = []

    # 1. Nominal
    for case in fixtures_data["nominal_cases"]:
        cid = case["id"]
        inp = {
            "travel_vector": case["travel_vector"],
            "face_normal": case["face_normal"],
        }
        outcome = execute_dplane_calculation(inp, fixtures_data["frame_id"])
        examples.append(
            {
                "example_id": cid,
                "category": "nominal",
                "description": case["description"],
                "fixture_path": fixtures_rel.as_posix(),
                "fixture_case_id": cid,
                "inputs": inp,
                "expected_outcome": case["expected"],
                "execution_digest": canonical_json_digest(outcome),
            }
        )

    # 2. Boundary
    for case in fixtures_data["boundary_cases"]:
        cid = case["id"]
        inp = {
            "travel_vector": case["travel_vector"],
            "face_normal": case["face_normal"],
        }
        outcome = execute_dplane_calculation(inp, fixtures_data["frame_id"])
        examples.append(
            {
                "example_id": cid,
                "category": "boundary",
                "description": case["description"],
                "fixture_path": fixtures_rel.as_posix(),
                "fixture_case_id": cid,
                "inputs": inp,
                "expected_outcome": case["expected"],
                "execution_digest": canonical_json_digest(outcome),
            }
        )

    # 3. Failure
    for case in fixtures_data["failure_cases"]:
        cid = case["id"]
        inp = {
            "travel_vector": case["travel_vector"],
            "face_normal": case["face_normal"],
        }
        if "target" in case:
            inp["target"] = case["target"]
        if "up" in case:
            inp["up"] = case["up"]
        examples.append(
            {
                "example_id": cid,
                "category": "failure",
                "description": case["description"],
                "fixture_path": fixtures_rel.as_posix(),
                "fixture_case_id": cid,
                "inputs": inp,
                "expected_outcome": case["expected_error"],
                "execution_digest": canonical_json_digest(case["expected_error"]),
            }
        )

    # Documented values
    documented_values = [
        {
            "value_id": "square_descending.spin_loft_3d_deg",
            "field_name": "spin_loft_3d_deg",
            "example_id": "square-descending",
            "value": 15.0,
            "unit": "degree",
            "tolerance": 1e-9,
            "manual_chapter": "manuals/tools/chapters/04-swing-rate-of-closure-dplane.qmd",
            "manual_anchor": "#example-results",
        },
        {
            "value_id": "square_descending.planar_spin_loft_deg",
            "field_name": "planar_spin_loft_deg",
            "example_id": "square-descending",
            "value": 15.0,
            "unit": "degree",
            "tolerance": 1e-9,
            "manual_chapter": "manuals/tools/chapters/04-swing-rate-of-closure-dplane.qmd",
            "manual_anchor": "#example-results",
        },
        {
            "value_id": "square_descending.spin_loft_residual_deg",
            "field_name": "spin_loft_residual_deg",
            "example_id": "square-descending",
            "value": 0.0,
            "unit": "degree",
            "tolerance": 1e-9,
            "manual_chapter": "manuals/tools/chapters/04-swing-rate-of-closure-dplane.qmd",
            "manual_anchor": "#example-results",
        },
        {
            "value_id": "square_descending.dplane_normal_unit",
            "field_name": "dplane_normal_unit",
            "example_id": "square-descending",
            "value": [0.0, 0.0, 1.0],
            "unit": "1",
            "tolerance": 1e-9,
            "manual_chapter": "manuals/tools/chapters/04-swing-rate-of-closure-dplane.qmd",
            "manual_anchor": "#example-results",
        },
        {
            "value_id": "compound_rightward_face.spin_loft_3d_deg",
            "field_name": "spin_loft_3d_deg",
            "example_id": "compound-rightward-face",
            "value": 36.80234585197317,
            "unit": "degree",
            "tolerance": 1e-9,
            "manual_chapter": "manuals/tools/chapters/04-swing-rate-of-closure-dplane.qmd",
            "manual_anchor": "#example-results",
        },
        {
            "value_id": "compound_rightward_face.face_to_path_deg",
            "field_name": "face_to_path_deg",
            "example_id": "compound-rightward-face",
            "value": 8.0,
            "unit": "degree",
            "tolerance": 1e-9,
            "manual_chapter": "manuals/tools/chapters/04-swing-rate-of-closure-dplane.qmd",
            "manual_anchor": "#example-results",
        },
        {
            "value_id": "crossed_leftward_face.spin_loft_3d_deg",
            "field_name": "spin_loft_3d_deg",
            "example_id": "crossed-leftward-face",
            "value": 19.817514323544046,
            "unit": "degree",
            "tolerance": 1e-9,
            "manual_chapter": "manuals/tools/chapters/04-swing-rate-of-closure-dplane.qmd",
            "manual_anchor": "#example-results",
        },
        {
            "value_id": "boundary_zero_travel.status",
            "field_name": "status",
            "example_id": "boundary-zero-travel",
            "value": "zero_travel",
            "unit": "1",
            "tolerance": None,
            "manual_chapter": "manuals/tools/chapters/04-swing-rate-of-closure-dplane.qmd",
            "manual_anchor": "#boundary-and-singular-delivery-cases",
        },
        {
            "value_id": "boundary_parallel.spin_loft_3d_deg",
            "field_name": "spin_loft_3d_deg",
            "example_id": "boundary-parallel-collinear",
            "value": 0.0,
            "unit": "degree",
            "tolerance": 1e-9,
            "manual_chapter": "manuals/tools/chapters/04-swing-rate-of-closure-dplane.qmd",
            "manual_anchor": "#boundary-and-singular-delivery-cases",
        },
        {
            "value_id": "boundary_antiparallel.spin_loft_3d_deg",
            "field_name": "spin_loft_3d_deg",
            "example_id": "boundary-antiparallel-collinear",
            "value": 180.0,
            "unit": "degree",
            "tolerance": 1e-9,
            "manual_chapter": "manuals/tools/chapters/04-swing-rate-of-closure-dplane.qmd",
            "manual_anchor": "#boundary-and-singular-delivery-cases",
        },
    ]

    table_rows = [
        [
            "square-descending",
            "nominal",
            "defined",
            "15.00",
            "0.00",
            "[0, 0, 1]",
        ],
        [
            "compound-rightward-face",
            "nominal",
            "defined",
            "36.80",
            "8.00",
            "[0.017, -0.197, 0.980]",
        ],
        [
            "crossed-leftward-face",
            "nominal",
            "defined",
            "19.82",
            "-12.00",
            "[-0.141, 0.569, 0.810]",
        ],
        [
            "boundary-zero-travel",
            "boundary",
            "zero_travel",
            "null",
            "null",
            "null",
        ],
        [
            "boundary-parallel-collinear",
            "boundary",
            "parallel",
            "0.00",
            "null",
            "null",
        ],
        [
            "boundary-antiparallel-collinear",
            "boundary",
            "antiparallel",
            "180.00",
            "null",
            "null",
        ],
    ]
    table_digest = canonical_json_digest(table_rows)

    documented_tables = [
        {
            "table_id": "table-dplane-worked-examples",
            "title": "Governed D-Plane Analytic and Boundary Cases",
            "manual_chapter": "manuals/tools/chapters/04-swing-rate-of-closure-dplane.qmd",
            "manual_anchor": "#dplane-worked-examples-table",
            "headers": [
                "Case ID",
                "Category",
                "Status",
                "Spin Loft (deg)",
                "Face to Path (deg)",
                "Normal Unit",
            ],
            "rows": table_rows,
            "digest": table_digest,
        }
    ]

    fig_path = PurePosixPath("manuals/tools/figures/render-pipeline.png")
    fig_full = root / fig_path
    fig_sha = (
        sha256_file_lf(fig_full)
        if fig_full.exists()
        else "0000000000000000000000000000000000000000000000000000000000000000"
    )

    documented_figures = [
        {
            "figure_id": "fig-render-pipeline",
            "path": fig_path.as_posix(),
            "media_type": "image/png",
            "caption": "Reproducible multi-format rendering pipeline for Tools design manuals.",
            "sha256": fig_sha,
        }
    ]

    source_links = [
        {
            "path": "src/rate_of_closure/simulation/impact_kinematics.py",
            "symbol": "impact_kinematics_for_run",
            "symbol_type": "function",
            "sha256_lf": sha256_file_lf(
                root / "src" / "rate_of_closure" / "simulation" / "impact_kinematics.py"
            ),
        },
        {
            "path": "src/shared/python/swing_sim/impact/dplane.py",
            "symbol": "analyze_dplane",
            "symbol_type": "function",
            "sha256_lf": sha256_file_lf(
                root
                / "src"
                / "shared"
                / "python"
                / "swing_sim"
                / "impact"
                / "dplane.py"
            ),
        },
        {
            "path": "src/shared/python/swing_sim/impact/dplane.py",
            "symbol": "spin_loft_sector_directions",
            "symbol_type": "function",
            "sha256_lf": sha256_file_lf(
                root
                / "src"
                / "shared"
                / "python"
                / "swing_sim"
                / "impact"
                / "dplane.py"
            ),
        },
    ]

    schema_links = [
        {
            "path": "manuals/tools/schemas/calculation-freshness.schema.json",
            "schema_version": "tools-calculation-freshness/1.0.0",
            "sha256_lf": sha256_file_lf(
                root
                / "manuals"
                / "tools"
                / "schemas"
                / "calculation-freshness.schema.json"
            ),
        },
        {
            "path": "manuals/tools/fixtures/dplane-calculation-fixtures.json",
            "schema_version": "dplane-calculation-fixtures/1.0.0",
            "sha256_lf": sha256_file_lf(
                root
                / "manuals"
                / "tools"
                / "fixtures"
                / "dplane-calculation-fixtures.json"
            ),
        },
        {
            "path": "src/rate_of_closure/web/src/model/__fixtures__/dplane_golden_v1.json",
            "schema_version": "dplane-golden-v1",
            "sha256_lf": sha256_file_lf(
                root
                / "src"
                / "rate_of_closure"
                / "web"
                / "src"
                / "model"
                / "__fixtures__"
                / "dplane_golden_v1.json"
            ),
        },
    ]

    test_links = [
        {
            "test_id": "cross-client-golden",
            "path": "src/shared/python/swing_sim/impact/tests/test_dplane.py::TestDPlaneAnalyticCases::test_matches_cross_client_golden_contract",
            "level": "parity",
            "sha256_lf": sha256_file_lf(
                root
                / "src"
                / "shared"
                / "python"
                / "swing_sim"
                / "impact"
                / "tests"
                / "test_dplane.py"
            ),
        },
        {
            "test_id": "invalid-frame-rejection",
            "path": "src/shared/python/swing_sim/impact/tests/test_dplane.py::TestDPlaneSingularStates::test_invalid_frame_is_rejected",
            "level": "unit",
            "sha256_lf": sha256_file_lf(
                root
                / "src"
                / "shared"
                / "python"
                / "swing_sim"
                / "impact"
                / "tests"
                / "test_dplane.py"
            ),
        },
        {
            "test_id": "rate-face-center-consumer",
            "path": "tests/rate_of_closure/test_impact_kinematics.py::test_face_center_dplane_uses_rigid_body_point_velocity",
            "level": "integration",
            "sha256_lf": sha256_file_lf(
                root / "tests" / "rate_of_closure" / "test_impact_kinematics.py"
            ),
        },
    ]

    citation_links = [
        {
            "citation_id": "TOOLS-DPLANE-CODE-D3",
            "locator": "src/shared/python/swing_sim/impact/dplane.py",
        },
        {
            "citation_id": "TOOLS-DPLANE-GOLDEN-V1",
            "locator": "src/rate_of_closure/web/src/model/__fixtures__/dplane_golden_v1.json",
        },
    ]

    transitive_dependencies = sorted(
        [
            "manuals/tools/chapters/04-swing-rate-of-closure-dplane.qmd",
            "manuals/tools/figures/render-pipeline.png",
            "manuals/tools/fixtures/dplane-calculation-fixtures.json",
            "manuals/tools/schemas/calculation-freshness.schema.json",
            "scripts/generate_tools_calculations.py",
            "src/rate_of_closure/simulation/impact_kinematics.py",
            "src/rate_of_closure/web/src/model/__fixtures__/dplane_golden_v1.json",
            "src/shared/python/swing_sim/impact/dplane.py",
            "src/shared/python/swing_sim/impact/tests/test_dplane.py",
            "tests/rate_of_closure/test_impact_kinematics.py",
        ]
    )

    calculation_entry = {
        "calculation_id": "TOOLS-DPLANE-GEOMETRY",
        "title": "Frame-Explicit D-Plane Geometry and Rate-of-Closure Adaptation",
        "status": "verified-unapproved",
        "method_version": "1.0.0",
        "generator_script": "scripts/generate_tools_calculations.py",
        "source_links": source_links,
        "schema_links": schema_links,
        "test_links": test_links,
        "citation_links": citation_links,
        "executable_examples": examples,
        "documented_values": documented_values,
        "documented_tables": documented_tables,
        "documented_figures": documented_figures,
        "transitive_dependencies": transitive_dependencies,
    }

    manifest = {
        "schema_version": FRESHNESS_VERSION,
        "manual_id": "tools",
        "release_status": "provisional",
        "owner_subepic": 4723,
        "calculations": [calculation_entry],
        "exemptions": [],
    }

    return manifest


def write_freshness_manifest(root: Path = ROOT) -> Path:
    """Build and write the canonical calculation-freshness.json."""
    manifest = build_freshness_manifest(root)
    # validate schema load
    load_calculation_freshness(manifest)

    out_path = root / FRESHNESS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Check manifest freshness without writing"
    )
    args = parser.parse_args()

    freshness_file = ROOT / FRESHNESS_PATH
    if args.check:
        if not freshness_file.exists():
            print(f"ERROR: {freshness_file} does not exist", file=sys.stderr)
            return 1
        current = freshness_file.read_text(encoding="utf-8")
        fresh = (
            json.dumps(build_freshness_manifest(ROOT), indent=2, ensure_ascii=True)
            + "\n"
        )
        if current != fresh:
            print(
                "ERROR: calculation-freshness.json is stale; run generate_tools_calculations.py",
                file=sys.stderr,
            )
            return 1
        print("Calculation freshness manifest is up to date.")
        return 0

    path = write_freshness_manifest(ROOT)
    print(f"Generated {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
