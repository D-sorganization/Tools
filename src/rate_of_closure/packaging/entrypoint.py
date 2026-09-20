"""Standalone frozen entry point and qualification seam for Rate of Closure.

Does not import repository _bootstrap.py or rely on editable workspace paths.
Operates both as the interactive PyQt6 double-clickable GUI entry point and as
the headless qualification and parity runner for frozen Windows artifacts.
"""

from __future__ import annotations

import argparse
import importlib.resources
import json
import os
import sys
from pathlib import Path
from typing import Any

# Ensure all managed variation keys (turf, putting) are loaded in the registry
import shared.python.golf_club.turf_variation  # noqa: F401

DIRECT_WORKER_RECOVERY_UNSUPPORTED_REASON = (
    "PyQt direct-worker restart recovery is separately tracked and unsupported "
    "in this release"
)


def _ensure_stdio() -> None:
    """Safeguard standard streams when running under Windows windowed subsystem."""
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115


def probe_frozen_capabilities() -> dict[str, Any]:
    """Inspect environment, collected models, optional-Rust, and recovery claim."""
    _ensure_stdio()

    # Package versions
    package_versions: dict[str, str] = {}
    for pkg in ("PyQt6", "matplotlib", "scipy", "numpy"):
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "unknown")
            package_versions[pkg] = str(version)
        except ImportError:
            package_versions[pkg] = "missing"

    # Collected model and configuration assets
    pkg_files = importlib.resources.files("rate_of_closure")
    collected_models = {
        "locus_capabilities": pkg_files.joinpath(
            "locus_execution_capabilities.v1.json"
        ).is_file(),
        "neural_capabilities": pkg_files.joinpath(
            "data", "neural_vendor_capabilities.v2.json"
        ).is_file(),
        "example_driver_head": pkg_files.joinpath(
            "assets", "example_driver_head.stl"
        ).is_file(),
        "visual_baselines": pkg_files.joinpath("visual_baselines.v1.json").is_file(),
        "visualization_tabs": pkg_files.joinpath(
            "visualization_tabs.v1.json"
        ).is_file(),
    }

    # Graceful optional-Rust capability messaging
    rust_info: dict[str, Any]
    try:
        from pendulum_simulator.src.double_pendulum_golf.native_backend import (
            golfer_native_available,
        )

        has_rust = bool(golfer_native_available())
    except (ImportError, ModuleNotFoundError):
        has_rust = False

    if has_rust:
        rust_info = {
            "available": True,
            "backend": "rust",
            "detail": "Compiled Rust acceleration is active.",
        }
    else:
        rust_info = {
            "available": False,
            "backend": "python",
            "detail": (
                "Pure-Python reference physics active; optional Rust "
                "acceleration is not compiled into this bundle."
            ),
        }

    return {
        "schema_version": "rate-of-closure/frozen-qualification/v1",
        "frozen": getattr(sys, "frozen", False),
        "package_versions": package_versions,
        "collected_models": collected_models,
        "rust_capability": rust_info,
        "direct_worker_restart_recovery": {
            "supported": False,
            "status": "unsupported",
            "reason": DIRECT_WORKER_RECOVERY_UNSUPPORTED_REASON,
        },
    }


def run_canonical_simulation(
    output_file: Path | None = None,
) -> dict[str, Any]:
    """Execute the bounded canonical simulation and extract core metrics."""
    _ensure_stdio()
    from rate_of_closure.club import get_club
    from rate_of_closure.model import ImpactScenario
    from rate_of_closure.simulation import SimulationConfig, run_simulation

    scenario = ImpactScenario(clubhead_speed_mph=113.0)
    club = get_club("Driver 10.5°")
    config = SimulationConfig(
        scenario=scenario,
        club=club,
        source_kind="double_pendulum",
    )
    run = run_simulation(config)

    import numpy as np

    carry_m = (
        float(run.launch["carry_m"]) if run.launch and "carry_m" in run.launch else 0.0
    )
    ball_speed_mph = (
        float(run.launch["ball_speed_mph"])
        if run.launch and "ball_speed_mph" in run.launch
        else 0.0
    )
    delivery_speed_mps = (
        float(np.linalg.norm(run.delivery.clubhead_velocity))
        if run.delivery is not None
        else 0.0
    )
    attack_angle = (
        float(run.delivery.dplane.attack_angle_deg) if run.delivery is not None else 0.0
    )
    dynamic_loft = (
        float(run.delivery.dplane.dynamic_loft_deg) if run.delivery is not None else 0.0
    )

    result: dict[str, Any] = {
        "status": "success",
        "impact_outcome": run.impact_outcome.status.value,
        "metrics": {
            "scenario_speed_mph": config.scenario.clubhead_speed_mph,
            "clubhead_speed_mps": delivery_speed_mps,
            "ball_speed_mph": ball_speed_mph,
            "carry_distance_m": carry_m,
            "attack_angle_deg": attack_angle,
            "dynamic_loft_deg": dynamic_loft,
        },
    }

    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


def execute_ground_study_evidence(
    result_dest: Path,
    csv_dest: Path | None = None,
    job_path: Path | None = None,
) -> dict[str, Any]:
    """Execute Ground Study job and atomically save result evidence and CSV rows."""
    _ensure_stdio()
    from rate_of_closure.application.regional_ground_execution_files import (
        write_regional_ground_execution_result_atomic,
        write_regional_ground_execution_rows_csv_atomic,
    )
    from rate_of_closure.application.regional_ground_execution_job import (
        regional_ground_execution_job_from_json,
    )
    from rate_of_closure.application.regional_ground_execution_result import (
        build_regional_ground_execution_result,
    )
    from rate_of_closure.variation.scalar_ensemble_wire import (
        scalar_ensemble_dataset_from_wire,
    )

    pkg_files = importlib.resources.files("rate_of_closure")
    fixtures_dir = pkg_files.joinpath("web", "src", "model", "__fixtures__")

    if job_path is not None and job_path.is_file():
        job_doc = json.loads(job_path.read_text(encoding="utf-8"))
        job = regional_ground_execution_job_from_json(
            json.dumps(job_doc.get("job", job_doc))
        )
    else:
        golden_job_res = fixtures_dir.joinpath(
            "regional_ground_execution_job_golden_v1.json"
        )
        job_doc = json.loads(golden_job_res.read_text(encoding="utf-8"))
        job = regional_ground_execution_job_from_json(json.dumps(job_doc["job"]))

    golden_result_res = fixtures_dir.joinpath(
        "regional_ground_execution_result_golden_v1.json"
    )
    result_doc = json.loads(golden_result_res.read_text(encoding="utf-8"))
    dataset_wire = result_doc["result"]["dataset"]

    dataset = scalar_ensemble_dataset_from_wire(dataset_wire)
    result = build_regional_ground_execution_result(job, dataset)

    write_regional_ground_execution_result_atomic(result, result_dest)
    if csv_dest is not None:
        write_regional_ground_execution_rows_csv_atomic(result, csv_dest)

    return {
        "status": "succeeded",
        "job_id": job.job_id,
        "dataset_sha256": result.dataset_sha256,
        "canonical_sha256": result.canonical_sha256,
        "result_path": str(result_dest),
        "csv_path": str(csv_dest) if csv_dest else None,
    }


def run_offscreen_smoke_test() -> int:
    """Boot the real PyQt6 main window offscreen, verify widgets, and exit cleanly."""
    _ensure_stdio()
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    from PyQt6.QtWidgets import QApplication

    from rate_of_closure.ui.pyqt6.main_window import RateOfClosureMainWindow

    app = QApplication.instance()
    owns_app = False
    if app is None:
        app = QApplication(["RateOfClosureExplorer", "-platform", "offscreen"])
        owns_app = True

    window = RateOfClosureMainWindow()
    window.show()
    app.processEvents()

    # Verify primary shell components
    assert window._tabs.count() > 0, "tabs must be populated"
    assert window._controls is not None, "controls panel must be present"

    window.close()
    app.processEvents()
    if owns_app:
        app.quit()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Unified CLI and interactive launcher."""
    _ensure_stdio()
    parser = argparse.ArgumentParser(description="Rate of Closure Impact Explorer")
    parser.add_argument(
        "--offscreen", action="store_true", help="Run Qt in offscreen mode"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run offscreen smoke test and exit 0",
    )
    parser.add_argument(
        "--probe-capabilities",
        action="store_true",
        help="Print frozen capability metadata as JSON",
    )
    parser.add_argument(
        "--run-canonical-simulation",
        action="store_true",
        help="Run bounded canonical simulation and print result JSON",
    )
    parser.add_argument(
        "--simulation-output",
        type=Path,
        default=None,
        help="Optional destination path for simulation result JSON",
    )
    parser.add_argument(
        "--execute-ground-study",
        action="store_true",
        help="Execute Ground Study evidence and save output JSON/CSV",
    )
    parser.add_argument(
        "--job-file",
        type=Path,
        default=None,
        help="Optional path to custom execution job JSON",
    )
    parser.add_argument(
        "--result-file",
        type=Path,
        default=None,
        help="Destination path for Ground Study execution result JSON",
    )
    parser.add_argument(
        "--csv-file",
        type=Path,
        default=None,
        help="Optional destination path for Ground Study CSV rows",
    )

    args = parser.parse_args(argv)

    if args.probe_capabilities:
        caps = probe_frozen_capabilities()
        sys.stdout.write(json.dumps(caps, indent=2) + "\n")
        sys.stdout.flush()
        return 0

    if args.run_canonical_simulation:
        sim = run_canonical_simulation(output_file=args.simulation_output)
        sys.stdout.write(json.dumps(sim, indent=2) + "\n")
        sys.stdout.flush()
        return 0

    if args.execute_ground_study:
        if args.result_file is None:
            sys.stderr.write("--result-file is required for --execute-ground-study\n")
            return 1
        res = execute_ground_study_evidence(
            result_dest=args.result_file,
            csv_dest=args.csv_file,
            job_path=args.job_file,
        )
        sys.stdout.write(json.dumps(res, indent=2) + "\n")
        sys.stdout.flush()
        return 0

    if args.smoke_test:
        return run_offscreen_smoke_test()

    if args.offscreen:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

    from rate_of_closure.ui.pyqt6.launcher import launch_rate_pyqt6

    return int(launch_rate_pyqt6())


if __name__ == "__main__":
    sys.exit(main())
