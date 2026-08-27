"""Measure bounded complete-trial retention memory and artifact scaling."""

from __future__ import annotations

import gc
import json
import platform
import subprocess
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import TypedDict

from rate_of_closure.club import get_club
from rate_of_closure.model import ImpactScenario
from rate_of_closure.simulation import ContactMode, SimulationConfig
from rate_of_closure.variation.durable_ensemble_chunks import DurableEnsembleChunkSink
from rate_of_closure.variation.request_builder import build_simulation_ensemble_request
from rate_of_closure.variation.simulation_adapter import run_simulation_ensemble_chunks
from shared.python.swing_sim.variation import NoiseSpec, VariationPlan

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/rate_of_closure/complete_trial_retention_scaling.v1.json"
TRIAL_COUNTS = (16, 64)
CHUNK_SIZE = 4


class Measurement(TypedDict):
    trial_count: int
    chunk_size: int
    chunk_count: int
    archive_bytes: int
    max_chunk_bytes: int
    peak_python_bytes: int
    elapsed_s: float


def _source_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _request(trial_count: int):  # type: ignore[no-untyped-def]
    plan = VariationPlan(
        mode="swing",
        noise=(NoiseSpec("swing_sim.swing.yaw_deg", scale=0.5),),
        n_runs=trial_count,
        seed=4758,
    )
    base = SimulationConfig(
        scenario=ImpactScenario(clubhead_speed_mph=100.0),
        club=get_club("Driver 10.5°"),
        source_kind="double_pendulum",
        swing_duration_s=0.05,
        contact_mode=ContactMode.DELIVERY_INSPECTION,
    )
    return build_simulation_ensemble_request(plan, base)


def _measure(trial_count: int) -> Measurement:
    gc.collect()
    with tempfile.TemporaryDirectory(prefix=f"rate-r11-1-{trial_count}-") as raw:
        directory = Path(raw).resolve()
        tracemalloc.start()
        started = time.perf_counter()
        archive = run_simulation_ensemble_chunks(
            _request(trial_count),
            DurableEnsembleChunkSink(directory),
            chunk_size=CHUNK_SIZE,
        )
        elapsed_s = time.perf_counter() - started
        _current, peak_python_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        files = tuple(path for path in directory.rglob("*") if path.is_file())
        chunk_files = tuple(directory.glob("chunk-*.npz"))
        return {
            "trial_count": trial_count,
            "chunk_size": CHUNK_SIZE,
            "chunk_count": archive.chunk_count,
            "archive_bytes": sum(path.stat().st_size for path in files),
            "max_chunk_bytes": max(path.stat().st_size for path in chunk_files),
            "peak_python_bytes": peak_python_bytes,
            "elapsed_s": round(elapsed_s, 6),
        }


def main() -> int:
    scenarios = [_measure(count) for count in TRIAL_COUNTS]
    small, large = scenarios
    artifact_per_trial_small = small["archive_bytes"] / small["trial_count"]
    artifact_per_trial_large = large["archive_bytes"] / large["trial_count"]
    document = {
        "schema_version": "tools-complete-trial-retention-scaling/v1",
        "source_revision": _source_revision(),
        "record_schema": "rate-complete-trial/v1",
        "durable_schema_version": 3,
        "measurement_command": (
            "python -m scripts.measure_complete_trial_retention_scaling"
        ),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "implementation": platform.python_implementation(),
        },
        "scenarios": scenarios,
        "observed": {
            "peak_python_ratio_64_to_16": round(
                large["peak_python_bytes"] / small["peak_python_bytes"], 6
            ),
            "artifact_bytes_per_trial_ratio_64_to_16": round(
                artifact_per_trial_large / artifact_per_trial_small, 6
            ),
        },
        "declared_bounds": {
            "max_chunk_file_bytes": 16000000,
            "max_chunk_uncompressed_bytes": 32000000,
            "peak_python_ratio_64_to_16_max": 2.0,
            "artifact_bytes_per_trial_ratio_64_to_16_max": 1.25,
        },
        "scientific_boundary": (
            "This is software resource-scaling evidence for deterministic model "
            "records; it is not human-swing validation or coaching evidence."
        ),
    }
    OUTPUT.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
