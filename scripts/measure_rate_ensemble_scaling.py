"""Measure durable Rate ensemble scaling in isolated Python processes.

The workload intentionally raises a retained numerical failure for every trial.
It measures transport, validation, bounded trace allocation, atomic persistence,
and manifest growth without claiming solver throughput or scientific evidence.
"""

# ruff: noqa: E402 -- repository source path is bootstrapped before local imports.

from __future__ import annotations

import argparse
import ctypes
import json
import platform
import subprocess
import sys
import tempfile
import time
from ctypes import wintypes
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rate_of_closure.club import get_club
from rate_of_closure.model import ImpactScenario
from rate_of_closure.simulation import SimulationConfig
from rate_of_closure.variation.durable_ensemble_chunks import DurableEnsembleChunkSink
from rate_of_closure.variation.ensemble_scaling_evidence import parse_scaling_evidence
from rate_of_closure.variation.request_builder import build_simulation_ensemble_request
from rate_of_closure.variation.simulation_adapter import run_simulation_ensemble_chunks
from shared.python.swing_sim.variation import CATEGORY_SWING, NoiseSpec, VariationPlan

DEFAULT_OUTPUT = ROOT / "docs/rate_of_closure/ensemble_stream_scaling.v1.json"
POINT_COUNT = 3
CHUNK_SIZE = 16
CASES = (("baseline", 128, 51), ("trial-axis", 512, 51), ("trace-axis", 128, 501))
BUDGETS = {
    "max_peak_resident_bytes": 536_870_912,
    "max_trial_axis_peak_growth_bytes": 67_108_864,
    "max_trace_axis_peak_growth_bytes": 100_663_296,
    "min_trials_per_second": 1.0,
}


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _peak_resident_bytes() -> int:
    """Return OS-recorded peak resident bytes for this fresh worker."""
    if platform.system() == "Windows":
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(_ProcessMemoryCounters),
            wintypes.DWORD,
        )
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        process = kernel32.GetCurrentProcess()
        success = psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb
        )
        if not success:
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.PeakWorkingSetSize)
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)  # type: ignore[attr-defined]
    peak = int(usage.ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def _source(trials: int, samples: int):  # type: ignore[no-untyped-def]
    duration_s = (samples - 1) * 0.001
    config = SimulationConfig(
        scenario=ImpactScenario(clubhead_speed_mph=30.0),
        club=get_club("Driver 10.5°"),
        source_kind="double_pendulum",
        swing_duration_s=duration_s,
    )
    plan = VariationPlan(
        mode="swing",
        noise=(NoiseSpec(f"{CATEGORY_SWING}.yaw_deg", scale=0.1),),
        n_runs=trials,
        seed=4626,
    )
    return build_simulation_ensemble_request(plan, config)


def _expected_failure(_config: SimulationConfig):  # type: ignore[no-untyped-def]
    raise RuntimeError("registered synthetic transport failure")


def _logical_trace_bytes(trials: int, samples: int) -> int:
    positions = trials * samples * POINT_COUNT * 3 * 8
    validity = trials * samples
    impacts = trials * 8
    inputs = trials * 8
    return positions + validity + impacts + inputs


def _worker(case_id: str, trials: int, samples: int) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="rate-ensemble-scaling-") as raw:
        directory = Path(raw).resolve() / "archive"
        started = time.perf_counter()
        archive = run_simulation_ensemble_chunks(
            _source(trials, samples),
            DurableEnsembleChunkSink(directory),
            chunk_size=CHUNK_SIZE,
            executor=_expected_failure,
        )
        elapsed_s = time.perf_counter() - started
        archive_bytes = sum(path.stat().st_size for path in directory.rglob("*"))
        if archive.status != "complete" or archive.next_index != trials:
            raise RuntimeError("scaling archive did not complete")
        peak_resident_bytes = _peak_resident_bytes()
    throughput = trials / elapsed_s
    passed = (
        peak_resident_bytes <= BUDGETS["max_peak_resident_bytes"]
        and throughput >= BUDGETS["min_trials_per_second"]
    )
    return {
        "case_id": case_id,
        "trial_count": trials,
        "trace_sample_count": samples,
        "point_count": POINT_COUNT,
        "chunk_size": CHUNK_SIZE,
        "elapsed_s": round(elapsed_s, 6),
        "trials_per_second": round(throughput, 6),
        "peak_resident_bytes": peak_resident_bytes,
        "archive_bytes": archive_bytes,
        "logical_trace_bytes": _logical_trace_bytes(trials, samples),
        "passed": passed,
    }


def _measure_case(case_id: str, trials: int, samples: int) -> dict[str, object]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        case_id,
        str(trials),
        str(samples),
    ]
    result = subprocess.run(
        command, cwd=ROOT, check=True, text=True, capture_output=True
    )
    value = json.loads(result.stdout.splitlines()[-1])
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise RuntimeError("scaling worker returned an invalid document")
    return cast(dict[str, object], value)


def _source_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _report() -> dict[str, Any]:
    observations = [_measure_case(*case) for case in CASES]
    report: dict[str, Any] = {
        "schema_id": "rate-of-closure/ensemble-stream-scaling-evidence",
        "schema_version": 1,
        "measurement_policy": "synthetic-failure-transport-diagnostic",
        "source_commit": _source_commit(),
        "generated_utc": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "workload": "lazy-config-preflight-plus-retained-failure-durable-chunks",
        "budgets": BUDGETS,
        "observations": observations,
        "passed": all(item["passed"] is True for item in observations),
    }
    parse_scaling_evidence(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worker", nargs=3, metavar=("CASE", "TRIALS", "SAMPLES"))
    args = parser.parse_args()
    if args.worker:
        case_id, trials, samples = args.worker
        print(json.dumps(_worker(case_id, int(trials), int(samples)), sort_keys=True))
        return 0
    report = _report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
