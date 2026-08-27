"""Clean installed-wheel proof for complete-trial durable readers."""

from __future__ import annotations

import json
import os
import subprocess  # nosec B404 - fixed local interpreters and artifacts
import sys
import sysconfig
import zipfile
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.headless_safe]

ROOT = Path(__file__).resolve().parents[2]
WHEEL_MEMBERS = {
    "rate_of_closure/variation/_complete_trial_fields.py",
    "rate_of_closure/variation/_complete_trial_state.py",
    "rate_of_closure/variation/_complete_trial_wire.py",
    "rate_of_closure/variation/complete_trial_record.py",
    "rate_of_closure/variation/durable_ensemble_chunks.py",
}


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("VIRTUAL_ENV", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PIP_NO_INDEX"] = "1"
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return environment


def _run(
    arguments: list[str], *, cwd: Path, environment: dict[str, str], timeout: int = 180
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # nosec B603 - fixed local command and paths
        arguments,
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.mark.timeout(240)
def test_exact_wheel_round_trips_complete_trial_archive(tmp_path: Path) -> None:
    environment = _environment()
    dist = tmp_path / "dist"
    dist.mkdir()
    built = _run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(dist),
        ],
        cwd=ROOT,
        environment=environment,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    wheels = tuple(dist.glob("ud_tools-*.whl"))
    assert len(wheels) == 1
    wheel = wheels[0].resolve()
    with zipfile.ZipFile(wheel) as archive:
        assert WHEEL_MEMBERS <= set(archive.namelist())

    parent_site = Path(sysconfig.get_paths()["purelib"]).resolve()
    assert parent_site.is_dir()
    assert not parent_site.is_relative_to(ROOT.resolve())

    venv = tmp_path / "venv"
    created = _run(
        [sys.executable, "-m", "venv", str(venv)],
        cwd=tmp_path,
        environment=environment,
    )
    assert created.returncode == 0, created.stdout + created.stderr
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    installed = _run(
        [str(python), "-m", "pip", "install", "--no-deps", str(wheel)],
        cwd=tmp_path,
        environment=environment,
    )
    assert installed.returncode == 0, installed.stdout + installed.stderr

    probe = r"""
import json
import site
import sys
import tempfile
from pathlib import Path

# Reuse only the already-qualified job environment's third-party dependencies.
# The project package itself must still resolve from the exact wheel installed
# in the otherwise isolated child environment, which is asserted below.
site.addsitedir(sys.argv[1])

import rate_of_closure.variation.complete_trial_record as record_module
from rate_of_closure.club import get_club
from rate_of_closure.model import ImpactScenario
from rate_of_closure.simulation import ContactMode, SimulationConfig
from rate_of_closure.variation import (
    COMPLETE_TRIAL_SCHEMA,
    DurableEnsembleChunkSink,
    build_simulation_ensemble_request,
    run_simulation_ensemble_chunks,
)
from shared.python.swing_sim.variation import NoiseSpec, VariationPlan

plan = VariationPlan(
    mode="swing",
    noise=(NoiseSpec("swing_sim.swing.yaw_deg", scale=0.1),),
    n_runs=2,
    seed=4758,
)
config = SimulationConfig(
    scenario=ImpactScenario(clubhead_speed_mph=100.0),
    club=get_club("Driver 10.5°"),
    source_kind="double_pendulum",
    swing_duration_s=0.05,
    contact_mode=ContactMode.DELIVERY_INSPECTION,
)
request = build_simulation_ensemble_request(plan, config)
with tempfile.TemporaryDirectory() as raw:
    directory = Path(raw).resolve()
    run_simulation_ensemble_chunks(
        request, DurableEnsembleChunkSink(directory), chunk_size=1
    )
    records = []
    archive = DurableEnsembleChunkSink(directory).scan(
        request, lambda chunk: records.extend(chunk.complete_records)
    )
    print(json.dumps({
        "schema": COMPLETE_TRIAL_SCHEMA,
        "retention": archive.trial_record_schema,
        "records": len(records),
        "origin": str(Path(record_module.__file__).resolve()),
    }))
"""
    checked = _run(
        [str(python), "-I", "-c", probe, str(parent_site)],
        cwd=tmp_path,
        environment=environment,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    payload = json.loads(checked.stdout.strip().splitlines()[-1])
    assert payload["schema"] == "rate-complete-trial/v1"
    assert payload["retention"] == "rate-complete-trial/v1"
    assert payload["records"] == 2
    origin = Path(payload["origin"]).resolve()
    assert origin.is_relative_to(venv.resolve())
    assert not origin.is_relative_to(ROOT.resolve())
