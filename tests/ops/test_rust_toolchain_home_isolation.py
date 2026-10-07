"""Workflow contract: Rust jobs on the self-hosted fleet isolate their toolchain.

Persistent fleet runners share ``~/.rustup`` and ``~/.cargo``. One job's
toolchain install or update deletes binaries from under a concurrent job
(D-sorganization/Repository_Management#2021, Tools#5456). Every job that installs
or runs Rust and can be scheduled on the fleet must therefore point
``RUSTUP_HOME`` and ``CARGO_HOME`` at its own workspace via job-level ``env``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

RUSTUP_HOME_VALUE = "${{ github.workspace }}/.rustup-home"
CARGO_HOME_VALUE = "${{ github.workspace }}/.cargo-home"

_RUST_ACTIONS = ("dtolnay/rust-toolchain", "Swatinem/rust-cache", "actions-rs/")
_RUST_COMMAND = re.compile(r"\b(cargo|rustup|rustc|maturin|wasm-pack)\b")
_HOSTED_RUNNERS = re.compile(r"^(ubuntu|windows|macos)-")


def _is_hosted_only(job: dict[str, Any]) -> bool:
    """Return True when ``runs-on`` is a literal GitHub-hosted label."""
    runs_on = job.get("runs-on")
    labels = runs_on if isinstance(runs_on, list) else [runs_on]
    return all(
        isinstance(label, str) and bool(_HOSTED_RUNNERS.match(label))
        for label in labels
    )


def _uses_rust(job: dict[str, Any]) -> bool:
    for step in job.get("steps") or []:
        if str(step.get("uses", "")).startswith(_RUST_ACTIONS):
            return True
        if _RUST_COMMAND.search(str(step.get("run", ""))):
            return True
    return False


def _fleet_rust_jobs() -> list[tuple[str, str, dict[str, Any]]]:
    found = []
    for path in sorted(WORKFLOWS_DIR.glob("*.yml")):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job_id, job in (workflow.get("jobs") or {}).items():
            if "steps" not in job or _is_hosted_only(job) or not _uses_rust(job):
                continue
            found.append((path.name, job_id, job))
    return found


FLEET_RUST_JOBS = _fleet_rust_jobs()
JOB_IDS = [f"{name}::{job_id}" for name, job_id, _ in FLEET_RUST_JOBS]


def test_detector_finds_the_known_rust_workflows() -> None:
    workflows = {name for name, _, _ in FLEET_RUST_JOBS}
    assert {
        "ci-standard.yml",
        "cross-repo-rust-integration.yml",
        "maturin-ai-backend.yml",
        "publish-artifacts.yml",
        "tauri-build.yml",
    } <= workflows


@pytest.mark.parametrize(("name", "job_id", "job"), FLEET_RUST_JOBS, ids=JOB_IDS)
def test_fleet_rust_job_isolates_rustup_and_cargo_homes(
    name: str, job_id: str, job: dict[str, Any]
) -> None:
    env = job.get("env") or {}
    assert env.get("RUSTUP_HOME") == RUSTUP_HOME_VALUE, f"{name}::{job_id}"
    assert env.get("CARGO_HOME") == CARGO_HOME_VALUE, f"{name}::{job_id}"


def test_rust_job_env_never_uses_runner_temp() -> None:
    for name, job_id, job in FLEET_RUST_JOBS:
        for key in ("RUSTUP_HOME", "CARGO_HOME"):
            assert "runner.temp" not in str(job.get("env", {}).get(key, "")), (
                f"{name}::{job_id}"
            )


def test_isolated_homes_are_gitignored() -> None:
    ignored = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".rustup-home/" in ignored
    assert ".cargo-home/" in ignored
