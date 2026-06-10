from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TAURI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "tauri-build.yml"
MIN_RUST_STACK_BYTES = 536_870_912
MIN_TAURI_CHECK_TIMEOUT_MINUTES = 30


def test_tauri_rust_jobs_reserve_stack_requested_by_rustc() -> None:
    workflow = yaml.safe_load(TAURI_WORKFLOW.read_text(encoding="utf-8"))
    jobs = workflow["jobs"]

    for job_name in ("check", "build"):
        env = jobs[job_name]["env"]
        assert int(env["RUST_MIN_STACK"]) >= MIN_RUST_STACK_BYTES


def test_tauri_rust_cache_does_not_restore_target_directories() -> None:
    workflow = yaml.safe_load(TAURI_WORKFLOW.read_text(encoding="utf-8"))
    jobs = workflow["jobs"]

    cache_steps = [
        step
        for job_name in ("check", "build")
        for step in jobs[job_name]["steps"]
        if step.get("uses", "").startswith("Swatinem/rust-cache")
    ]

    assert len(cache_steps) == 2
    for step in cache_steps:
        assert step["with"]["cache-targets"] is False


def test_tauri_rust_jobs_use_isolated_toolchain_homes() -> None:
    workflow = yaml.safe_load(TAURI_WORKFLOW.read_text(encoding="utf-8"))
    jobs = workflow["jobs"]

    for job_name in ("check", "build"):
        steps = jobs[job_name]["steps"]
        setup_index = next(
            index
            for index, step in enumerate(steps)
            if step.get("name") == "Setup Rust"
        )
        isolate_steps = [
            step
            for step in steps[:setup_index]
            if step.get("name") == "Use isolated Rust homes"
        ]

        assert len(isolate_steps) == 1
        script = isolate_steps[0]["run"]
        assert "RUSTUP_HOME=$RUNNER_TEMP/rustup" in script
        assert "CARGO_HOME=$RUNNER_TEMP/cargo" in script
        assert "$RUNNER_TEMP/cargo/bin" in script


def test_tauri_check_timeout_allows_serial_cold_cache_builds() -> None:
    workflow = yaml.safe_load(TAURI_WORKFLOW.read_text(encoding="utf-8"))
    check_job = workflow["jobs"]["check"]

    assert int(check_job["timeout-minutes"]) >= MIN_TAURI_CHECK_TIMEOUT_MINUTES
