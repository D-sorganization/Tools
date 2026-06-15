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

    check_steps = jobs["check"]["steps"]
    check_setup_index = next(
        index
        for index, step in enumerate(check_steps)
        if step.get("name") == "Setup Rust"
    )
    check_isolate_steps = [
        step
        for step in check_steps[:check_setup_index]
        if step.get("name") == "Use isolated Rust homes"
    ]

    assert len(check_isolate_steps) == 1
    check_script = check_isolate_steps[0]["run"]
    assert "RUSTUP_HOME=$RUNNER_TEMP/rustup" in check_script
    assert "CARGO_HOME=$RUNNER_TEMP/cargo" in check_script
    assert "$RUNNER_TEMP/cargo/bin" in check_script

    build_steps = jobs["build"]["steps"]
    build_setup_index = next(
        index
        for index, step in enumerate(build_steps)
        if step.get("name") == "Setup Rust"
    )
    build_isolate_steps = [
        step
        for step in build_steps[:build_setup_index]
        if step.get("name", "").startswith("Use isolated Rust homes")
    ]

    assert {step["name"] for step in build_isolate_steps} == {
        "Use isolated Rust homes (Linux)",
        "Use isolated Rust homes (Windows)",
    }
    linux_script = next(
        step["run"]
        for step in build_isolate_steps
        if step["name"] == "Use isolated Rust homes (Linux)"
    )
    windows_script = next(
        step["run"]
        for step in build_isolate_steps
        if step["name"] == "Use isolated Rust homes (Windows)"
    )
    assert "RUSTUP_HOME=$RUNNER_TEMP/rustup" in linux_script
    assert "CARGO_HOME=$RUNNER_TEMP/cargo" in linux_script
    assert "$RUNNER_TEMP/cargo/bin" in linux_script
    assert 'Join-Path $env:RUNNER_TEMP "rustup"' in windows_script
    assert 'Join-Path $env:RUNNER_TEMP "cargo"' in windows_script


def test_tauri_build_matrix_uses_stable_labels_for_display_and_artifacts() -> None:
    workflow = yaml.safe_load(TAURI_WORKFLOW.read_text(encoding="utf-8"))
    build_job = workflow["jobs"]["build"]
    platforms = build_job["strategy"]["matrix"]["platform"]

    assert (
        build_job["name"]
        == "Build ${{ matrix.app.name }} (${{ matrix.platform.label }})"
    )
    assert build_job["runs-on"] == "${{ matrix.platform.runs_on }}"
    assert {platform["label"] for platform in platforms} == {"linux-x64", "windows-x64"}
    assert all("os" not in platform for platform in platforms)

    upload_step = next(
        step for step in build_job["steps"] if step.get("name") == "Upload artifacts"
    )
    assert (
        upload_step["with"]["name"]
        == "${{ matrix.app.name }}-${{ matrix.platform.label }}"
    )


def test_tauri_windows_build_uses_powershell_for_runner_path_setup() -> None:
    workflow = yaml.safe_load(TAURI_WORKFLOW.read_text(encoding="utf-8"))
    build_steps = workflow["jobs"]["build"]["steps"]

    windows_steps = [
        step
        for step in build_steps
        if step.get("if") == "matrix.platform.label == 'windows-x64'"
    ]

    windows_run_steps = [step for step in windows_steps if "run" in step]

    assert windows_run_steps
    assert all(step["shell"] == "pwsh" for step in windows_run_steps)
    assert all(step["name"].endswith("(Windows)") for step in windows_run_steps)


def test_tauri_check_timeout_allows_serial_cold_cache_builds() -> None:
    workflow = yaml.safe_load(TAURI_WORKFLOW.read_text(encoding="utf-8"))
    check_job = workflow["jobs"]["check"]

    assert int(check_job["timeout-minutes"]) >= MIN_TAURI_CHECK_TIMEOUT_MINUTES


def test_tauri_local_node_selection_skips_broken_npm_toolcaches() -> None:
    workflow = yaml.safe_load(TAURI_WORKFLOW.read_text(encoding="utf-8"))
    jobs = workflow["jobs"]

    local_node_steps = [
        step
        for job_name in ("check", "build")
        for step in jobs[job_name]["steps"]
        if step.get("name", "").startswith("Use local Node.js 24 toolcache")
    ]

    assert len(local_node_steps) == 2
    for step in local_node_steps:
        script = step["run"]
        assert 'PATH="$candidate:$PATH" "$candidate/npm" --version >/dev/null' in script
        assert "::warning::Skipping broken Node.js toolcache" in script
        assert '[ -z "$node_dir" ] || [ ! -x "$node_dir/node" ]' in script
