"""Security and reproducibility contracts for the Rate Playwright workflows."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
PR_WORKFLOW_PATH = WORKFLOW_DIR / "rate-web-playwright.yml"
TRUSTED_WORKFLOW_PATH = WORKFLOW_DIR / "rate-web-playwright-trusted.yml"
RUNNER_GUARD_PATH = WORKFLOW_DIR / "local-only-runner-guard.yml"
FULL_ACTION_SHA = re.compile(r"^[^@]+@[0-9a-f]{40}$")
PLAYWRIGHT_EVIDENCE_PATHS = (
    "src/rate_of_closure/web/playwright-report/\n"
    "src/rate_of_closure/web/test-results/\n"
)
PR_EVIDENCE_PATHS = (
    PLAYWRIGHT_EVIDENCE_PATHS + "rate-pyqt-screenshots/\nvisual-baseline-candidates/\n"
)
TRUSTED_PYQT_EVIDENCE_PATHS = "rate-pyqt-screenshots/\nvisual-baseline-candidates/\n"
TRUSTED_CANDIDATE_ARTIFACT = (
    "rate-web-baseline-candidates-${{ github.run_id }}-${{ github.run_attempt }}"
)
PYQT_AUTHORITY_PATHS = {
    "requirements-rate-pyqt.txt",
    "scripts/check_rate_pyqt_environment.py",
    "src/rate_of_closure/club/**",
    "src/rate_of_closure/club_camera.py",
    "src/rate_of_closure/club_mesh_source.py",
    "src/rate_of_closure/flight_accepted_study.py",
    "src/rate_of_closure/flight_sample_inspector.py",
    "src/rate_of_closure/mesh.py",
    "src/rate_of_closure/model.py",
    "src/rate_of_closure/plot_point_inspector.py",
    "src/rate_of_closure/plot_workspace_limits.py",
    "src/rate_of_closure/plotting/**",
    "src/rate_of_closure/putting.py",
    "src/rate_of_closure/putting_sample_inspector.py",
    "src/rate_of_closure/putting_result_contract.py",
    "src/rate_of_closure/simulation/**",
    "src/rate_of_closure/variation/**",
    "src/rate_of_closure/variation_visual_state.py",
    "src/rate_of_closure/visual_layout_preferences.py",
    "src/rate_of_closure/ui/pyqt6/**",
    "src/rate_of_closure/visualization_tab_manifest.py",
    "src/rate_of_closure/visualization_tabs.v1.json",
    "src/rate_of_closure/visualization_performance_manifest.py",
    "src/rate_of_closure/visualization_performance.v1.json",
    "src/rate_of_closure/visualization_accessibility_manifest.py",
    "src/rate_of_closure/visualization_accessibility.v1.json",
    "src/rate_of_closure/visual_baseline_compare.py",
    "src/rate_of_closure/visual_baseline_manifest.py",
    "src/rate_of_closure/visual_baselines.v1.json",
    "src/rate_of_closure/visual_baselines/**",
    "src/shared/python/swing_sim/variation/**",
    "src/shared/python/swing_sim/putting.py",
    "tests/rate_of_closure/pyqt_putting_sample_inspector_probe.py",
    "tests/rate_of_closure/pyqt_club_camera_probe.py",
    "tests/rate_of_closure/pyqt_flight_sample_inspector_probe.py",
    "tests/rate_of_closure/pyqt_simulation_scrub_probe.py",
    "tests/rate_of_closure/pyqt_plot_point_inspector_probe.py",
    "tests/rate_of_closure/pyqt_visual_layout_persistence_probe.py",
    "tests/rate_of_closure/test_club_camera.py",
    "tests/rate_of_closure/test_club_mesh_source.py",
    "tests/rate_of_closure/test_club_view_camera.py",
    "tests/rate_of_closure/test_flight_accepted_study.py",
    "tests/rate_of_closure/test_flight_explorer.py",
    "tests/rate_of_closure/test_flight_explorer_atomic_gui.py",
    "tests/rate_of_closure/test_flight_sample_inspector.py",
    "tests/rate_of_closure/test_flight_sample_inspector_gui.py",
    "tests/rate_of_closure/test_mesh.py",
    "tests/rate_of_closure/test_plot_canvas_inspector.py",
    "tests/rate_of_closure/test_plot_point_inspector.py",
    "tests/rate_of_closure/test_plot_workspace_limits.py",
    "tests/rate_of_closure/test_plots_gui.py",
    "tests/rate_of_closure/test_plots_async_performance.py",
    "tests/rate_of_closure/test_plotting.py",
    "tests/rate_of_closure/test_pyqt_club_camera_rendered.py",
    "tests/rate_of_closure/test_pyqt_flight_sample_inspector_rendered.py",
    "tests/rate_of_closure/test_pyqt_simulation_scrub_rendered.py",
    "tests/rate_of_closure/test_pyqt_plot_point_inspector_rendered.py",
    "tests/rate_of_closure/test_pyqt_visual_layout_persistence_rendered.py",
    "tests/rate_of_closure/test_simulation_gui.py",
    "tests/rate_of_closure/test_simulation_scrub_authority.py",
    "tests/rate_of_closure/test_pyqt_putting_sample_inspector_rendered.py",
    "tests/rate_of_closure/pyqt_variation_render_probe.py",
    "tests/rate_of_closure/test_pyqt_variation_rendered_interactions.py",
    "tests/rate_of_closure/pyqt_visualization_tab_probe.py",
    "tests/rate_of_closure/test_pyqt_visualization_tab_visibility.py",
    "tests/rate_of_closure/pyqt_variation_visual_state_probe.py",
    "tests/rate_of_closure/test_pyqt_variation_visual_state_rendered.py",
    "tests/rate_of_closure/test_visualization_tab_manifest.py",
    "tests/rate_of_closure/test_visualization_tab_audit.py",
    "tests/rate_of_closure/test_visualization_performance_manifest.py",
    "tests/rate_of_closure/test_visualization_accessibility.py",
    "tests/rate_of_closure/test_visual_baseline_compare.py",
    "tests/rate_of_closure/test_visual_layout_gui.py",
    "tests/rate_of_closure/test_visual_layout_preferences.py",
    "tests/scripts/test_check_rate_pyqt_environment.py",
    "tests/ops/test_rate_web_playwright_workflow.py",
    "pyproject.toml",
}
FULL_WINDOW_IMPORT_DEPENDENCIES = {
    "pandas": "pandas>=2.0,<3",
    "scipy": "scipy>=1.10.0,<1.18",
    "sympy": "sympy>=1.12",
}


def _workflow(path: Path) -> dict[Any, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _run_steps(job: dict[str, Any]) -> dict[str, str]:
    return {
        str(step["name"]): str(step["run"])
        for step in job["steps"]
        if isinstance(step, dict) and "name" in step and "run" in step
    }


def _checkout(job: dict[str, Any]) -> dict[str, Any]:
    return next(
        step
        for step in job["steps"]
        if str(step.get("uses", "")).startswith("actions/checkout@")
    )


def _named_step(job: dict[str, Any], name: str) -> dict[str, Any]:
    return next(step for step in job["steps"] if step.get("name") == name)


def test_pull_request_workflow_is_hosted_only_without_fleet_vocabulary() -> None:
    text = PR_WORKFLOW_PATH.read_text(encoding="utf-8")
    workflow = _workflow(PR_WORKFLOW_PATH)

    assert "d-sorg-fleet" not in text.lower()
    assert "self-hosted" not in text.lower()
    assert "\n  pull_request:" in text
    assert "\n  push:" not in text
    assert "workflow_dispatch" not in text
    assert set(workflow["jobs"]) == {"production-worker-e2e"}
    assert workflow["jobs"]["production-worker-e2e"]["runs-on"] == "ubuntu-latest"


def test_trusted_workflow_is_main_push_only_without_untrusted_ref_seam() -> None:
    text = TRUSTED_WORKFLOW_PATH.read_text(encoding="utf-8")
    jobs = _workflow(TRUSTED_WORKFLOW_PATH)["jobs"]

    assert "pull_request" not in text
    assert "github.event.pull_request" not in text
    assert "inputs." not in text
    assert "${{ github.ref" not in text
    assert "${{ github.head_ref" not in text
    assert "${{ github.sha" not in text
    assert "\n  push:" in text
    assert "workflow_dispatch" not in text
    assert set(jobs) == {
        "push-production-worker-e2e",
        "push-pyqt-rendered-evidence",
    }
    assert all(job["runs-on"] == "d-sorg-fleet" for job in jobs.values())

    for job in jobs.values():
        push_checkout = _checkout(job)
        assert "with" not in push_checkout or "ref" not in push_checkout["with"]


def test_pr_runs_locked_cross_browser_gate_and_trusted_keeps_chromium_gate() -> None:
    pr_job = _workflow(PR_WORKFLOW_PATH)["jobs"]["production-worker-e2e"]
    trusted_jobs = _workflow(TRUSTED_WORKFLOW_PATH)["jobs"]
    trusted_web_job = trusted_jobs["push-production-worker-e2e"]
    trusted_pyqt_job = trusted_jobs["push-pyqt-rendered-evidence"]
    pr_commands = _run_steps(pr_job)
    trusted_web_commands = _run_steps(trusted_web_job)
    trusted_pyqt_commands = _run_steps(trusted_pyqt_job)

    trusted_step_names = [step.get("name") for step in trusted_pyqt_job["steps"]]
    setup_python_index = trusted_step_names.index("Set up Python")
    declare_pyqt_index = trusted_step_names.index("Declare isolated PyQt paths")
    create_pyqt_index = trusted_step_names.index("Create isolated PyQt environment")
    install_pyqt_index = trusted_step_names.index(
        "Install constrained PyQt render dependencies"
    )
    smoke_pyqt_index = trusted_step_names.index("Verify isolated PyQt runtime")
    exercise_pyqt_index = trusted_step_names.index(
        "Exercise protected PyQt tab visibility"
    )
    baseline_index = trusted_step_names.index("Enforce protected visual baseline drift")
    assert (
        setup_python_index
        < declare_pyqt_index
        < create_pyqt_index
        < install_pyqt_index
        < smoke_pyqt_index
        < exercise_pyqt_index
        < baseline_index
    )

    assert pr_job["env"]["RATE_VISUAL_BASELINE_CANDIDATE_DIR"] == (
        "${{ github.workspace }}/visual-baseline-candidates"
    )
    assert pr_job["env"]["RATE_VISUAL_BASELINE_SOURCE_COMMIT"] == (
        "${{ github.event.pull_request.head.sha }}"
    )
    assert trusted_pyqt_job["env"]["RATE_VISUAL_BASELINE_CANDIDATE_DIR"] == (
        "${{ github.workspace }}/visual-baseline-candidates"
    )
    assert pr_commands["Install locked web dependencies"] == "npm ci"
    assert trusted_web_commands["Install locked web dependencies"] == "npm ci"
    assert pr_commands["Install Playwright-pinned browser runtimes"] == (
        "npx --no-install playwright install --with-deps chromium firefox webkit"
    )
    assert trusted_web_commands["Install Playwright-pinned Chromium runtime"] == (
        "npx --no-install playwright install --with-deps chromium"
    )
    assert (
        pr_commands["Exercise production Worker lifecycle, layouts, and browser parity"]
        == "npm run test:e2e"
    )
    assert pr_commands["Install bounded PyQt render dependencies"] == (
        'python -m pip install -e ".[gui,dev]" "scipy>=1.10,<1.18" '
        '"pytest-benchmark==5.2.3"'
    )
    assert pr_commands["Exercise PyQt tab visibility at 100 and 150 percent DPI"] == (
        "python -m pytest "
        "tests/rate_of_closure/test_pyqt_variation_rendered_interactions.py "
        "tests/rate_of_closure/test_pyqt_variation_visual_state_rendered.py "
        "tests/rate_of_closure/test_pyqt_putting_sample_inspector_rendered.py "
        "tests/rate_of_closure/test_pyqt_club_camera_rendered.py "
        "tests/rate_of_closure/test_pyqt_flight_sample_inspector_rendered.py "
        "tests/rate_of_closure/test_pyqt_simulation_scrub_rendered.py "
        "tests/rate_of_closure/test_pyqt_plot_point_inspector_rendered.py "
        "tests/rate_of_closure/test_pyqt_visual_layout_persistence_rendered.py "
        "tests/rate_of_closure/test_visualization_accessibility.py "
        "tests/rate_of_closure/test_pyqt_visualization_tab_visibility.py -q -n 0"
    )
    assert (
        trusted_web_commands["Exercise production Worker lifecycle and layouts"]
        == "npm run test:e2e -- --project=chromium-desktop --project=chromium-narrow "
        "--grep-invert=@trusted-isolated"
    )
    assert trusted_web_commands["Build production web bundle once"] == "npm run build"
    assert trusted_web_commands["Audit primary-tab accessibility in isolation"] == (
        "npm run test:e2e -- e2e/visualization-accessibility.spec.ts "
        "--project=chromium-desktop"
    )
    assert trusted_web_commands[
        "Measure protected visualization budgets in isolation"
    ] == (
        "npm run test:e2e -- e2e/visualization-performance.spec.ts "
        "--project=chromium-desktop"
    )
    assert trusted_web_job["env"]["RATE_E2E_PREBUILT"] == "1"
    assert "RATE_PYQT_VENV" not in trusted_pyqt_job["env"]
    assert "PYTEST_DEBUG_TEMPROOT" not in trusted_pyqt_job["env"]
    assert trusted_web_job["timeout-minutes"] == 30
    assert trusted_pyqt_job["timeout-minutes"] == 30
    trusted_steps = {
        str(step.get("name")): step
        for step in trusted_web_job["steps"]
        if "name" in step
    }
    assert trusted_steps["Exercise production Worker lifecycle and layouts"]["env"] == {
        "RATE_E2E_EVIDENCE_PHASE": "functional"
    }
    assert trusted_steps["Audit primary-tab accessibility in isolation"]["env"] == {
        "RATE_E2E_EVIDENCE_PHASE": "accessibility"
    }
    performance_step = trusted_steps[
        "Measure protected visualization budgets in isolation"
    ]
    assert performance_step["env"] == {"RATE_E2E_EVIDENCE_PHASE": "performance"}
    assert trusted_pyqt_commands["Declare isolated PyQt paths"] == (
        'echo "RATE_PYQT_VENV=${RUNNER_TEMP}/rate-pyqt-'
        '${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}" >> "$GITHUB_ENV"\n'
        'echo "PYTEST_DEBUG_TEMPROOT=${RUNNER_TEMP}/rate-pyqt-pytest-'
        '${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}" >> "$GITHUB_ENV"'
    )
    assert trusted_pyqt_commands["Create isolated PyQt environment"] == (
        'python -m venv "$RATE_PYQT_VENV" && mkdir -p "$PYTEST_DEBUG_TEMPROOT"'
    )
    assert trusted_pyqt_commands["Install constrained PyQt render dependencies"] == (
        '"$RATE_PYQT_VENV/bin/python" -m pip install --no-cache-dir '
        '--constraint requirements-rate-pyqt.txt -e ".[gui,dev]"'
    )
    assert trusted_pyqt_commands["Verify isolated PyQt runtime"] == (
        '"$RATE_PYQT_VENV/bin/python" -m pip check && '
        '"$RATE_PYQT_VENV/bin/python" scripts/check_rate_pyqt_environment.py '
        "--constraints requirements-rate-pyqt.txt"
    )
    assert trusted_pyqt_commands["Exercise protected PyQt tab visibility"] == (
        '"$RATE_PYQT_VENV/bin/python" -m pytest '
        "tests/rate_of_closure/test_pyqt_variation_visual_state_rendered.py "
        "tests/rate_of_closure/test_pyqt_putting_sample_inspector_rendered.py "
        "tests/rate_of_closure/test_pyqt_club_camera_rendered.py "
        "tests/rate_of_closure/test_pyqt_flight_sample_inspector_rendered.py "
        "tests/rate_of_closure/test_pyqt_simulation_scrub_rendered.py "
        "tests/rate_of_closure/test_pyqt_plot_point_inspector_rendered.py "
        "tests/rate_of_closure/test_pyqt_visual_layout_persistence_rendered.py "
        "tests/rate_of_closure/test_visualization_accessibility.py "
        "tests/rate_of_closure/test_pyqt_visualization_tab_visibility.py -q -n 0"
    )
    assert pr_commands["Enforce protected visual baseline drift"] == (
        "python -m rate_of_closure.visual_baseline_compare "
        '--candidate-root "$RATE_VISUAL_BASELINE_CANDIDATE_DIR" '
        '--candidate-commit "$RATE_VISUAL_BASELINE_SOURCE_COMMIT"'
    )
    assert trusted_pyqt_commands["Enforce protected visual baseline drift"] == (
        '"$RATE_PYQT_VENV/bin/python" -m rate_of_closure.visual_baseline_compare '
        '--candidate-root "$RATE_VISUAL_BASELINE_CANDIDATE_DIR" '
        '--candidate-commit "$GITHUB_SHA"'
    )


def test_full_pyqt_window_dependency_is_declared_by_shared_gui_extra() -> None:
    """Both rendered lanes install the extra needed by every registered tab."""
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    gui_dependencies = project["project"]["optional-dependencies"]["gui"]

    assert set(FULL_WINDOW_IMPORT_DEPENDENCIES.values()) <= set(gui_dependencies)
    for path in (PR_WORKFLOW_PATH, TRUSTED_WORKFLOW_PATH):
        job_name = (
            "production-worker-e2e"
            if path == PR_WORKFLOW_PATH
            else "push-pyqt-rendered-evidence"
        )
        install_commands = "\n".join(
            _run_steps(_workflow(path)["jobs"][job_name]).values()
        )
        assert ".[gui,dev]" in install_commands


def test_pr_trigger_tracks_every_pyqt_render_authority() -> None:
    workflow = _workflow(PR_WORKFLOW_PATH)
    assert PYQT_AUTHORITY_PATHS <= set(workflow[True]["pull_request"]["paths"])


def test_trusted_trigger_tracks_every_pyqt_render_authority() -> None:
    workflow = _workflow(TRUSTED_WORKFLOW_PATH)
    assert PYQT_AUTHORITY_PATHS <= set(workflow[True]["push"]["paths"])


def test_external_actions_are_immutable_and_artifacts_identify_attempts() -> None:
    for path in (PR_WORKFLOW_PATH, TRUSTED_WORKFLOW_PATH):
        workflow = _workflow(path)
        assert workflow["permissions"] == {"contents": "read"}
        for job in workflow["jobs"].values():
            for step in job["steps"]:
                if "uses" in step:
                    assert FULL_ACTION_SHA.fullmatch(str(step["uses"]))

    pr_job = _workflow(PR_WORKFLOW_PATH)["jobs"]["production-worker-e2e"]
    trusted_jobs = _workflow(TRUSTED_WORKFLOW_PATH)["jobs"]
    evidence_steps = (
        (_named_step(pr_job, "Retain Playwright evidence"), PR_EVIDENCE_PATHS),
        (
            _named_step(
                trusted_jobs["push-production-worker-e2e"],
                "Retain Playwright evidence",
            ),
            PLAYWRIGHT_EVIDENCE_PATHS,
        ),
        (
            _named_step(
                trusted_jobs["push-pyqt-rendered-evidence"],
                "Retain PyQt and baseline evidence",
            ),
            TRUSTED_PYQT_EVIDENCE_PATHS,
        ),
    )
    for artifact, expected_paths in evidence_steps:
        name = artifact["with"]["name"]
        assert "${{ github.run_id }}" in name
        assert "${{ github.run_attempt }}" in name
        assert artifact["with"]["path"] == expected_paths


def test_trusted_pyqt_job_runs_after_web_failure_and_transfers_candidates() -> None:
    jobs = _workflow(TRUSTED_WORKFLOW_PATH)["jobs"]
    web_job = jobs["push-production-worker-e2e"]
    pyqt_job = jobs["push-pyqt-rendered-evidence"]

    assert pyqt_job["needs"] == "push-production-worker-e2e"
    assert pyqt_job["if"] == (
        "${{ always() && needs.push-production-worker-e2e.result != 'cancelled' }}"
    )
    assert "continue-on-error" not in web_job
    assert "continue-on-error" not in pyqt_job

    upload = _named_step(web_job, "Retain React baseline candidates")
    assert upload["if"] == "${{ !cancelled() }}"
    assert upload["with"] == {
        "name": TRUSTED_CANDIDATE_ARTIFACT,
        "path": "visual-baseline-candidates/",
        "if-no-files-found": "error",
        "retention-days": 1,
    }

    download = _named_step(pyqt_job, "Restore React baseline candidates")
    assert download["with"] == {
        "name": TRUSTED_CANDIDATE_ARTIFACT,
        "path": "visual-baseline-candidates",
    }


def test_touched_hosted_runner_guard_uses_only_immutable_actions() -> None:
    loaded = _workflow(RUNNER_GUARD_PATH)
    steps = loaded["jobs"]["reject-hosted-runner-routing"]["steps"]

    action_uses = [str(step["uses"]) for step in steps if "uses" in step]
    assert action_uses
    assert all(FULL_ACTION_SHA.fullmatch(value) for value in action_uses)
