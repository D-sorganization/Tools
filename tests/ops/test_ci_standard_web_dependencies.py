from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_STANDARD = REPO_ROOT / ".github" / "workflows" / "ci-standard.yml"
WORKFLOW_LINT = REPO_ROOT / ".github" / "workflows" / "workflow-lint.yml"


def test_ci_standard_installs_fastapi_multipart_parser() -> None:
    workflow = CI_STANDARD.read_text(encoding="utf-8")
    fastapi_install_lines = [
        line.strip()
        for line in workflow.splitlines()
        if line.strip().startswith("python -m pip install fastapi")
    ]

    assert fastapi_install_lines
    assert all("python-multipart" in line.split() for line in fastapi_install_lines)


def test_ci_standard_installs_p1am_runtime_dependencies_without_skips() -> None:
    import yaml

    workflow = yaml.safe_load(CI_STANDARD.read_text(encoding="utf-8"))

    for job_name in ("quality-gate", "tests"):
        install_step = next(
            step
            for step in workflow["jobs"][job_name]["steps"]
            if step.get("name") == "Install Dependencies"
        )

        assert "python -m pip install pymodbus requests sqlmodel" in install_step["run"]


def test_ci_standard_limits_sidekick_runtime_lane_to_runtime_sources() -> None:
    workflow = CI_STANDARD.read_text(encoding="utf-8")

    assert "sidekick_runtime_tests_required=false" in workflow
    assert "sidekick_runtime_tests_required=true" in workflow
    assert "src/shared/python/sidekick/(api|calculators" in workflow
    assert "tests/unit/sidekick" in workflow


def test_ci_standard_serializes_apt_installs_on_shared_runners() -> None:
    workflow = CI_STANDARD.read_text(encoding="utf-8")

    install_steps = workflow.count("Install System Dependencies")

    assert install_steps == 2
    assert workflow.count("flock /tmp/d-sorg-apt-install.lock") == 4
    assert "sudo -n true" in workflow
    assert "sudo -n flock /tmp/d-sorg-apt-install.lock" in workflow
    assert "Passwordless sudo is unavailable" in workflow
    assert "apt-get -o DPkg::Lock::Timeout=300 update --fix-missing" in workflow
    assert "apt-get -o DPkg::Lock::Timeout=300 install -y --fix-missing" in workflow


def test_quality_gate_dependency_install_does_not_use_shared_pip_cache() -> None:
    import yaml

    workflow = yaml.safe_load(CI_STANDARD.read_text(encoding="utf-8"))
    install_step = next(
        step
        for step in workflow["jobs"]["quality-gate"]["steps"]
        if step.get("name") == "Install Dependencies"
    )

    assert install_step["env"]["PIP_NO_CACHE_DIR"] == "1"
    assert install_step["env"]["PIP_CACHE_DIR"] == "${{ runner.temp }}/pip-quality-gate"


def test_workflow_lint_installs_actionlint_without_sudo() -> None:
    workflow = WORKFLOW_LINT.read_text(encoding="utf-8")

    assert 'chmod +x "$ACTIONLINT_BIN/actionlint"' in workflow
    assert 'echo "$ACTIONLINT_BIN" >> "$GITHUB_PATH"' in workflow
    assert "run: actionlint -color" in workflow
    assert "sudo mv actionlint" not in workflow
