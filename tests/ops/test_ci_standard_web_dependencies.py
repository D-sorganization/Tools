from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_STANDARD = REPO_ROOT / ".github" / "workflows" / "ci-standard.yml"
WORKFLOW_LINT = REPO_ROOT / ".github" / "workflows" / "workflow-lint.yml"
PYTHON_TOOLCACHE_GUARD = REPO_ROOT / ".github" / "scripts" / "clean-python-toolcache.sh"
PYTHON_TOOLCACHE_RESTORE = (
    REPO_ROOT / ".github" / "scripts" / "restore-python-toolcache.sh"
)


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


def test_ci_standard_installs_standalone_build_and_runtime_dependencies() -> None:
    """Standalone source and wheel tests must not depend on runner residue."""
    import yaml

    workflow = yaml.safe_load(CI_STANDARD.read_text(encoding="utf-8"))

    for job_name in ("quality-gate", "tests"):
        install_step = next(
            step
            for step in workflow["jobs"][job_name]["steps"]
            if step.get("name") == "Install Dependencies"
        )
        install_commands = install_step["run"]

        assert (
            "python -m pip install --upgrade --force-reinstall --no-cache-dir "
            '"build>=1.2.2" "platformdirs>=4.2.0"'
        ) in install_commands
        assert (
            'python -c "import build, platformdirs; '
            "assert build.__version__; assert platformdirs.__version__"
        ) in install_commands


def test_ci_standard_uses_bounded_checkout_history() -> None:
    """PR gates need the merge parents, not every branch, tag, and packfile."""
    import yaml

    workflow = yaml.safe_load(CI_STANDARD.read_text(encoding="utf-8"))

    for job_name in ("quality-gate", "tests"):
        checkout_step = next(
            step
            for step in workflow["jobs"][job_name]["steps"]
            if str(step.get("uses", "")).startswith("actions/checkout@")
        )

        assert checkout_step["with"]["fetch-depth"] == 2


def test_ci_standard_uses_persistent_python_toolcache_and_cold_cache_budgets() -> None:
    """Cold setup downloads must not consume the entire protected-job budget."""
    import yaml

    workflow = yaml.safe_load(CI_STANDARD.read_text(encoding="utf-8"))
    minimum_timeouts = {
        "quality-gate": 60,
        "tests": 90,
    }

    for job_name, minimum_timeout in minimum_timeouts.items():
        job = workflow["jobs"][job_name]
        assert int(job["timeout-minutes"]) >= minimum_timeout
        cache_step = next(
            step
            for step in job["steps"]
            if step.get("name") == "Select persistent Python tool cache"
        )
        assert "AGENT_TOOLSDIRECTORY=$RUNNER_TOOL_CACHE" in cache_step["run"]
        assert "runner.temp" not in cache_step["run"]

        setup_step = next(
            step
            for step in job["steps"]
            if str(step.get("uses", "")).startswith("actions/setup-python@")
        )
        setup_environment = setup_step.get("env", {})
        assert "${{ runner.temp }}/_tool_cache" not in setup_environment.values()


def test_ci_standard_rejects_semantically_broken_cached_python() -> None:
    """An executable empty/stub file must not satisfy the Python cache probe."""
    import yaml

    workflow = yaml.safe_load(CI_STANDARD.read_text(encoding="utf-8"))

    assert PYTHON_TOOLCACHE_GUARD.is_file()
    guard = PYTHON_TOOLCACHE_GUARD.read_text(encoding="utf-8")
    assert 'python_version="$(' in guard
    assert '"$py_bin" -c' in guard
    assert "sys.version_info" in guard
    assert '"$py_bin" -m pip --version' in guard
    assert '[[ "$pip_version" != pip\\ *" from "* ]]' in guard
    assert '"${arch_dir}.complete"' in guard

    restore = PYTHON_TOOLCACHE_RESTORE.read_text(encoding="utf-8")
    assert '"$interpreter" -m pip --version' in restore
    assert '[[ "$pip_version" != pip\\ *" from "* ]]' in restore

    expected_versions = {
        "quality-gate": "3.12",
        "tests": "${{ matrix.python-version }}",
    }
    for job_name, expected_version in expected_versions.items():
        job = workflow["jobs"][job_name]
        clean_step = next(
            step
            for step in job["steps"]
            if step.get("name") == "Force-clean stale Python tool cache (NVMe runners)"
        )
        assert clean_step["run"] == (
            f"bash .github/scripts/clean-python-toolcache.sh '{expected_version}'"
        )
        clean_index = job["steps"].index(clean_step)
        assert job["steps"][clean_index - 1]["name"] == (
            "Restore local Python tool cache"
        )

        setup_index = next(
            index
            for index, step in enumerate(job["steps"])
            if str(step.get("uses", "")).startswith("actions/setup-python@")
        )
        verify_step = job["steps"][setup_index + 1]
        assert verify_step["name"] == "Verify Python runtime"
        assert "python -c" in verify_step["run"]
        assert "sys.version_info" in verify_step["run"]
        assert "python -m pip --version" in verify_step["run"]
        assert 'pip\\ *" from "*' in verify_step["run"]

        venv_step = job["steps"][setup_index + 2]
        assert venv_step["name"] == "Create isolated CI virtual environment"
        assert 'python -m venv "$RUNNER_TEMP/ci-venv"' in venv_step["run"]
        assert 'echo "$RUNNER_TEMP/ci-venv/bin" >> "$GITHUB_PATH"' in venv_step["run"]
        assert (
            'echo "VIRTUAL_ENV=$RUNNER_TEMP/ci-venv" >> "$GITHUB_ENV"'
            in venv_step["run"]
        )


def test_test_matrix_repairs_and_probes_compiled_numerical_stack() -> None:
    import yaml

    workflow = yaml.safe_load(CI_STANDARD.read_text(encoding="utf-8"))
    install_step = next(
        step
        for step in workflow["jobs"]["tests"]["steps"]
        if step.get("name") == "Install Dependencies"
    )

    assert '"numpy>=2.0.1,<2.4.0" "scipy>=1.13.1,<1.18"' in install_step["run"]
    assert "from scipy.signal import butter" in install_step["run"]


def test_quality_gate_invokes_mypy_through_verified_python() -> None:
    import yaml

    workflow = yaml.safe_load(CI_STANDARD.read_text(encoding="utf-8"))
    quality_steps = workflow["jobs"]["quality-gate"]["steps"]
    install_step = next(
        step for step in quality_steps if step.get("name") == "Install Dependencies"
    )
    type_step = next(
        step for step in quality_steps if step.get("name") == "Type Check (Mypy)"
    )

    assert "python -m mypy --version" in install_step["run"]
    assert "python -m ruff --version" in install_step["run"]
    assert "xargs -r -d '\\n' python -m mypy" in type_step["run"]
    assert workflow["jobs"]["quality-gate"]["env"]["PYTHONNOUSERSITE"] == "1"


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
