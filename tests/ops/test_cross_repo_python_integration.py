from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = (
    REPO_ROOT / ".github" / "workflows" / "cross-repo-python-integration.yml"
)
REQUIRED_SPARSE_PATHS = {
    "D-sorganization/Gasification_Model": {
        "src",
        "tests/shared_contracts",
    },
    "D-sorganization/UpstreamDrift": {
        "chat",
        "contracts.py",
        "python/src/utils",
        "shared",
        "sidekick",
        "src/shared/python",
        "tests/shared_contracts",
        "tests/support",
        "vendor/ud-tools",
    },
}


def _workflow() -> dict[str, object]:
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


def test_downstream_checkout_is_shallow_and_uses_matrix_sparse_scope() -> None:
    workflow = _workflow()
    steps = workflow["jobs"]["downstream-consumer-contracts"]["steps"]
    checkout = next(
        step
        for step in steps
        if step.get("name") == "Checkout ${{ matrix.downstream.repo }}"
    )

    assert checkout["with"]["fetch-depth"] == 1
    assert (
        checkout["with"]["sparse-checkout"]
        == "${{ matrix.downstream.sparse_checkout }}"
    )


def test_each_downstream_declares_its_required_sparse_scope() -> None:
    workflow = _workflow()
    downstreams = workflow["jobs"]["downstream-consumer-contracts"]["strategy"][
        "matrix"
    ]["downstream"]

    actual = {
        downstream["repo"]: set(downstream["sparse_checkout"].splitlines())
        for downstream in downstreams
    }

    assert actual == REQUIRED_SPARSE_PATHS
    upstream_scope = actual["D-sorganization/UpstreamDrift"]
    assert "src" not in upstream_scope
    assert "ui" not in upstream_scope


def test_upstream_scope_includes_every_release_build_package_root() -> None:
    workflow = _workflow()
    downstreams = workflow["jobs"]["downstream-consumer-contracts"]["strategy"][
        "matrix"
    ]["downstream"]
    upstream = next(
        downstream
        for downstream in downstreams
        if downstream["repo"] == "D-sorganization/UpstreamDrift"
    )

    scope = set(upstream["sparse_checkout"].splitlines())
    assert {
        "chat",
        "contracts.py",
        "python/src/utils",
        "shared",
        "sidekick",
        "vendor/ud-tools",
    } <= scope


def test_upstream_initializes_the_pinned_tools_submodule_before_install() -> None:
    workflow = _workflow()
    steps = workflow["jobs"]["downstream-consumer-contracts"]["steps"]
    initialize_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Initialize pinned Tools submodule"
    )
    initialize = steps[initialize_index]
    install_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Install downstream dependencies"
    )

    assert initialize_index < install_index
    assert initialize["working-directory"] == "${{ matrix.downstream.path }}"
    expected_if = (
        "steps.checkout_downstream.outcome == 'success' && "
        "matrix.downstream.repo == 'D-sorganization/UpstreamDrift'"
    )
    assert initialize["if"] == expected_if
    assert "git submodule update --init --depth 1 vendor/ud-tools" in initialize["run"]


def test_downstream_checkout_keeps_sparse_checkout_authoritative() -> None:
    workflow = _workflow()
    steps = workflow["jobs"]["downstream-consumer-contracts"]["steps"]
    checkout = next(
        step
        for step in steps
        if step.get("name") == "Checkout ${{ matrix.downstream.repo }}"
    )

    assert checkout["with"]["sparse-checkout-cone-mode"] is True
    assert not checkout["with"].get("filter")


def test_cross_repo_job_uses_persistent_python_toolcache_and_cold_cache_budget() -> (
    None
):
    workflow = _workflow()
    job = workflow["jobs"]["downstream-consumer-contracts"]
    setup_step = next(
        step for step in job["steps"] if step.get("name") == "Set up Python"
    )

    assert int(job["timeout-minutes"]) >= 60
    cache_step = next(
        step
        for step in job["steps"]
        if step.get("name") == "Select persistent Python tool cache"
    )
    assert "AGENT_TOOLSDIRECTORY=$RUNNER_TOOL_CACHE" in cache_step["run"]
    assert "runner.temp" not in cache_step["run"]
    assert setup_step["with"] == {
        "python-version": "3.11",
        "cache": "pip",
    }
    assert "${{ runner.temp }}/_tool_cache" not in setup_step.get("env", {}).values()
