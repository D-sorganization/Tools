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
        # UpstreamDrift moved its consumed packages under src/shared/python, so
        # this is the narrow root that carries them. Deliberately NOT bare `src`
        # -- see the assertions in the test below.
        "README.md",
        "build_hooks.py",
        "launch_golf_suite.py",
        "launch_upstream_drift.py",
        "pyproject.toml",
        "scripts",
        "src/__init__.py",
        "src/bunkershot3d",
        "src/engines/__init__.py",
        "src/engines/pendulum_models/python/__init__.py",
        "src/engines/pendulum_models/python/double_pendulum_model",
        "src/shared",
        "tests/conftest.py",
        "tests/shared_contracts",
        "tests/support",
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


def test_upstream_scope_includes_every_imported_contract_package_root() -> None:
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
    # The shared provider gateway imports UpstreamDrift's simulation-backend
    # package, whose wrench contract resolves bunkershot3d.postproc. Sparse
    # checkout must preserve all three import roots without pulling all of `src`.
    assert {
        "src/shared",
        "src/__init__.py",
        "src/bunkershot3d",
        "src/engines/__init__.py",
        "src/engines/pendulum_models/python/__init__.py",
        "src/engines/pendulum_models/python/double_pendulum_model",
    } <= scope


def test_upstream_install_uses_current_tools_without_repackaging_pinned_snapshot() -> (
    None
):
    workflow = _workflow()
    downstreams = workflow["jobs"]["downstream-consumer-contracts"]["strategy"][
        "matrix"
    ]["downstream"]
    upstream = next(
        downstream
        for downstream in downstreams
        if downstream["repo"] == "D-sorganization/UpstreamDrift"
    )

    assert upstream["install"] == (
        'CI= SKIP_UI_BUILD=1 pip install -e ".[dev,gui-test]"'
    )


def test_upstream_consumer_contracts_run_serially() -> None:
    workflow = _workflow()
    downstreams = workflow["jobs"]["downstream-consumer-contracts"]["strategy"][
        "matrix"
    ]["downstream"]
    upstream = next(
        downstream
        for downstream in downstreams
        if downstream["repo"] == "D-sorganization/UpstreamDrift"
    )

    assert " -n 0 " in upstream["test_command"]


def test_downstream_checkout_keeps_sparse_checkout_authoritative() -> None:
    workflow = _workflow()
    steps = workflow["jobs"]["downstream-consumer-contracts"]["steps"]
    checkout = next(
        step
        for step in steps
        if step.get("name") == "Checkout ${{ matrix.downstream.repo }}"
    )

    assert checkout["with"]["sparse-checkout-cone-mode"] == (
        "${{ matrix.downstream.sparse_cone }}"
    )
    assert not checkout["with"].get("filter")


def test_sparse_mode_preserves_each_downstream_packaging_contract() -> None:
    workflow = _workflow()
    downstreams = workflow["jobs"]["downstream-consumer-contracts"]["strategy"][
        "matrix"
    ]["downstream"]
    by_repo = {item["repo"]: item for item in downstreams}

    assert by_repo["D-sorganization/Gasification_Model"]["sparse_cone"] is True
    upstream = by_repo["D-sorganization/UpstreamDrift"]
    assert upstream["sparse_cone"] is False
    assert "README.md" in upstream["sparse_checkout"].splitlines()
    assert "build_hooks.py" in upstream["sparse_checkout"].splitlines()
    assert "pyproject.toml" in upstream["sparse_checkout"].splitlines()


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


def test_missing_downstream_suite_fails_instead_of_skipping() -> None:
    """Tools #4920: no consumer suite means a failed lane, never a green skip."""
    workflow = _workflow()
    job = workflow["jobs"]["downstream-consumer-contracts"]
    steps = job["steps"]
    checkout = next(
        step
        for step in steps
        if step.get("name") == "Checkout ${{ matrix.downstream.repo }}"
    )
    assert "continue-on-error" not in checkout
    require = next(
        step
        for step in steps
        if step.get("name") == "Require the downstream consumer-contract suite"
    )
    assert "exit 1" in require["run"]
    assert "tests/shared_contracts" in require["run"]
    names = {step.get("name") for step in steps}
    assert "Summary (suite missing)" not in names
    assert "Summary (skipped)" not in names


def test_downstream_checkout_authorizes_consumer_with_least_privilege() -> None:
    """Tools #5305: downstream consumer checkout must use least-privilege token."""
    workflow = _workflow()
    job = workflow["jobs"]["downstream-consumer-contracts"]
    downstreams = job["strategy"]["matrix"]["downstream"]
    for item in downstreams:
        assert item.get("name") == item["repo"].split("/")[-1]

    steps = job["steps"]
    creds_step = next(
        step for step in steps if step.get("name") == "Check Jules app credentials"
    )
    assert creds_step["id"] == "app-creds"
    assert creds_step["env"] == {
        "JULES_APP_ID": "${{ secrets.JULES_APP_ID }}",
        "JULES_APP_PRIVATE_KEY": "${{ secrets.JULES_APP_PRIVATE_KEY }}",
    }
    assert "configured=true" in creds_step["run"]

    token_step = next(
        step for step in steps if step.get("name") == "Generate consumer checkout token"
    )
    assert token_step["id"] == "consumer_token"
    assert token_step["if"] == "steps.app-creds.outputs.configured == 'true'"
    assert token_step["uses"] == "actions/create-github-app-token@v1"
    assert token_step["with"]["app-id"] == "${{ secrets.JULES_APP_ID }}"
    assert token_step["with"]["private-key"] == "${{ secrets.JULES_APP_PRIVATE_KEY }}"
    assert token_step["with"]["owner"] == "${{ github.repository_owner }}"
    assert token_step["with"]["repositories"] == "${{ matrix.downstream.name }}"

    checkout_step = next(
        step
        for step in steps
        if step.get("name") == "Checkout ${{ matrix.downstream.repo }}"
    )
    assert checkout_step["with"]["token"] == (
        "${{ steps.app-creds.outputs.configured == 'true' && "
        "steps.consumer_token.outputs.token || "
        "secrets.RUNNER_CHECK_TOKEN || "
        "github.token }}"
    )
    assert "continue-on-error" not in checkout_step


def test_gasification_test_command_disables_nested_xvfb() -> None:
    workflow = _workflow()
    downstreams = workflow["jobs"]["downstream-consumer-contracts"]["strategy"][
        "matrix"
    ]["downstream"]
    gasification = next(
        item
        for item in downstreams
        if item["repo"] == "D-sorganization/Gasification_Model"
    )
    cmd = gasification["test_command"]
    assert "-p no:xvfb" in cmd
    assert cmd.startswith("xvfb-run")
