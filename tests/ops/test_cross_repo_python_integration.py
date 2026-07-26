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
        "src",
        "tests/shared_contracts",
        "ui",
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

    assert actual.keys() == REQUIRED_SPARSE_PATHS.keys()
    for repo, required_paths in REQUIRED_SPARSE_PATHS.items():
        assert required_paths <= actual[repo]
