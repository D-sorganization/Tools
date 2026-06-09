from __future__ import annotations

import json
from pathlib import Path

from scripts.check_workflow_pinning import check, scan_workflow


def _write_workflow(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_rejects_mutable_action_refs(tmp_path: Path) -> None:
    workflow = _write_workflow(
        tmp_path / "workflow.yml",
        "jobs:\n  test:\n    steps:\n      - uses: actions/checkout@v6\n",
    )

    violations = scan_workflow(workflow)

    assert violations[0].kind == "mutable-action"
    assert violations[0].value == "actions/checkout@v6"


def test_allows_full_sha_action_refs_and_local_actions(tmp_path: Path) -> None:
    workflow = _write_workflow(
        tmp_path / "workflow.yml",
        "\n".join(
            [
                "jobs:",
                "  test:",
                "    steps:",
                "      - uses: actions/checkout@"
                "0123456789abcdef0123456789abcdef01234567",
                "      - uses: ./.github/actions/local",
            ]
        ),
    )

    assert scan_workflow(workflow) == []


def test_rejects_curl_pipe_installers(tmp_path: Path) -> None:
    workflow = _write_workflow(
        tmp_path / "workflow.yml",
        "jobs:\n"
        "  test:\n"
        "    steps:\n"
        "      - run: curl https://example.invalid/install.sh | sh\n",
    )

    assert scan_workflow(workflow)[0].kind == "curl-pipe"


def test_rejects_unversioned_global_npm_installs(tmp_path: Path) -> None:
    workflow = _write_workflow(
        tmp_path / "workflow.yml",
        "jobs:\n  test:\n    steps:\n      - run: npm install -g @google/jules\n",
    )

    assert scan_workflow(workflow)[0].kind == "unpinned-global-npm"


def test_allows_exact_version_global_npm_installs(tmp_path: Path) -> None:
    workflow = _write_workflow(
        tmp_path / "workflow.yml",
        "jobs:\n  test:\n    steps:\n      - run: npm install -g @google/jules@0.7.1\n",
    )

    assert scan_workflow(workflow) == []


def test_baseline_allows_existing_violations_but_not_new_ones(tmp_path: Path) -> None:
    existing = _write_workflow(
        tmp_path / "existing.yml",
        "jobs:\n  test:\n    steps:\n      - uses: actions/checkout@v6\n",
    )
    new = _write_workflow(
        tmp_path / "new.yml",
        "jobs:\n  test:\n    steps:\n      - uses: actions/setup-python@v6\n",
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "allowlisted_violations": [
                    f"{existing.as_posix()}|mutable-action|actions/checkout@v6"
                ]
            }
        ),
        encoding="utf-8",
    )

    violations = check([existing, new], baseline)

    assert [violation.value for violation in violations] == ["actions/setup-python@v6"]
