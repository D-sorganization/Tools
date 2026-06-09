from pathlib import Path

from scripts.check_workflow_pinning import check, scan_workflow


def _write_workflow(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_allows_first_party_action_tag_refs(tmp_path: Path) -> None:
    workflow = _write_workflow(
        tmp_path / "workflow.yml",
        "jobs:\n"
        "  test:\n"
        "    steps:\n"
        "      - uses: actions/checkout@v6\n"
        "      - uses: github/codeql-action/init@v4\n",
    )

    assert scan_workflow(workflow) == []


def test_rejects_mutable_third_party_action_refs(tmp_path: Path) -> None:
    workflow = _write_workflow(
        tmp_path / "workflow.yml",
        "jobs:\n  test:\n    steps:\n      - uses: owner/action@v1\n",
    )

    violations = scan_workflow(workflow)

    assert violations[0].kind == "mutable-action"
    assert violations[0].value == "owner/action@v1"


def test_allows_full_sha_action_refs_and_local_actions(tmp_path: Path) -> None:
    workflow = _write_workflow(
        tmp_path / "workflow.yml",
        "\n".join(
            [
                "jobs:",
                "  test:",
                "    steps:",
                "      - uses: actions/checkout@"
                "0000000000000000000000000000000000000000",
                "      - uses: ./.github/actions/local",
                "      - uses: docker://alpine:3.20",
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


def test_check_scans_selected_workflows_without_baseline(tmp_path: Path) -> None:
    first_party = _write_workflow(
        tmp_path / "first-party.yml",
        "jobs:\n  test:\n    steps:\n      - uses: actions/checkout@v6\n",
    )
    third_party = _write_workflow(
        tmp_path / "third-party.yml",
        "jobs:\n  test:\n    steps:\n      - uses: owner/action@v1\n",
    )

    violations = check([first_party, third_party])

    assert [violation.value for violation in violations] == ["owner/action@v1"]
