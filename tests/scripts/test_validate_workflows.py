from pathlib import Path

from scripts import validate_workflows


def test_validate_workflow_text_fallback_accepts_jobs_mapping(
    tmp_path: Path, monkeypatch
) -> None:
    workflow = tmp_path / "workflow.yml"
    workflow.write_text(
        "name: Test\njobs:\n  check:\n    steps: []\n", encoding="utf-8"
    )
    monkeypatch.setattr(validate_workflows, "yaml", None)

    assert validate_workflows.validate_workflow(workflow) == []


def test_validate_workflow_text_fallback_rejects_missing_jobs(
    tmp_path: Path, monkeypatch
) -> None:
    workflow = tmp_path / "workflow.yml"
    workflow.write_text("name: Test\non: push\n", encoding="utf-8")
    monkeypatch.setattr(validate_workflows, "yaml", None)

    assert validate_workflows.validate_workflow(workflow) == [
        f"{workflow}: missing top-level 'jobs'"
    ]


def test_validate_workflow_rejects_sudo_shellcheck_install(tmp_path: Path) -> None:
    workflow = tmp_path / "workflow-lint.yml"
    workflow.write_text(
        "name: Workflow Lint\n"
        "on: push\n"
        "jobs:\n"
        "  lint:\n"
        "    runs-on: d-sorg-fleet\n"
        "    steps:\n"
        "      - name: Install shellcheck\n"
        "        run: sudo apt-get -o DPkg::Lock::Timeout=300 install -y shellcheck\n",
        encoding="utf-8",
    )

    assert validate_workflows.validate_workflow(workflow) == [
        f"{workflow}: install shellcheck only when passwordless sudo is "
        "available, or run actionlint without shellcheck"
    ]


def test_validate_workflow_rejects_sudo_actionlint_install(tmp_path: Path) -> None:
    workflow = tmp_path / "workflow.yml"
    workflow.write_text(
        "\n".join(
            [
                "name: Test",
                "jobs:",
                "  lint:",
                "    steps:",
                "      - run: sudo mv actionlint /usr/local/bin/actionlint",
            ]
        ),
        encoding="utf-8",
    )

    expected = (
        f"{workflow}: install actionlint into a runner-local directory, "
        "not /usr/local/bin with sudo"
    )
    assert validate_workflows.validate_workflow(workflow) == [expected]
