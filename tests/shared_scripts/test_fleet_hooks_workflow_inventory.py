from argparse import Namespace
from pathlib import Path

from shared_scripts import fleet_hooks


def _workflow(root: Path, body: str) -> None:
    directory = root / ".github" / "workflows"
    directory.mkdir(parents=True)
    (directory / "qualification.yml").write_text(body, encoding="utf-8")


def test_workflow_inventory_allows_policy_routed_hosted_fallback(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    _workflow(
        tmp_path,
        "jobs:\n"
        "  pick-runner:\n"
        "    steps:\n"
        "      - run: echo 'runner=ubuntu-latest' >> $GITHUB_OUTPUT\n",
    )
    monkeypatch.setattr(fleet_hooks, "ROOT", tmp_path)

    assert fleet_hooks.check_workflow_inventory(Namespace(warn_only=False)) == 0


def test_workflow_inventory_rejects_deprecated_fixed_hardware_label(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    _workflow(tmp_path, "jobs:\n  test:\n    runs-on: d-sorg-fleet-14core\n")
    monkeypatch.setattr(fleet_hooks, "ROOT", tmp_path)

    assert fleet_hooks.check_workflow_inventory(Namespace(warn_only=False)) == 1
