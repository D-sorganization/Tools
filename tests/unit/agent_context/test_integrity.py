"""Negative evidence tests for races, corrupt metadata and dependency drift."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from test_agent_context_catalog import rewrite

from agent_context.catalog import CatalogError, load_catalog
from agent_context.cli import main
from agent_context.service import ContextService
from agent_context.workspace import git, snapshot


def test_explicit_checkout_ignores_inherited_hook_index(
    repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GIT_INDEX_FILE", str(repository / "unrelated-index"))
    assert "src/provider.py" in git(repository, "ls-files")


def test_catalog_change_after_validation_is_rejected(repository: Path) -> None:
    catalog = load_catalog(repository)
    rewrite(repository, lambda data: data.update(repository="another-project"))
    with pytest.raises(CatalogError, match="changed"):
        snapshot(catalog)


def test_unknown_schema_field_is_not_silently_ignored(repository: Path) -> None:
    rewrite(repository, lambda data: data.update(dependancies=[]))
    with pytest.raises(CatalogError, match="Unknown"):
        load_catalog(repository)


def test_cli_budget_includes_json_serialization(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(["--root", str(repository), "context", "provider", "--max-chars", "4000"])
        == 0
    )
    output = capsys.readouterr().out
    assert len(output) <= 4000
    assert json.loads(output)["component"]["id"] == "provider"


def test_dependency_drift_is_never_source_authority(repository: Path) -> None:
    child = repository / "vendor/shared"
    child.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(child)], check=True)
    (child / "module.py").write_text("# provider\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(child), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(child),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(repository), "add", "vendor/shared"], check=True)
    rewrite(repository, lambda data: data.update(dependencies=["vendor/shared"]))
    service = ContextService(repository)
    assert (
        service.status()["provenance"]["dependencies"]["vendor/shared"]["state"]
        == "verified"
    )
    (child / "module.py").write_text("# dirty provider\n", encoding="utf-8")
    assert service.context("provider")["authority"] == "dependency-unverified"
    provider = child / "src/agent_context"
    provider.mkdir(parents=True)
    (provider / "__init__.py").write_text("# different runtime\n", encoding="utf-8")
    state = service.status()["provenance"]["dependencies"]["vendor/shared"]
    assert state["runtime"] == "different-context-package"


def test_dependency_change_during_snapshot_is_rejected(
    repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_context import workspace

    states = iter([{}, {"vendor/shared": {"state": "unverified"}}])
    monkeypatch.setattr(workspace, "dependency_state", lambda catalog: next(states))
    with pytest.raises(CatalogError, match="changed"):
        snapshot(load_catalog(repository))


def test_corrupt_review_schema_is_a_controlled_error(repository: Path) -> None:
    (repository / "docs/agent_context/reviews.json").write_text(
        '{"version": true, "reviews": {}}', encoding="utf-8"
    )
    with pytest.raises(CatalogError, match="schema"):
        ContextService(repository).status()


def test_unwritable_cache_does_not_block_live_source(
    repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_context import service

    def refused(*args: object) -> None:
        raise PermissionError("read-only checkout cache")

    monkeypatch.setattr(service, "write_json", refused)
    result = ContextService(repository).context("provider")
    assert result["cache_state"] == "unavailable"
    assert result["sources"][0]["path"] == "src/provider.py"
