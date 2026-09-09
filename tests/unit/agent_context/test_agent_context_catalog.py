"""Contracts for source-backed context; no scientific runtime is imported."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agent_context.catalog import CatalogError, load_catalog
from agent_context.workspace import snapshot


def rewrite(root: Path, change: Callable[[dict[str, Any]], None]) -> None:
    path = root / "docs/agent_context/catalog.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_valid_catalog_resolves_public_symbols(repository: Path) -> None:
    catalog = load_catalog(repository)
    assert catalog.components[0].entrypoints[0].symbol == "convert"
    assert catalog.relations[0].consumer == "consumer"


def test_catalog_can_explicitly_mark_unsupported_components(repository: Path) -> None:
    rewrite(repository, lambda data: data["components"][0].update(status="unsupported"))
    assert load_catalog(repository).components[0].status == "unsupported"


@pytest.mark.parametrize(
    "target", ["../outside.py", "/etc/passwd", "C:/secret", "src/../../escape"]
)
def test_catalog_rejects_escape_paths(repository: Path, target: str) -> None:
    rewrite(repository, lambda d: d["components"][0].update(sources=[target]))
    with pytest.raises(CatalogError, match="path"):
        load_catalog(repository)


def test_missing_symbol_cannot_claim_validity(repository: Path) -> None:
    rewrite(
        repository,
        lambda d: d["components"][0]["entrypoints"][0].update(symbol="missing"),
    )
    with pytest.raises(CatalogError, match="missing"):
        load_catalog(repository)


def test_unknown_endpoint_and_duplicate_ids_rejected(repository: Path) -> None:
    rewrite(repository, lambda d: d["relations"][0].update(consumer="unknown"))
    with pytest.raises(CatalogError, match="unknown"):
        load_catalog(repository)
    rewrite(repository, lambda d: d["relations"][0].update(consumer="consumer"))
    rewrite(repository, lambda d: d["components"].append(d["components"][0]))
    with pytest.raises(CatalogError, match="duplicate"):
        load_catalog(repository)


def test_contract_must_explain_interaction(repository: Path) -> None:
    path = repository / "docs/agent_context/boundary.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace("## Data Contract", "## Undocumented"),
        encoding="utf-8",
    )
    with pytest.raises(CatalogError, match="Data Contract"):
        load_catalog(repository)


def test_content_changes_invalidate_snapshot_without_commit(repository: Path) -> None:
    first = snapshot(load_catalog(repository))
    (repository / "src/provider.py").write_text(
        "def convert(value):\n    return value * 2\n", encoding="utf-8"
    )
    second = snapshot(load_catalog(repository))
    assert first["digest"] != second["digest"]
    assert first["commit"] == second["commit"]


def test_directory_membership_tracks_new_deleted_and_renamed_files(
    repository: Path,
) -> None:
    rewrite(repository, lambda d: d["components"][0].update(sources=["src"]))
    first = snapshot(load_catalog(repository))
    extra = repository / "src/new.py"
    extra.write_text("# new\n", encoding="utf-8")
    second = snapshot(load_catalog(repository))
    assert first["digest"] != second["digest"]
    extra.rename(repository / "src/renamed.py")
    third = snapshot(load_catalog(repository))
    assert second["digest"] != third["digest"]
    (repository / "src/renamed.py").unlink()
    assert snapshot(load_catalog(repository))["digest"] == first["digest"]


def test_line_endings_do_not_create_false_staleness(repository: Path) -> None:
    first = snapshot(load_catalog(repository))
    path = repository / "src/provider.py"
    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert snapshot(load_catalog(repository))["digest"] == first["digest"]


def test_external_symlink_is_not_context(
    repository: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    outside = tmp_path_factory.mktemp("outside") / "leak.py"
    outside.write_text("secret = 1\n", encoding="utf-8")
    link = repository / "src/link.py"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("OS does not permit symlink creation")
    rewrite(repository, lambda d: d["components"][0].update(sources=["src/link.py"]))
    with pytest.raises(CatalogError, match="path"):
        load_catalog(repository)
