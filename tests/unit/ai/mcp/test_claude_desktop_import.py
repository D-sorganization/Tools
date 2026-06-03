"""Tests for Claude-Desktop config discovery and import."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from src.shared.python.ai.mcp.config_loader import (
    discover_claude_desktop_config,
    merge_external_config,
)

_SAMPLE_CD_CONFIG = {
    "mcpServers": {
        "github": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_xxx"},
        },
        "memory": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-memory"],
        },
    }
}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


# -------------------- discover_claude_desktop_config --------------------


def test_discover_returns_none_when_no_config_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData" / "Roaming"))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert discover_claude_desktop_config() is None


def test_discover_finds_windows_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    appdata = tmp_path / "AppData" / "Roaming"
    cd_path = appdata / "Claude" / "claude_desktop_config.json"
    _write_json(cd_path, _SAMPLE_CD_CONFIG)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("APPDATA", str(appdata))

    found = discover_claude_desktop_config()
    assert found == cd_path


def test_discover_finds_macos_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cd_path = (
        tmp_path
        / "Library"
        / "Application Support"
        / "Claude"
        / "claude_desktop_config.json"
    )
    _write_json(cd_path, _SAMPLE_CD_CONFIG)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("APPDATA", raising=False)

    found = discover_claude_desktop_config()
    assert found == cd_path


def test_discover_finds_linux_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cd_path = tmp_path / ".config" / "Claude" / "claude_desktop_config.json"
    _write_json(cd_path, _SAMPLE_CD_CONFIG)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("APPDATA", raising=False)

    found = discover_claude_desktop_config()
    assert found == cd_path


# -------------------- merge_external_config --------------------


def test_merge_into_empty_target(tmp_path: Path) -> None:
    target = tmp_path / "mcp_servers.json"
    source = tmp_path / "cd.json"
    _write_json(source, _SAMPLE_CD_CONFIG)

    added = merge_external_config(target.resolve(), source.resolve())
    assert added == 2
    data = json.loads(target.read_text())
    assert set(data["mcpServers"]) == {"github", "memory"}


def test_merge_skip_existing_default(tmp_path: Path) -> None:
    target = tmp_path / "mcp_servers.json"
    _write_json(
        target,
        {"mcpServers": {"github": {"command": "existing", "args": []}}},
    )
    source = tmp_path / "cd.json"
    _write_json(source, _SAMPLE_CD_CONFIG)

    added = merge_external_config(target.resolve(), source.resolve())
    assert added == 1  # only memory added; github skipped
    data = json.loads(target.read_text())
    assert data["mcpServers"]["github"]["command"] == "existing"
    assert "memory" in data["mcpServers"]


def test_merge_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "mcp_servers.json"
    source = tmp_path / "cd.json"
    _write_json(source, _SAMPLE_CD_CONFIG)

    first = merge_external_config(target.resolve(), source.resolve())
    second = merge_external_config(target.resolve(), source.resolve())
    assert first == 2
    assert second == 0
    data = json.loads(target.read_text())
    assert len(data["mcpServers"]) == 2


def test_merge_overwrite_strategy(tmp_path: Path) -> None:
    target = tmp_path / "mcp_servers.json"
    _write_json(
        target,
        {"mcpServers": {"github": {"command": "stale", "args": []}}},
    )
    source = tmp_path / "cd.json"
    _write_json(source, _SAMPLE_CD_CONFIG)

    added = merge_external_config(
        target.resolve(), source.resolve(), strategy="overwrite"
    )
    # overwrite counts replacements + new
    assert added == 2
    data = json.loads(target.read_text())
    assert data["mcpServers"]["github"]["command"] == "npx"


def test_merge_prefix_imported_strategy(tmp_path: Path) -> None:
    target = tmp_path / "mcp_servers.json"
    _write_json(
        target,
        {"mcpServers": {"github": {"command": "existing", "args": []}}},
    )
    source = tmp_path / "cd.json"
    _write_json(source, _SAMPLE_CD_CONFIG)

    added = merge_external_config(
        target.resolve(), source.resolve(), strategy="prefix_imported"
    )
    assert added == 2
    data = json.loads(target.read_text())
    assert "github" in data["mcpServers"]  # original preserved
    assert "imported_github" in data["mcpServers"]
    assert "memory" in data["mcpServers"]  # no conflict, no prefix


def test_merge_prefix_imported_strategy_avoids_prefixed_collision(
    tmp_path: Path,
) -> None:
    target = tmp_path / "mcp_servers.json"
    _write_json(
        target,
        {
            "mcpServers": {
                "github": {"command": "existing", "args": []},
                "imported_github": {"command": "also-existing", "args": []},
            }
        },
    )
    source = tmp_path / "cd.json"
    _write_json(
        source, {"mcpServers": {"github": _SAMPLE_CD_CONFIG["mcpServers"]["github"]}}
    )

    added = merge_external_config(
        target.resolve(), source.resolve(), strategy="prefix_imported"
    )

    assert added == 1
    data = json.loads(target.read_text())
    assert "imported_github_2" in data["mcpServers"]
    assert data["mcpServers"]["github"]["command"] == "existing"


def test_merge_skips_invalid_entries_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    target = tmp_path / "mcp_servers.json"
    source = tmp_path / "cd.json"
    _write_json(
        source,
        {
            "mcpServers": {
                "good": {"command": "python", "args": []},
                "bad": {"transport": "websocket"},  # unsupported
            }
        },
    )
    with caplog.at_level(logging.WARNING):
        added = merge_external_config(target.resolve(), source.resolve())
    assert added == 1
    data = json.loads(target.read_text())
    assert set(data["mcpServers"]) == {"good"}
    assert any(
        "bad" in record.message.lower() or "skipping" in record.message.lower()
        for record in caplog.records
    )


def test_merge_requires_absolute_paths_target(tmp_path: Path) -> None:
    rel = Path("relative.json")
    abs_source = tmp_path / "cd.json"
    _write_json(abs_source, _SAMPLE_CD_CONFIG)
    with pytest.raises(ValueError, match="absolute"):
        merge_external_config(rel, abs_source.resolve())


def test_merge_requires_absolute_paths_source(tmp_path: Path) -> None:
    abs_target = tmp_path / "mcp_servers.json"
    rel = Path("relative.json")
    with pytest.raises(ValueError, match="absolute"):
        merge_external_config(abs_target.resolve(), rel)


def test_merge_missing_source_raises(tmp_path: Path) -> None:
    target = tmp_path / "mcp_servers.json"
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        merge_external_config(target.resolve(), missing.resolve())


def test_merge_unknown_strategy_raises(tmp_path: Path) -> None:
    target = tmp_path / "mcp_servers.json"
    source = tmp_path / "cd.json"
    _write_json(source, _SAMPLE_CD_CONFIG)
    with pytest.raises(ValueError, match="strategy"):
        merge_external_config(target.resolve(), source.resolve(), strategy="bogus")


def test_merge_preserves_existing_unrelated_entries(tmp_path: Path) -> None:
    target = tmp_path / "mcp_servers.json"
    _write_json(
        target,
        {"mcpServers": {"local_only": {"command": "python", "args": ["-m", "thing"]}}},
    )
    source = tmp_path / "cd.json"
    _write_json(source, _SAMPLE_CD_CONFIG)
    added = merge_external_config(target.resolve(), source.resolve())
    assert added == 2
    data = json.loads(target.read_text())
    assert "local_only" in data["mcpServers"]
    assert "github" in data["mcpServers"]


def test_merge_source_mcp_servers_array_returns_zero(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    target = tmp_path / "mcp_servers.json"
    source = tmp_path / "cd.json"
    _write_json(source, {"mcpServers": []})

    with caplog.at_level(logging.WARNING):
        added = merge_external_config(target.resolve(), source.resolve())

    assert added == 0
    assert "source 'mcpServers' is not a JSON object" in caplog.text


def test_merge_non_dict_target_servers_are_replaced(tmp_path: Path) -> None:
    target = tmp_path / "mcp_servers.json"
    source = tmp_path / "cd.json"
    _write_json(target, {"mcpServers": ["stale"]})
    _write_json(
        source, {"mcpServers": {"memory": _SAMPLE_CD_CONFIG["mcpServers"]["memory"]}}
    )

    added = merge_external_config(target.resolve(), source.resolve())

    assert added == 1
    data = json.loads(target.read_text())
    assert set(data["mcpServers"]) == {"memory"}
