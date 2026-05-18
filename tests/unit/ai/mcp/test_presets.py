"""Tests for the MCP preset server catalogue.

Each preset must:

- Have stable, unique, snake_case ``id``.
- Round-trip through ``parse_mcp_servers`` without warnings (after env vars
  are filled in by the user).
- Declare all required env vars used by its ``config.env`` template values.
"""

from __future__ import annotations

import logging
import re

import pytest

from src.shared.python.ai.mcp.config_loader import parse_mcp_servers
from src.shared.python.ai.mcp.contracts import McpServerConfig, McpTransport
from src.shared.python.ai.mcp.presets import (
    MCP_SERVER_PRESETS,
    McpServerPreset,
    apply_preset,
    list_preset_ids,
)

_SNAKE_CASE = re.compile(r"^[a-z][a-z0-9_]*$")


def test_preset_catalogue_non_empty() -> None:
    assert len(MCP_SERVER_PRESETS) >= 10


def test_expected_preset_ids_present() -> None:
    expected = {
        "github",
        "linear",
        "notion",
        "slack",
        "obsidian",
        "filesystem",
        "memory",
        "sequential_thinking",
        "brave_search",
        "time",
        "notebooklm",
    }
    assert expected.issubset(set(MCP_SERVER_PRESETS))


def test_preset_ids_are_unique() -> None:
    ids = [preset.id for preset in MCP_SERVER_PRESETS.values()]
    assert len(ids) == len(set(ids))
    # The dict key must equal the preset's declared id.
    for key, preset in MCP_SERVER_PRESETS.items():
        assert key == preset.id


@pytest.mark.parametrize("preset_id", sorted(MCP_SERVER_PRESETS))
def test_preset_id_is_snake_case(preset_id: str) -> None:
    assert _SNAKE_CASE.match(preset_id), preset_id


@pytest.mark.parametrize("preset_id", sorted(MCP_SERVER_PRESETS))
def test_preset_has_required_metadata(preset_id: str) -> None:
    preset = MCP_SERVER_PRESETS[preset_id]
    assert isinstance(preset, McpServerPreset)
    assert preset.display_name
    assert preset.description
    assert preset.category
    assert isinstance(preset.config, McpServerConfig)
    assert isinstance(preset.required_env, tuple)
    # docs_url is optional but must be a string if provided.
    if preset.docs_url is not None:
        assert preset.docs_url.startswith(("http://", "https://"))


@pytest.mark.parametrize("preset_id", sorted(MCP_SERVER_PRESETS))
def test_preset_config_round_trips_without_warning(
    preset_id: str, caplog: pytest.LogCaptureFixture
) -> None:
    preset = MCP_SERVER_PRESETS[preset_id]
    raw = {
        "mcpServers": {
            preset.id: {
                "transport": preset.config.transport.value,
                "command": preset.config.command,
                "args": list(preset.config.args),
                "env": dict(preset.config.env),
                "url": preset.config.url,
                "timeoutSeconds": preset.config.timeout_seconds,
            }
        }
    }
    with caplog.at_level(logging.WARNING):
        configs = parse_mcp_servers(raw)
    assert len(configs) == 1, f"{preset_id} failed to parse"
    # No 'skipping' warnings should fire.
    assert not any("skipping" in record.message.lower() for record in caplog.records), [
        r.message for r in caplog.records
    ]


@pytest.mark.parametrize("preset_id", sorted(MCP_SERVER_PRESETS))
def test_preset_required_env_referenced_in_config(preset_id: str) -> None:
    """Every required_env must appear as a ${VAR} placeholder in config.env."""
    preset = MCP_SERVER_PRESETS[preset_id]
    rendered = " ".join(preset.config.env.values())
    for env_name in preset.required_env:
        assert f"${{{env_name}}}" in rendered, (
            f"{preset_id} declares required env {env_name} "
            f"but it does not appear in config.env"
        )


def test_list_preset_ids_returns_sorted() -> None:
    ids = list_preset_ids()
    assert ids == sorted(ids)
    assert set(ids) == set(MCP_SERVER_PRESETS)


def test_apply_preset_unknown_raises() -> None:
    with pytest.raises(KeyError, match="not_a_real_preset"):
        apply_preset("not_a_real_preset")


def test_apply_preset_default_returns_clone() -> None:
    cfg = apply_preset("memory")
    assert isinstance(cfg, McpServerConfig)
    assert cfg.name == "memory"
    # Modifying the returned config must not mutate the catalogue.
    cfg.args.append("--mutated")
    assert "--mutated" not in MCP_SERVER_PRESETS["memory"].config.args


def test_apply_preset_name_override() -> None:
    cfg = apply_preset("memory", name="my_memory")
    assert cfg.name == "my_memory"


def test_apply_preset_env_override_resolves_required() -> None:
    cfg = apply_preset(
        "github",
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_real_token"},
    )
    assert cfg.env["GITHUB_PERSONAL_ACCESS_TOKEN"] == "ghp_real_token"


def test_apply_preset_missing_required_env_raises() -> None:
    # GitHub requires GITHUB_PERSONAL_ACCESS_TOKEN.
    with pytest.raises(ValueError, match="GITHUB_PERSONAL_ACCESS_TOKEN"):
        apply_preset("github", validate_env=True)


def test_apply_preset_no_validate_env_allows_placeholder() -> None:
    cfg = apply_preset("github", validate_env=False)
    assert cfg.env["GITHUB_PERSONAL_ACCESS_TOKEN"].startswith("${")


def test_apply_preset_command_override() -> None:
    cfg = apply_preset("memory", command="python")
    assert cfg.command == "python"


def test_apply_preset_args_override() -> None:
    cfg = apply_preset("filesystem", args=["/extra/path"])
    assert cfg.args == ["/extra/path"]


def test_filesystem_preset_has_directory_args() -> None:
    """Filesystem preset must include a directory whitelist in args."""
    preset = MCP_SERVER_PRESETS["filesystem"]
    assert len(preset.config.args) >= 2  # at least: -y pkg + 1 directory


def test_github_preset_transport_stdio() -> None:
    preset = MCP_SERVER_PRESETS["github"]
    assert preset.config.transport is McpTransport.STDIO
    assert "GITHUB_PERSONAL_ACCESS_TOKEN" in preset.required_env


def test_notebooklm_preset_has_no_required_env() -> None:
    preset = MCP_SERVER_PRESETS["notebooklm"]
    assert preset.required_env == ()


def test_memory_preset_has_no_required_env() -> None:
    preset = MCP_SERVER_PRESETS["memory"]
    assert preset.required_env == ()


@pytest.mark.parametrize("preset_id", sorted(MCP_SERVER_PRESETS))
def test_preset_category_is_known(preset_id: str) -> None:
    preset = MCP_SERVER_PRESETS[preset_id]
    assert preset.category in {
        "code",
        "docs",
        "productivity",
        "search",
        "memory",
        "filesystem",
        "thinking",
        "time",
    }
