"""Local CLI and MCP expose the same bounded read-only retrieval service."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_context.cli import main
from agent_context.mcp_server import build_server


def test_cli_search_and_invalid_component(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--root", str(repository), "search", "physical"]) == 0
    assert json.loads(capsys.readouterr().out)["matches"][0]["id"] == "provider"
    assert main(["--root", str(repository), "context", "absent"]) == 2
    assert "Unknown component" in capsys.readouterr().err


def test_cli_review_render_and_check(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prefix = ["--root", str(repository)]
    assert main([*prefix, "check"]) == 2
    assert (
        main(
            [
                *prefix,
                "review",
                "conversion-flow",
                "--rationale",
                "Reviewed provider and consumer against the conversion test",
            ]
        )
        == 0
    )
    assert main([*prefix, "render"]) == 0
    assert main([*prefix, "check"]) == 0
    capsys.readouterr()


class FakeMCP:
    def __init__(self, name: str) -> None:
        self.tools: dict[str, Any] = {}

    def tool(self, **kwargs: Any) -> Any:
        def register(fn: Any) -> Any:
            self.tools[fn.__name__] = fn
            return fn

        return register


def test_mcp_only_exposes_context_reads(repository: Path) -> None:
    server = build_server(repository, factory=FakeMCP)
    assert set(server.tools) == {
        "search_components",
        "get_component_context",
        "context_status",
    }
    assert (
        server.tools["get_component_context"]("provider")["component"]["id"]
        == "provider"
    )
    assert (
        server.tools["search_components"]("physical")["matches"][0]["id"] == "provider"
    )
    assert server.tools["context_status"]()["contract_reviews"][0]["verified"] is False
