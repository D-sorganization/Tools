"""Tests for declarative GitHub MCP tool descriptors."""

from __future__ import annotations

import pytest

from src.shared.python.ai.integrations.github_mcp.tool_descriptors import (
    GITHUB_MCP_TOOL_DESCRIPTORS,
    GitHubMcpToolDescriptor,
    get_tool_descriptor,
    write_tool_names,
)

_EXPECTED_TOOLS = {
    "list_repos",
    "list_issues",
    "list_prs",
    "get_issue",
    "get_pr_diff",
    "create_issue",
    "add_comment",
    "search_code",
    "search_issues",
    "merge_pr",
}

_EXPECTED_WRITE_TOOLS = {"create_issue", "add_comment", "merge_pr"}


def test_all_expected_tools_present() -> None:
    """Declarative list covers every tool named in the issue scope."""
    names = {tool.name for tool in GITHUB_MCP_TOOL_DESCRIPTORS}
    assert names == _EXPECTED_TOOLS


def test_descriptor_is_immutable_dataclass() -> None:
    """Descriptors are frozen so the declarative list cannot be mutated at runtime."""
    descriptor = GITHUB_MCP_TOOL_DESCRIPTORS[0]
    assert isinstance(descriptor, GitHubMcpToolDescriptor)
    with pytest.raises((AttributeError, TypeError)):
        descriptor.name = "mutated"  # type: ignore[misc]


def test_write_tools_require_confirmation() -> None:
    """Mutating tools must opt into ``requires_confirmation=True``."""
    for tool in GITHUB_MCP_TOOL_DESCRIPTORS:
        if tool.name in _EXPECTED_WRITE_TOOLS:
            assert tool.requires_confirmation is True, (
                f"write tool {tool.name} must require confirmation"
            )
        else:
            assert tool.requires_confirmation is False, (
                f"read tool {tool.name} must not require confirmation"
            )


def test_write_tool_names_helper() -> None:
    """``write_tool_names`` returns exactly the confirmation-gated tools."""
    assert set(write_tool_names()) == _EXPECTED_WRITE_TOOLS


def test_every_descriptor_has_description() -> None:
    """No-op-looking tools are a footgun; require a non-empty description."""
    for tool in GITHUB_MCP_TOOL_DESCRIPTORS:
        assert tool.description.strip(), f"{tool.name} missing description"


def test_get_tool_descriptor_returns_known() -> None:
    descriptor = get_tool_descriptor("list_issues")
    assert descriptor is not None
    assert descriptor.name == "list_issues"
    assert descriptor.requires_confirmation is False


def test_get_tool_descriptor_returns_none_for_unknown() -> None:
    assert get_tool_descriptor("definitely_not_a_real_tool") is None


def test_tool_descriptor_list_has_no_duplicates() -> None:
    names = [tool.name for tool in GITHUB_MCP_TOOL_DESCRIPTORS]
    assert len(names) == len(set(names))
