"""Tests for SkillsManager."""

from __future__ import annotations

from pathlib import Path

import pytest
from chat.skills_manager import SkillsManager


@pytest.fixture
def mock_workspace(tmp_path: Path) -> Path:
    """Create a mock workspace with skills and workflows."""
    agents_dir = tmp_path / ".agents"
    agents_dir.mkdir()

    # Create skills
    skills_dir = agents_dir / "skills"
    skills_dir.mkdir()

    lint_skill = skills_dir / "lint"
    lint_skill.mkdir()
    (lint_skill / "SKILL.md").write_text(
        "# Lint Code\nRun formatting and linting.",
        encoding="utf-8",
    )

    tests_skill = skills_dir / "tests"
    tests_skill.mkdir()
    (tests_skill / "SKILL.md").write_text(
        "# Run Tests\nExecute the test suite.",
        encoding="utf-8",
    )

    # Create workflows
    workflows_dir = agents_dir / "workflows"
    workflows_dir.mkdir()

    (workflows_dir / "sync-all.md").write_text(
        "# Sync All Repos\nPull latest changes.",
        encoding="utf-8",
    )

    return tmp_path


def test_skills_manager_discovery(mock_workspace: Path) -> None:
    """Test that skills and workflows are discovered correctly."""
    manager = SkillsManager(mock_workspace)

    assert len(manager.skills) == 2
    assert "lint" in manager.skills
    assert manager.skills["lint"].name == "Lint Code"
    assert manager.skills["lint"].description == "Run formatting and linting."

    assert len(manager.workflows) == 1
    assert "sync-all" in manager.workflows
    assert manager.workflows["sync-all"].name == "Sync All Repos"


def test_skills_manager_tool_schemas(mock_workspace: Path) -> None:
    """Test that valid function schemas are returned."""
    manager = SkillsManager(mock_workspace)
    schemas = manager.get_tool_schemas()

    assert len(schemas) == 3
    names = [s["function"]["name"] for s in schemas]
    assert "execute_lint" in names
    assert "execute_sync_all" in names


def test_skills_manager_get_skill(mock_workspace: Path) -> None:
    """Test retrieval of specific skills/workflows."""
    manager = SkillsManager(mock_workspace)

    skill = manager.get_skill_or_workflow("lint")
    assert skill is not None
    assert not skill.is_workflow

    workflow = manager.get_skill_or_workflow("sync-all")
    assert workflow is not None
    assert workflow.is_workflow

    assert manager.get_skill_or_workflow("missing") is None
