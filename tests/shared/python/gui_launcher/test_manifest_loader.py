"""Focused coverage for GUI launcher manifest loading."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest
from gui_launcher.manifest_loader import load_manifest

pytestmark = pytest.mark.unit


def _write_manifest(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_load_manifest_reads_default_manifest() -> None:
    tools = load_manifest()

    assert tools
    assert {tool["tool_name"] for tool in tools} >= {
        "data_processor",
        "financial_calculator",
    }
    assert all(isinstance(tool, dict) for tool in tools)


def test_load_manifest_reads_custom_manifest_and_logs_count(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manifest = _write_manifest(
        tmp_path / "tools.yaml",
        """
tools:
  - tool_name: example
    name: Example Tool
    description: Example description
    category: Testing
    icon: flask
    pyqt6:
      module: example.ui
      class: ExampleWindow
      dependencies:
        - PyQt6
""",
    )

    with caplog.at_level(logging.DEBUG, logger="gui_launcher.manifest_loader"):
        tools = load_manifest(manifest)

    assert tools == [
        {
            "tool_name": "example",
            "name": "Example Tool",
            "description": "Example description",
            "category": "Testing",
            "icon": "flask",
            "pyqt6": {
                "module": "example.ui",
                "class": "ExampleWindow",
                "dependencies": ["PyQt6"],
            },
        }
    ]
    assert f"Loaded 1 tool registrations from {manifest}" in caplog.text


def test_load_manifest_rejects_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="GUI tool manifest not found"):
        load_manifest(missing)


def test_load_manifest_rejects_invalid_yaml(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "invalid.yaml", "tools: [")

    with pytest.raises(ValueError, match="Invalid YAML in tool manifest"):
        load_manifest(manifest)


@pytest.mark.parametrize(
    "content",
    [
        "[]",
        "name: Missing tools",
    ],
)
def test_load_manifest_requires_mapping_with_tools_key(
    tmp_path: Path,
    content: str,
) -> None:
    manifest = _write_manifest(tmp_path / "missing_tools.yaml", content)

    with pytest.raises(ValueError, match="must be a YAML mapping with a 'tools' key"):
        load_manifest(manifest)


@pytest.mark.parametrize(
    ("content", "expected_type"),
    [
        ("tools: example", "str"),
        ("tools:\n  name: Example", "dict"),
    ],
)
def test_load_manifest_requires_tools_sequence(
    tmp_path: Path,
    content: str,
    expected_type: str,
) -> None:
    manifest = _write_manifest(tmp_path / "bad_tools.yaml", content)

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"'tools' in {manifest} must be a YAML sequence, got {expected_type}"
        ),
    ):
        load_manifest(manifest)


def test_load_manifest_allows_empty_tools_sequence(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "empty.yaml", "tools: []")

    assert load_manifest(manifest) == []
