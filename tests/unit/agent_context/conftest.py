"""Isolated Git checkout shared by the context contract tests."""

import json
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "src/provider.py").write_text(
        "def convert(value: float) -> float:\n    return value\n", encoding="utf-8"
    )
    (tmp_path / "src/consumer.py").write_text("# consumer\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_flow.py").write_text("# contract test\n", encoding="utf-8")
    docs = tmp_path / "docs/agent_context"
    docs.mkdir(parents=True)
    (docs / "boundary.md").write_text(
        "# Conversion Boundary\n\n"
        "## Responsibilities\nProvider converts; consumer orchestrates.\n"
        "## Data Contract\nSI metres; scalar float.\n"
        "## Lifecycle and Failures\nSynchronous; ValueError on invalid input.\n"
        "## Evidence\nSee tests/test_flow.py.\n"
        "## Rationale\nKeep conversion independent of orchestration.\n",
        encoding="utf-8",
    )
    data = {
        "version": 1,
        "repository": "fixture",
        "components": [
            {
                "id": name,
                "title": name.title(),
                "summary": "Convert physical measurements"
                if name == "provider"
                else "Orchestrate conversions",
                "owner": "fixture",
                "status": "implemented",
                "sources": [f"src/{name}.py"],
                "documentation": ["docs/agent_context/boundary.md"],
                "tests": ["tests/test_flow.py"],
                "entrypoints": [{"path": "src/provider.py", "symbol": "convert"}]
                if name == "provider"
                else [],
                "tags": ["conversion"],
            }
            for name in ("provider", "consumer")
        ],
        "relations": [
            {
                "id": "conversion-flow",
                "provider": "provider",
                "consumer": "consumer",
                "kind": "calls",
                "contract": "docs/agent_context/boundary.md",
                "inputs": ["src/provider.py", "src/consumer.py"],
                "tests": ["tests/test_flow.py"],
            }
        ],
        "inventories": [],
        "dependencies": [],
    }
    (docs / "catalog.json").write_text(json.dumps(data), encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    return tmp_path
