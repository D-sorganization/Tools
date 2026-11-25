from pathlib import Path

import yaml


def test_required_rest_shard_provisions_mcp_before_tests() -> None:
    workflow = yaml.safe_load(
        (
            Path(__file__).resolve().parents[2] / ".github/workflows/ci-standard.yml"
        ).read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["tests"]["steps"]
    provision = next(
        i
        for i, step in enumerate(steps)
        if step.get("name") == "Provision MCP SDK for acceptance probes"
    )
    run = next(
        i
        for i, step in enumerate(steps)
        if str(step.get("name", "")).startswith("Run Tests with Coverage")
    )
    step = steps[provision]
    assert "matrix.shard == 'tests-rest'" in step["if"]
    assert "mcp>=1,<2" in step["run"]
    assert "from mcp.server.fastmcp import FastMCP" in step["run"]
    assert "from mcp.client.stdio import stdio_client" in step["run"]
    assert provision < run
