"""Tests for Tools #6021 MCP calculator server hardening, bounds, and provenance."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import subprocess
import time
import tomllib
from pathlib import Path
from typing import Any

import pytest

from shared.python import lifecycle
from shared.python.calc_backend import agent_api, mcp_server

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REGISTRY_PATH = _REPO_ROOT / "manuals" / "tools" / "calculation-registry.json"


def test_calculation_status_reads_registry_and_enforces_approval_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """calculation_status resolves calculation/equation IDs dynamically."""
    default_status = agent_api.DEFAULT_CALCULATION_STATUS
    assert agent_api.calculation_status(()) == default_status
    assert agent_api.calculation_status(("UNKNOWN-ID",)) == default_status
    assert (
        agent_api.calculation_status(("TOOLS-DPLANE-GEOMETRY",)) == "model-conditioned"
    )
    assert agent_api.calculation_status(("TOOLS-DPLANE-NORMAL",)) == "model-conditioned"

    # When a registry entry claims status='approved' without approval.state=='approved',
    # never surface 'approved' provenance.
    synthetic = tmp_path / "calculation-registry.json"
    synthetic.write_text(
        json.dumps(
            {
                "calculations": [
                    {
                        "calculation_id": "UNAPPROVED-APPROVED-CLAIM",
                        "status": "approved",
                        "approval": {"state": "blocked-pending-human-approval"},
                        "equations": [{"equation_id": "UNAPPROVED-EQ-1"}],
                    },
                    {
                        "calculation_id": "ACTUALLY-APPROVED",
                        "status": "approved",
                        "approval": {"state": "approved"},
                        "equations": [{"equation_id": "APPROVED-EQ-1"}],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(agent_api, "_REGISTRY_PATH", synthetic)
    agent_api._calculation_registry_status_index.cache_clear()
    try:
        assert (
            agent_api.calculation_status(("UNAPPROVED-APPROVED-CLAIM",))
            == agent_api.DEFAULT_CALCULATION_STATUS
        )
        assert (
            agent_api.calculation_status(("UNAPPROVED-EQ-1",))
            == agent_api.DEFAULT_CALCULATION_STATUS
        )
        assert agent_api.calculation_status(("ACTUALLY-APPROVED",)) == "approved"
        assert agent_api.calculation_status(("APPROVED-EQ-1",)) == "approved"
    finally:
        agent_api._calculation_registry_status_index.cache_clear()


def test_build_result_enforces_canonical_lifecycle_words() -> None:
    """build_result rejects non-canonical status words."""
    with pytest.raises(ValueError, match="lifecycle"):
        agent_api.build_result(
            calculator_id="unit-converter",
            formula_ids=(),
            outputs={"value": 1.0},
            warnings=[],
            units={"value": "kg"},
            status="COMPLETED",
        )

    ok = agent_api.build_result(
        calculator_id="unit-converter",
        formula_ids=(),
        outputs={"value": 1.0},
        warnings=[],
        units={"value": "kg"},
        status=lifecycle.LifecycleStatus.SUCCEEDED.name,
    )
    assert ok["status"] in lifecycle.CANONICAL_LIFECYCLE_WORDS


def test_fastmcp_tools_expose_readonly_and_idempotent_hints() -> None:
    """All four calculator MCP tools expose readOnlyHint and idempotentHint."""
    server = mcp_server._build_fastmcp()
    tools = asyncio.run(server.list_tools())
    by_name = {tool.name: tool for tool in tools}
    assert set(by_name) == {
        "list_calculators",
        "describe_calculator",
        "run_calculation",
        "convert_units",
    }
    for name, tool in by_name.items():
        assert tool.annotations is not None, name
        assert tool.annotations.readOnlyHint is True, name
        assert tool.annotations.idempotentHint is True, name


def test_run_calculator_enforces_timeout_and_output_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_run_calculator returns FAILED envelopes on timeout and output overflow."""
    mcp_server._ensure_registry()

    sample_id = next(iter(mcp_server.CALCULATOR_REGISTRY))
    orig_entry = mcp_server.CALCULATOR_REGISTRY[sample_id]
    example = dict(mcp_server.registry_describe(sample_id)["worked_example"]["request"])

    class _SlowSpec:
        def __init__(self, wrapped: Any) -> None:
            self.request_model = wrapped.request_model
            self.response_model = wrapped.response_model
            self.formula_ids = wrapped.formula_ids
            self.units = wrapped.units

        def fn(self, _req: Any) -> dict[str, Any]:
            time.sleep(0.25)
            return {}

    monkeypatch.setitem(
        mcp_server.CALCULATOR_REGISTRY,
        sample_id,
        dataclasses.replace(orig_entry, spec=_SlowSpec(orig_entry.spec)),
    )
    timed_out = mcp_server._run_calculator(sample_id, example, timeout_s=0.05)
    assert timed_out["status"] == lifecycle.LifecycleStatus.FAILED.name
    assert timed_out["errors"][0]["field"] == "timeout"

    class _HugeSpec:
        def __init__(self, wrapped: Any) -> None:
            self.request_model = wrapped.request_model
            self.response_model = wrapped.response_model
            self.formula_ids = wrapped.formula_ids
            self.units = wrapped.units

        def fn(self, _req: Any) -> dict[str, Any]:
            return {"blob": "x" * (agent_api.MAX_OUTPUT_BYTES * 2)}

    monkeypatch.setitem(
        mcp_server.CALCULATOR_REGISTRY,
        sample_id,
        dataclasses.replace(orig_entry, spec=_HugeSpec(orig_entry.spec)),
    )
    capped = mcp_server._run_calculator(sample_id, example)
    assert capped["status"] == lifecycle.LifecycleStatus.FAILED.name
    assert capped["errors"][0]["field"] == "outputs"


def test_run_calculator_does_not_invoke_subprocess_after_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once primed at startup, _run_calculator performs zero subprocess calls."""
    mcp_server._ensure_registry()

    def _forbid_subprocess(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("subprocess.run must not be called inside tool execution")

    monkeypatch.setattr(subprocess, "run", _forbid_subprocess)
    sample_id = next(iter(mcp_server.CALCULATOR_REGISTRY))
    example = dict(mcp_server.registry_describe(sample_id)["worked_example"]["request"])
    result = mcp_server._run_calculator(sample_id, example)
    assert result["status"] == lifecycle.LifecycleStatus.SUCCEEDED.name


def test_pyproject_reconciles_mcp_optional_dependencies() -> None:
    """codemap-mcp and calc-mcp share the bounded mcp>=1.2,<2 constraint."""
    pyproject_text = (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    pyproject = tomllib.loads(pyproject_text)
    optional = pyproject["project"]["optional-dependencies"]
    assert "mcp>=1.2,<2" in optional["calc-mcp"]
    assert optional["codemap-mcp"] == ["mcp>=1.2,<2"]
