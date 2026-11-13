"""Fleet Agent API v1 helpers for calculator MCP (issues #6021, #6022)."""

from __future__ import annotations

import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from shared.python.lifecycle import CANONICAL_LIFECYCLE_WORDS

CONTRACT_VERSION = "fleet-agent-api/v1"
MAX_OUTPUT_BYTES = 32 * 1024
MAX_CALL_SECONDS = 10
DEFAULT_CALCULATION_STATUS = "verified-unapproved"
_REGISTRY_PATH = (
    Path(__file__).resolve().parents[4]
    / "manuals"
    / "tools"
    / "calculation-registry.json"
)


@lru_cache(maxsize=1)
def tools_version() -> str:
    try:
        from importlib.metadata import version

        return version("ud-tools")
    except Exception:
        return "0.0.0"


@lru_cache(maxsize=1)
def git_sha() -> str:
    env = os.environ.get("TOOLS_GIT_SHA")
    if env:
        return env
    try:
        result = subprocess.run(  # nosec B603 - fixed argv
            ["git", "rev-parse", "HEAD"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "unknown"


@lru_cache(maxsize=1)
def _calculation_registry_status_index() -> dict[str, str]:
    """Load calculation/equation ID -> status mapping with approval gate enforcement."""
    if not _REGISTRY_PATH.is_file():
        return {}
    try:
        payload = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    index: dict[str, str] = {}
    for entry in payload.get("calculations", []):
        if not isinstance(entry, dict):
            continue
        raw_status = str(entry.get("status") or DEFAULT_CALCULATION_STATUS)
        approval = entry.get("approval")
        approval_state = (
            str(approval.get("state") or "") if isinstance(approval, dict) else ""
        )
        effective = (
            DEFAULT_CALCULATION_STATUS
            if raw_status == "approved" and approval_state != "approved"
            else raw_status
        )
        calc_id = entry.get("calculation_id")
        if isinstance(calc_id, str) and calc_id:
            index[calc_id] = effective
        for equation in entry.get("equations", []):
            if isinstance(equation, dict):
                eq_id = equation.get("equation_id")
                if isinstance(eq_id, str) and eq_id:
                    index[eq_id] = effective
    return index


def prime_agent_api_metadata() -> None:
    """Warm cached provenance metadata at startup so tool handlers avoid I/O."""
    tools_version()
    git_sha()
    _calculation_registry_status_index()


def calculation_status(formula_ids: tuple[str, ...]) -> str:
    """Resolve calculation approval status from calculation-registry.json."""
    if not formula_ids:
        return DEFAULT_CALCULATION_STATUS
    index = _calculation_registry_status_index()
    resolved = [index.get(fid) for fid in formula_ids]
    if any(status is None for status in resolved):
        return DEFAULT_CALCULATION_STATUS
    for status in resolved:
        if status is not None and status != "approved":
            return status
    return "approved"


def validation_problems(exc: ValidationError) -> list[dict[str, Any]]:
    problems: list[dict[str, Any]] = []
    for err in exc.errors():
        loc = err.get("loc", ())
        field = ".".join(str(part) for part in loc) if loc else "inputs"
        problems.append(
            {
                "field": field,
                "problem": err.get("msg", "validation error"),
                "allowed_range": err.get("ctx"),
                "example": None,
            }
        )
    return problems


def serialize_outputs(value: Any) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    if isinstance(value, BaseModel):
        return value.model_dump(), warnings
    if isinstance(value, dict):
        return value, warnings
    return {"result": value}, warnings


def enforce_output_cap(payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")
    if len(encoded) > MAX_OUTPUT_BYTES:
        raise ValueError(
            f"MCP output exceeded {MAX_OUTPUT_BYTES} bytes (got {len(encoded)})"
        )


def build_result(
    *,
    calculator_id: str,
    formula_ids: tuple[str, ...],
    outputs: dict[str, Any],
    warnings: list[str],
    units: dict[str, str],
    status: str,
) -> dict[str, Any]:
    if status not in CANONICAL_LIFECYCLE_WORDS:
        raise ValueError(f"Unsupported lifecycle status: {status!r}")
    body = {
        "status": status,
        "outputs": outputs,
        "units": units,
        "warnings": warnings,
        "provenance": {
            "tools_version": tools_version(),
            "git_sha": git_sha(),
            "formula_ids": list(formula_ids),
            "calculation_status": calculation_status(formula_ids),
            "contract_version": CONTRACT_VERSION,
            "calculator_id": calculator_id,
        },
    }
    enforce_output_cap(body)
    return body


def parse_inputs(model: type[BaseModel], raw: dict[str, Any]) -> BaseModel:
    return model.model_validate(raw)


def main_module() -> str:
    return "shared.python.calc_backend.mcp_server"
