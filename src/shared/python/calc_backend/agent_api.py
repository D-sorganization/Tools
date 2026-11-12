"""Fleet Agent API v1 helpers for calculator MCP (issues #6021, #6022)."""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any

from pydantic import BaseModel, ValidationError

CONTRACT_VERSION = "fleet-agent-api/v1"
MAX_OUTPUT_BYTES = 32 * 1024
MAX_CALL_SECONDS = 10
DEFAULT_CALCULATION_STATUS = "verified-unapproved"


def tools_version() -> str:
    try:
        from importlib.metadata import version

        return version("ud-tools")
    except Exception:
        return "0.0.0"


def git_sha() -> str:
    env = os.environ.get("TOOLS_GIT_SHA")
    if env:
        return env
    try:
        result = subprocess.run(  # nosec B603 - fixed argv
            ["git", "rev-parse", "HEAD"],
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


def calculation_status(formula_ids: tuple[str, ...]) -> str:
    """Conservative default until design-manual links are complete per calculator."""
    return DEFAULT_CALCULATION_STATUS if not formula_ids else "registered"


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
