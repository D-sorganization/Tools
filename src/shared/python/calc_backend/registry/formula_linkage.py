"""Resolve Agent API calculator formula_ids from the design-manual registry."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_MANUAL_ROOT = Path(__file__).resolve().parents[5] / "manuals" / "tools"
_REGISTRY_PATH = _MANUAL_ROOT / "calculation-registry.json"
_BINDINGS_PATH = _MANUAL_ROOT / "agent_api_calculator_bindings.json"


class FormulaLinkageError(ValueError):
    """Raised when calculation-registry bindings are missing or inconsistent."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormulaLinkageError(f"{label} unavailable: {path}") from exc
    if not isinstance(payload, dict):
        raise FormulaLinkageError(f"{label} must be a JSON object")
    return payload


def _calculations_index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload.get("calculations")
    if not isinstance(rows, list):
        raise FormulaLinkageError("calculations must be a list")
    index: dict[str, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            raise FormulaLinkageError("calculation entry must be an object")
        calc_id = raw.get("calculation_id")
        if not isinstance(calc_id, str) or not calc_id.strip():
            raise FormulaLinkageError("calculation entry missing calculation_id")
        if calc_id in index:
            raise FormulaLinkageError(f"duplicate calculation_id: {calc_id!r}")
        index[calc_id] = raw
    return index


def _equation_ids_for_calculation(calculation: dict[str, Any]) -> frozenset[str]:
    calc_id = calculation["calculation_id"]
    equations = calculation.get("equations")
    if not isinstance(equations, list):
        raise FormulaLinkageError(f"{calc_id}: equations must be a list")
    ids: set[str] = set()
    for entry in equations:
        if not isinstance(entry, dict):
            raise FormulaLinkageError(f"{calc_id}: invalid equation entry")
        equation_id = entry.get("equation_id")
        if not isinstance(equation_id, str) or not equation_id.strip():
            raise FormulaLinkageError(f"{calc_id}: equation missing equation_id")
        ids.add(equation_id)
    return frozenset(ids)


def _parse_binding_target(
    calculator_id: str,
    raw_target: object,
) -> tuple[str, frozenset[str] | None]:
    if isinstance(raw_target, str) and raw_target.strip():
        return raw_target.strip(), None
    if isinstance(raw_target, dict):
        calculation_id = raw_target.get("calculation_id")
        if not isinstance(calculation_id, str) or not calculation_id.strip():
            raise FormulaLinkageError(
                f"{calculator_id}: binding object requires calculation_id"
            )
        equation_ids = raw_target.get("equation_ids")
        if equation_ids is None:
            return calculation_id.strip(), None
        if not isinstance(equation_ids, list) or not equation_ids:
            raise FormulaLinkageError(
                f"{calculator_id}: equation_ids must be a non-empty list"
            )
        parsed: list[str] = []
        for item in equation_ids:
            if not isinstance(item, str) or not item.strip():
                raise FormulaLinkageError(
                    f"{calculator_id}: equation_ids entries must be non-empty strings"
                )
            parsed.append(item.strip())
        return calculation_id.strip(), frozenset(parsed)
    raise FormulaLinkageError(
        f"{calculator_id}: binding must be a calculation_id string or object"
    )


def _formula_ids_for_binding(
    calculation: dict[str, Any],
    *,
    equation_filter: frozenset[str] | None,
) -> tuple[str, ...]:
    calc_id = calculation["calculation_id"]
    available = _equation_ids_for_calculation(calculation)
    if equation_filter is None:
        return (calc_id, *sorted(available))
    missing = sorted(equation_filter - available)
    if missing:
        raise FormulaLinkageError(
            f"{calc_id}: equation_ids not in registry: {', '.join(missing)}"
        )
    ordered = [calc_id, *sorted(equation_filter)]
    return tuple(dict.fromkeys(ordered))


def _resolve_bindings(
    bindings: dict[str, object],
    calculations: dict[str, dict[str, Any]],
) -> dict[str, tuple[str, ...]]:
    resolved: dict[str, tuple[str, ...]] = {}
    for calculator_id, raw_target in bindings.items():
        if not isinstance(calculator_id, str) or not calculator_id.strip():
            raise FormulaLinkageError("binding key must be a non-empty calculator id")
        if calculator_id in resolved:
            raise FormulaLinkageError(
                f"duplicate agent_api calculator id: {calculator_id!r}"
            )
        calculation_id, equation_filter = _parse_binding_target(
            calculator_id,
            raw_target,
        )
        if calculation_id not in calculations:
            raise FormulaLinkageError(f"unknown calculation_id: {calculation_id!r}")
        resolved[calculator_id] = _formula_ids_for_binding(
            calculations[calculation_id],
            equation_filter=equation_filter,
        )
    return resolved


def _bindings_from_payloads(
    *,
    bindings_payload: dict[str, Any],
    registry_payload: dict[str, Any],
) -> dict[str, tuple[str, ...]]:
    raw_bindings = bindings_payload.get("bindings")
    if not isinstance(raw_bindings, dict):
        raise FormulaLinkageError("bindings must be an object")
    return _resolve_bindings(raw_bindings, _calculations_index(registry_payload))


@lru_cache(maxsize=1)
def load_agent_api_bindings() -> dict[str, tuple[str, ...]]:
    """Return calculator-id → formula_ids for every bound MCP calculator."""
    return _bindings_from_payloads(
        bindings_payload=_load_json(_BINDINGS_PATH, "agent API bindings"),
        registry_payload=_load_json(_REGISTRY_PATH, "calculation registry"),
    )


def expected_formula_ids(
    calculator_id: str,
    *,
    bindings_payload: dict[str, Any] | None = None,
    registry_payload: dict[str, Any] | None = None,
) -> tuple[str, ...]:
    """Resolve expected formula_ids for one calculator id (DbC).

    Preconditions:
        ``calculator_id`` is a non-empty string.

    Postconditions:
        Returns the tuple wired into :class:`CalculatorSpec` when a binding exists.

    Raises:
        FormulaLinkageError: Unknown calculator id, or registry inconsistency.
    """
    if not isinstance(calculator_id, str) or not calculator_id.strip():
        raise FormulaLinkageError("calculator_id must be a non-empty string")
    if bindings_payload is not None and registry_payload is not None:
        bindings = _bindings_from_payloads(
            bindings_payload=bindings_payload, registry_payload=registry_payload
        )
    else:
        bindings = load_agent_api_bindings()
    if calculator_id not in bindings:
        raise FormulaLinkageError(f"unknown calculator_id binding: {calculator_id!r}")
    return bindings[calculator_id]
