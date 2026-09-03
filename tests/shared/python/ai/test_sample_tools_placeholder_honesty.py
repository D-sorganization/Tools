"""Chat tools must not claim work was queued when nothing was started.

Three tools in :mod:`shared.python.ai.sample_tools` are registered but
unimplemented. Before this suite existed they returned payloads reading
``"... queued ..."`` / ``status="simulation_pending"`` -- and for
``run_inverse_dynamics``, ``success=True`` -- for work that never starts.
UpstreamDrift PR #7391 made them honest; the consolidation PR #8322 reverted
the ``src/`` half while keeping the tests, and the regression rode into Tools
with the vendored copy.

The invariant under test is deliberately stated over *every* registered
placeholder rather than the three known ones, so a newly added placeholder
that hand-rolls a dishonest dict fails here instead of shipping.

Part of D-sorganization/UpstreamDrift#9474.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.shared.python.ai.sample_tools import (
    _not_implemented_tool_result,
    register_golf_suite_tools,
)
from src.shared.python.ai.tool_registry import ToolRegistry

pytestmark = [pytest.mark.unit]

# Tools that are registered but have no implementation behind them.
_PLACEHOLDER_TOOLS: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "run_inverse_dynamics",
        {"file_path": "swing.c3d", "engine": "mujoco"},
    ),
    (
        "validate_cross_engine",
        {"file_path": "swing.c3d"},
    ),
    (
        "check_energy_conservation",
        {},
    ),
)

# Words that assert a job exists somewhere. A placeholder may not use them.
_PROGRESS_CLAIMS = ("queued", "pending", "in progress", "running", "started")


def _invoke(tool_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    registry = ToolRegistry()
    register_golf_suite_tools(registry)
    tool = registry.get_tool(tool_name)
    assert tool is not None, f"{tool_name} is not registered"
    return tool.handler(**kwargs)


@pytest.mark.parametrize(("tool_name", "kwargs"), _PLACEHOLDER_TOOLS)
def test_placeholder_reports_failure_not_success(
    tool_name: str, kwargs: dict[str, Any]
) -> None:
    """A tool that starts no work must report ``success=False``."""
    payload = _invoke(tool_name, kwargs)

    assert payload["success"] is False, (
        f"{tool_name} reported success for work it never performed. "
        "Route unimplemented tools through _not_implemented_tool_result()."
    )
    assert payload["status"] == "not_implemented"
    assert payload["error"] == "not_implemented"


@pytest.mark.parametrize(("tool_name", "kwargs"), _PLACEHOLDER_TOOLS)
def test_placeholder_does_not_claim_a_job_exists(
    tool_name: str, kwargs: dict[str, Any]
) -> None:
    """No placeholder may describe queued, pending or running work."""
    payload = _invoke(tool_name, kwargs)
    prose = " ".join(
        str(value)
        for key, value in payload.items()
        if key in {"message", "note", "status"}
    ).lower()

    offenders = [claim for claim in _PROGRESS_CLAIMS if claim in prose]
    assert not offenders, (
        f"{tool_name} claims work is {offenders} but enqueues nothing. "
        f"Payload prose was: {prose!r}"
    )


def test_placeholder_helper_enforces_its_postcondition() -> None:
    """The shared helper is the single place the invariant is expressed."""
    payload = _not_implemented_tool_result(
        capability="example",
        message="Example is not implemented yet.",
        extra="carried through",
    )

    assert payload["success"] is False
    assert payload["status"] == "not_implemented"
    assert payload["error"] == "not_implemented"
    assert payload["capability"] == "example"
    assert payload["extra"] == "carried through"


def test_helper_rejects_metadata_that_would_overwrite_the_verdict() -> None:
    """Metadata must not be able to smuggle ``success=True`` back in.

    Without this the postcondition is defeatable by a caller passing
    ``success=True`` as metadata, which is exactly the shape of the original
    regression.
    """
    from src.shared.python.contracts import ContractViolationError

    with pytest.raises((ContractViolationError, TypeError, ValueError)):
        _not_implemented_tool_result(
            capability="example",
            message="Example is not implemented yet.",
            success=True,
        )
