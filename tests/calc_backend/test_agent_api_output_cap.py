"""Output cap guard for calculator MCP (issue #6021)."""

from __future__ import annotations

import pytest

from shared.python.calc_backend.agent_api import MAX_OUTPUT_BYTES, enforce_output_cap


def test_enforce_output_cap_raises_when_payload_too_large() -> None:
    huge = {"data": "x" * (MAX_OUTPUT_BYTES + 1)}
    with pytest.raises(ValueError, match="exceeded"):
        enforce_output_cap(huge)
