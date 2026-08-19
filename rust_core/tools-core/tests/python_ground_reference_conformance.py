"""Run the analytic corpus through the actually installed PyO3 wheel."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import tools_core

SUPPORT_DIRECTORY = (
    Path(__file__).parents[3]
    / "src"
    / "shared"
    / "python"
    / "swing_sim"
    / "ground"
    / "tests"
)
sys.path.insert(0, str(SUPPORT_DIRECTORY))

from conformance_support import (  # noqa: E402
    assert_conformance_case,
    load_conformance_cases,
    materialize_case,
)


def main() -> None:
    """Require the installed extension to satisfy every shared case."""
    template, cases = load_conformance_cases()
    for case in cases:
        assert "pyo3" in case["platforms"]
        request, execution = materialize_case(template, case)
        result = json.loads(
            tools_core.run_flight_to_ground_reference_v1(
                json.dumps(request, separators=(",", ":")),
                json.dumps(execution, separators=(",", ":")),
            )
        )
        assert_conformance_case(result, request, case)


if __name__ == "__main__":
    main()
