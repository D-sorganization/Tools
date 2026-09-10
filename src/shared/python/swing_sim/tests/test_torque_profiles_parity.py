"""Cross-runtime parity consumer for ``torque_profile_parity.json``.

The fixture is the literal Python-produced payload that the TypeScript side
consumes in ``src/rate_of_closure/web/src/model/torqueProfiles.test.ts``; this
module closes the loop so the same bytes are checked against the Python
implementation too (issue #4560).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.python.swing_sim.torque_profiles import (
    PrescribedTorqueProfile,
    evaluate_ascending_polynomial,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[5]
_FIXTURE_PATH = (
    _REPO_ROOT
    / "src"
    / "rate_of_closure"
    / "web"
    / "src"
    / "model"
    / "__fixtures__"
    / "torque_profile_parity.json"
)


@pytest.fixture(scope="module")
def fixture_payload() -> dict[str, object]:
    """Load the shared parity fixture exactly as the TypeScript side does."""
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def test_python_implementation_loads_the_literal_payload(
    fixture_payload: dict[str, object],
) -> None:
    """The strict Python loader must accept the TS-shared bytes verbatim."""
    profile = PrescribedTorqueProfile.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))

    assert profile.to_json_dict() == fixture_payload


def test_python_evaluation_matches_the_pinned_type_script_values() -> None:
    """Pinned evaluation values shared with ``torqueProfiles.test.ts``."""
    profile = PrescribedTorqueProfile.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))

    assert profile.evaluate(0.5) == {
        "joint.shoulder": 9.0,
        "joint.wrist": 1.375,
    }
    assert evaluate_ascending_polynomial([2, 3, 4], 2) == 24


def test_python_round_trips_deterministically() -> None:
    """Serialization must be sorted-key deterministic, matching the TS gate."""
    profile = PrescribedTorqueProfile.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    first = profile.dumps()

    assert PrescribedTorqueProfile.loads(first).dumps() == first
    assert first.index('"author"') < first.index('"run_id"')
