"""Python consumer for the shared runtime-manifest parity fixture.

The fixture is shared byte-for-byte with
``src/rate_of_closure/web/src/model/runtimeManifest.test.ts``. The Python side
owns the canonical numeric JSON authority (``canonical_numeric_json``), so this
module pins the numeric policy cases and the canonical manifest bytes against
the same fixture the TypeScript runtime checks.

Scope note: the *manifest schema validator* half of this fixture has no Python
implementation yet — the TypeScript ``parseRuntimeManifest`` rules are the only
enforcement of surfaces, domains, versions, reasons, and placeholders. Until a
Python validator lands, that half stays TypeScript-only by design; this module
pins the canonical-JSON contract that both runtimes already share (issue
#4560).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from shared.python.swing_sim.canonical_numeric_json import canonical_numeric_json

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).parents[3].resolve()
_FIXTURE_PATH = (
    _REPO_ROOT
    / "src"
    / "rate_of_closure"
    / "web"
    / "src"
    / "model"
    / "__fixtures__"
    / "runtime_manifest_parity_v1.json"
)


@pytest.fixture(scope="module")
def fixture_payload() -> dict[str, Any]:
    """Load the shared parity fixture exactly as the TypeScript side does."""
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def test_canonical_encoder_matches_the_pinned_manifest_bytes(
    fixture_payload: dict[str, Any],
) -> None:
    """Python canonicalization must reproduce the TS canonical bytes."""
    assert (
        canonical_numeric_json(fixture_payload["manifest"])
        == fixture_payload["expected_canonical_json"]
    )


def test_safe_integer_boundaries_serialize_exactly(
    fixture_payload: dict[str, Any],
) -> None:
    """The cross-runtime safe range must survive canonicalization verbatim."""
    case = fixture_payload["numeric_policy_cases"]

    assert (
        canonical_numeric_json(case["safe_boundaries"])
        == case["expected_canonical_json"]
    )


def test_unsafe_magnitudes_fail_closed(fixture_payload: dict[str, Any]) -> None:
    """Magnitudes beyond the JS safe range must raise, matching the TS gate."""
    case = fixture_payload["numeric_policy_cases"]

    for value in case["unsafe_magnitudes"]:
        with pytest.raises(ValueError, match="cross-runtime safe range"):
            canonical_numeric_json([value])
