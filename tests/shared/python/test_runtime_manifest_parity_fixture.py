"""Python consumer for the shared runtime-manifest parity fixture.

The fixture is shared byte-for-byte with
``src/rate_of_closure/web/src/model/runtimeManifest.test.ts``. The Python side
owns the canonical numeric JSON authority (``canonical_numeric_json``), so this
module pins the numeric policy cases and the canonical manifest bytes against
the same fixture the TypeScript runtime checks.

Scope note: the manifest schema validator half of this fixture is implemented
by ``shared.python.swing_sim.runtime_manifest.parse_runtime_manifest``, a
one-for-one port of the TypeScript ``parseRuntimeManifest``. The parser cases
below mirror ``runtimeManifest.test.ts`` and are driven from the same fixture
(issues #4560, #5416).
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from shared.python.swing_sim.canonical_numeric_json import canonical_numeric_json
from shared.python.swing_sim.runtime_manifest import (
    RUNTIME_MANIFEST_SCHEMA,
    parse_runtime_manifest,
    runtime_manifest_from_json,
    stable_runtime_manifest_json,
)

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

Manifest = dict[str, Any]
_SAFE = 9_007_199_254_740_991


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


# --- parseRuntimeManifest parity (issue #5416) -------------------------------


def _source(payload: dict[str, Any]) -> Manifest:
    return copy.deepcopy(payload["manifest"])


def _calc(value: Manifest, index: int) -> dict[str, Any]:
    return value["calculations"][index]  # type: ignore[no-any-return]


def _set_option_value(value: Manifest, option_value: Any) -> None:
    _calc(value, 1)["numerical_options"][0]["value"] = option_value


def _set_version(value: Manifest, version: str) -> None:
    value["build"]["package_version"] = version


def _set_reason(value: Manifest, reason: str | None) -> None:
    _calc(value, 2)["reason"] = reason


def _duplicate_option(value: Manifest) -> None:
    options = _calc(value, 0)["numerical_options"]
    options.append(copy.deepcopy(options[0]))


def _set_item(container: str, key: str, item: Any) -> Callable[[Manifest], None]:
    def mutate(value: Manifest) -> None:
        target = value if container == "" else value[container]
        target[key] = item

    return mutate


def _set_calc(index: int, key: str, item: Any) -> Callable[[Manifest], None]:
    def mutate(value: Manifest) -> None:
        _calc(value, index)[key] = item

    return mutate


_REJECTIONS: list[tuple[str, Callable[[Manifest], None]]] = [
    ("unknown top-level field", _set_item("", "extra", True)),
    ("unknown nested field", _set_item("build", "extra", True)),
    (
        "unsupported schema",
        _set_item("", "schema_version", "calculation-runtime-manifest/v2"),
    ),
    ("unknown surface", _set_item("", "surface_id", "tools.cli")),
    ("non-SHA revision", _set_item("build", "tools_commit", "working-tree")),
    ("uppercase SHA", _set_item("build", "tools_commit", "A" * 40)),
    ("leading-zero major", lambda v: _set_version(v, "01.0.0")),
    ("leading-zero minor", lambda v: _set_version(v, "1.00.0")),
    ("leading-zero patch", lambda v: _set_version(v, "1.0.00")),
    ("leading-zero prerelease", lambda v: _set_version(v, "1.0.0-01")),
    ("duplicate domain", _set_calc(2, "domain", "flight")),
    ("out-of-order domains", lambda v: v["calculations"].reverse()),
    ("available reason", _set_calc(0, "reason", "fallback")),
    ("available missing authority", _set_calc(0, "implementation_authority", None)),
    ("unavailable model leak", _set_calc(2, "model_id", "unqualified")),
    ("unavailable missing reason", lambda v: _set_reason(v, None)),
    ("placeholder unavailable reason", lambda v: _set_reason(v, "Unknown")),
    ("one-letter unavailable reason", lambda v: _set_reason(v, "x")),
    ("abbreviated unavailable reason", lambda v: _set_reason(v, "n/a")),
    (
        "whitespace sentinel unavailable reason",
        lambda v: _set_reason(v, " \tUNAVAILABLE\n"),
    ),
    (
        "surrounding whitespace on explanatory reason",
        lambda v: _set_reason(
            v, " No qualified ground producer was selected for this run. "
        ),
    ),
    ("duplicate option", _duplicate_option),
    (
        "numeric option without unit",
        lambda v: _calc(v, 1)["numerical_options"][0].update(unit=None),
    ),
    (
        "text option with unit",
        lambda v: _calc(v, 0)["numerical_options"][0].update(unit="1"),
    ),
    (
        "duplicate evidence",
        lambda v: v["provenance"]["evidence_ids"].append("issue-4261"),
    ),
    (
        "surrogate provenance text",
        _set_item("provenance", "source_reference", "fixture-\ud800"),
    ),
    ("calculations not an array", _set_item("", "calculations", {})),
]


def test_parser_accepts_fixture_and_round_trips_canonical_bytes(
    fixture_payload: dict[str, Any],
) -> None:
    parsed = parse_runtime_manifest(fixture_payload["manifest"])

    assert (
        stable_runtime_manifest_json(parsed)
        == fixture_payload["expected_canonical_json"]
    )
    assert parsed == fixture_payload["manifest"]
    assert parsed["schema_version"] == RUNTIME_MANIFEST_SCHEMA
    assert runtime_manifest_from_json(json.dumps(fixture_payload["manifest"])) == parsed


def test_parser_returns_independent_copy(fixture_payload: dict[str, Any]) -> None:
    parsed = parse_runtime_manifest(fixture_payload["manifest"])
    parsed["calculations"][0]["numerical_options"][0]["value"] = "mutated"

    original = fixture_payload["manifest"]["calculations"][0]["numerical_options"][0]
    assert original["value"] == "constant"


@pytest.mark.parametrize(
    "mutate", [m for _, m in _REJECTIONS], ids=[n for n, _ in _REJECTIONS]
)
def test_parser_rejects_invalid_manifest(
    fixture_payload: dict[str, Any], mutate: Callable[[Manifest], None]
) -> None:
    value = _source(fixture_payload)
    mutate(value)

    with pytest.raises((TypeError, ValueError)):
        parse_runtime_manifest(value)


def test_parser_rejects_non_mapping_input() -> None:
    with pytest.raises(TypeError, match="must be an object"):
        parse_runtime_manifest([])


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_parser_rejects_nonfinite_option(
    fixture_payload: dict[str, Any], bad: float
) -> None:
    value = _source(fixture_payload)
    _set_option_value(value, bad)

    with pytest.raises(ValueError, match="finite"):
        parse_runtime_manifest(value)


@pytest.mark.parametrize("bad", [_SAFE + 1, 1e16, -1e20, -(_SAFE + 1)])
def test_parser_rejects_unsafe_numeric_magnitude(
    fixture_payload: dict[str, Any], bad: float
) -> None:
    value = _source(fixture_payload)
    _set_option_value(value, bad)

    with pytest.raises(ValueError, match="safe numeric magnitude"):
        parse_runtime_manifest(value)


@pytest.mark.parametrize("edge", [-_SAFE, _SAFE])
def test_parser_accepts_safe_boundaries(
    fixture_payload: dict[str, Any], edge: int
) -> None:
    value = _source(fixture_payload)
    _set_option_value(value, edge)

    assert str(edge) in stable_runtime_manifest_json(parse_runtime_manifest(value))


def test_boolean_options_are_not_numeric(fixture_payload: dict[str, Any]) -> None:
    value = _source(fixture_payload)
    option = _calc(value, 0)["numerical_options"][0]
    option["value"] = True
    parsed = parse_runtime_manifest(value)
    assert parsed["calculations"][0]["numerical_options"][0]["value"] is True

    option["unit"] = "1"
    with pytest.raises(ValueError, match="numeric options require a unit"):
        parse_runtime_manifest(value)


def test_parser_rejects_non_scalar_option_value(
    fixture_payload: dict[str, Any],
) -> None:
    value = _source(fixture_payload)
    _set_option_value(value, [1])

    with pytest.raises(TypeError, match="option value"):
        parse_runtime_manifest(value)


def test_json_entry_point_rejects_duplicates_and_malformed_text() -> None:
    with pytest.raises(ValueError, match="duplicate JSON field"):
        runtime_manifest_from_json(
            '{"schema_version":"first","schema_version":"second"}'
        )
    with pytest.raises(ValueError, match="invalid runtime manifest JSON"):
        runtime_manifest_from_json("{")
    with pytest.raises(TypeError, match="must be text"):
        runtime_manifest_from_json(b"{}")


def test_accepts_explanatory_reason_and_strict_semver(
    fixture_payload: dict[str, Any],
) -> None:
    value = _source(fixture_payload)
    _set_version(value, "1.2.3-alpha.1+build.5")
    _set_reason(value, "No qualified ground producer was selected for this run.")

    assert parse_runtime_manifest(value) == value


def test_numeric_policy_cases_match_the_parser(
    fixture_payload: dict[str, Any],
) -> None:
    for unsafe in fixture_payload["numeric_policy_cases"]["unsafe_magnitudes"]:
        value = _source(fixture_payload)
        _set_option_value(value, unsafe)
        with pytest.raises(ValueError, match="safe numeric magnitude"):
            parse_runtime_manifest(value)


def test_reason_policy_cases(fixture_payload: dict[str, Any]) -> None:
    cases = fixture_payload["reason_policy_cases"]
    valid = _source(fixture_payload)
    _set_reason(valid, cases["valid_astral_reason"])
    parsed = parse_runtime_manifest(valid)

    assert runtime_manifest_from_json(stable_runtime_manifest_json(parsed)) == parsed
    assert cases["valid_astral_reason"] in stable_runtime_manifest_json(parsed)
    for boundary in cases["boundary_whitespace"]:
        for reason in (
            boundary + cases["valid_astral_reason"],
            cases["valid_astral_reason"] + boundary,
        ):
            value = _source(fixture_payload)
            _set_reason(value, reason)
            with pytest.raises(ValueError, match="surrounding whitespace"):
                parse_runtime_manifest(value)

    invalid = _source(fixture_payload)
    _set_reason(invalid, f"No qualified {cases['unpaired_surrogate']} ground producer.")
    with pytest.raises(ValueError, match="surrogate"):
        parse_runtime_manifest(invalid)


@pytest.mark.parametrize(
    ("reason", "message"),
    [
        ("too short here", "16 to 500"),
        ("a" * 501, "16 to 500"),
        ("1234567890 12345 6789 a b", "three explanatory words"),
    ],
)
def test_reason_length_and_word_rules(
    fixture_payload: dict[str, Any], reason: str, message: str
) -> None:
    value = _source(fixture_payload)
    _set_reason(value, reason)

    with pytest.raises(ValueError, match=message):
        parse_runtime_manifest(value)


def test_placeholder_token_boundaries(fixture_payload: dict[str, Any]) -> None:
    cases = fixture_payload["placeholder_policy_cases"]
    for token in cases["tokens"]:
        for separator in cases["stable_id_separators"]:
            for build_id in (
                f"release{separator}{token}",
                f"{token}{separator}release",
            ):
                value = _source(fixture_payload)
                value["build"]["build_id"] = build_id
                with pytest.raises(ValueError, match="placeholder"):
                    parse_runtime_manifest(value)

    for build_id in cases["valid_substrings"]:
        value = _source(fixture_payload)
        value["build"]["build_id"] = build_id
        assert parse_runtime_manifest(value)["build"]["build_id"] == build_id
