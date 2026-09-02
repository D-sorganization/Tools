"""Data-free golden conformance bundle for launch-monitor analytics consumers.

Travels with step **P17** of the ADR-0046 G1 port plan (UpstreamDrift
``docs/adr/0048-launch-monitor-port-plan.md``) from UpstreamDrift's
``tests/unit/launch_monitor/test_conformance_bundle.py``.

What travels and what does not
------------------------------
UpstreamDrift's suite asserts against a bundle produced by
``scripts/launch_monitor_conformance_fixture.py`` and compares it to two
committed artifacts under ``docs/api/contracts/``. Neither the script nor the
artifacts are in the port plan's inventory, and the P10-P11 and P12-P16 ports
already set the precedent for the artifacts: a published contract file is
UpstreamDrift's own API surface, and a second committed copy here would be a
second thing to drift. So
``test_canonical_bundle_hash_and_published_artifacts_match_authority`` does not
travel; its obligations are asserted directly against
``launch_monitor_conformance_bundle_json_schema()`` instead, which additionally
pins the discriminated payload union and ``extra=forbid`` reaching the wire —
things a file that has to be regenerated cannot guarantee.

The *builder* does travel, as ``build_conformance_bundle`` below, because
without a real ten-scenario bundle there is nothing to validate. It is
UpstreamDrift's construction with three necessary adaptations, each of which is
a fact about this repository rather than a change of behaviour:

* the expected-strokes **baseline half** did not travel at P12 (it is already
  home in ``rate_of_closure.launch_monitor_strokes_gained_baseline``), so the
  strokes-gained scenario builds its baseline from the already-home
  ``StrokesGainedBaseline`` through the ``ExpectedStrokesBaselineLike``
  protocol — exactly the seam P12 left behind;
* the longitudinal scenario reads the attested fixture that travelled with P16
  into ``tests/shared/python/launch_monitor/fixtures/``;
* ``analyze_player_covariation_v1`` is P18's **union**, so the covariation
  payload carries the folded-in ``method_description`` and ruling D22's
  ``interval_withheld_reason``. The bundle is content-addressed over whatever
  the payloads actually are, so this is absorbed by construction rather than
  pinned to UpstreamDrift's digest — the two stacks' bundle hashes are not
  expected to be equal and nothing here claims they are.

The stay-green gate for P17 is not in this file. It is
``src/rate_of_closure/web/src/model/launchMonitorConformanceGolden.test.ts``
(and its Python sibling
``tests/rate_of_closure/test_launch_monitor_conformance_golden.py``), which
drive a committed golden bundle through the v2 client validators. This
port must not move it, and does not: it adds a Python module and touches no
fixture.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import BaseModel, ValidationError

from rate_of_closure.launch_monitor_strokes_gained_baseline import (
    BaselineState,
    StrokesGainedBaseline,
    baseline_table_hash,
)
from shared.python.launch_monitor import (
    BETWEEN_PLAYER_INTERVAL_MIN_GROUPS,
    LAUNCH_MONITOR_CONFORMANCE_BUNDLE_VERSION,
    AnalysisContextV2,
    CourseStateColumnsV1,
    FlexibleAnalysisRequest,
    LaunchMonitorConformanceBundleV1,
    LaunchMonitorConformanceScenarioV1,
    LongitudinalSessionRequestV1,
    MetricUnitsV2,
    OutcomeProxyRequestV1,
    PlayerCovariationRequestV1,
    PlayerIdentityV2,
    SourceFileReferenceV2,
    StrokesGainedRequestV1,
    analyze_longitudinal_sessions,
    analyze_outcome_proxy,
    analyze_player_covariation_v1,
    analyze_source_backed_strokes_gained,
    analyze_variables_v2,
    build_analysis_lineage_v2,
    launch_monitor_conformance_bundle_json_schema,
    launch_monitor_conformance_bundle_sha256,
    launch_monitor_conformance_scenario_sha256,
)

pytestmark = pytest.mark.unit

FIXTURE_DIR = Path(__file__).parent / "fixtures"
_SOURCE_ID = "synthetic-conformance-source"
_PORTABLE_FLOAT_SIGNIFICANT_DIGITS = 8

REQUIRED_CASES = {
    ("analysis_v2", "available"),
    ("analysis_v2", "unavailable"),
    ("player_covariation", "available"),
    ("player_covariation", "unavailable"),
    ("attested_longitudinal", "available"),
    ("attested_longitudinal", "unavailable"),
    ("source_backed_strokes_gained", "available"),
    ("source_backed_strokes_gained", "unavailable"),
    ("distance_target_proxy", "available"),
    ("distance_target_proxy", "unavailable"),
}


# ── the builder, ported from UpstreamDrift's fixture script ──────────────


def portable_snapshot_value(value: object) -> object:
    """Return JSON-compatible fixture content with portable float precision."""
    if isinstance(value, BaseModel):
        return portable_snapshot_value(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {key: portable_snapshot_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable_snapshot_value(item) for item in value]
    if isinstance(value, float):
        rounded = float(f"{value:.{_PORTABLE_FLOAT_SIGNIFICANT_DIGITS}g}")
        return 0.0 if rounded == 0.0 else rounded
    return value


def _source(frame: pd.DataFrame) -> SourceFileReferenceV2:
    content = json.dumps(
        frame.to_dict(orient="records"),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return SourceFileReferenceV2(
        source_id=_SOURCE_ID,
        file_sha256=sha256(content).hexdigest(),
        rights_status="public_redistributable",
    )


def _context(
    frame: pd.DataFrame, *, player_identity: PlayerIdentityV2 | None = None
) -> AnalysisContextV2:
    return AnalysisContextV2(
        sources=(_source(frame),),
        player_identity=player_identity or PlayerIdentityV2(),
        source_units={
            "ball_speed": "mph",
            "club_speed": "mph",
            "face_angle": "deg",
            "club_path": "deg",
            "carry_distance": "yd",
        },
    )


def _analysis_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "shot_id": [f"analysis-{index}" for index in range(6)],
            "source_id": [_SOURCE_ID] * 6,
            "source_row": list(range(6)),
            "monitor_vendor": ["TrackMan"] * 6,
            "monitor_model": ["synthetic-comparable"] * 6,
            "software_version": ["fixture-1"] * 6,
            "club_speed": [90, 91, 92, 93, 94, 95],
            "ball_speed": [130, 132, 133, 135, 136, 138],
        }
    )


def _covariation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "shot_id": [f"covariation-{index}" for index in range(8)],
            "source_id": [_SOURCE_ID] * 8,
            "source_row": list(range(8)),
            "player_id": ["player-a"] * 4 + ["player-b"] * 4,
            "face_angle": [0, 1, 2, 3, 10, 11, 12, 13],
            "club_path": [3, 2, 1, 0, 13, 12, 11, 10],
            "monitor_vendor": ["Foresight"] * 8,
            "monitor_model": ["synthetic-comparable"] * 8,
            "software_version": ["fixture-1"] * 8,
        }
    )


def _covariation_context(frame: pd.DataFrame) -> AnalysisContextV2:
    return _context(
        frame,
        player_identity=PlayerIdentityV2(
            trust_level="explicit_user_attested",
            identifier_column="player_id",
            evidence="Synthetic fixture identities are attested by construction.",
        ),
    )


def _longitudinal_inputs() -> tuple[pd.DataFrame, Any, AnalysisContextV2]:
    path = FIXTURE_DIR / "longitudinal_attested_v1.json"
    fixture = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame.from_records(fixture["records"])
    frame["source_id"] = _SOURCE_ID
    context_payload = fixture["context"]
    context_payload["sources"] = [_source(frame).model_dump(mode="json")]
    return (
        frame,
        LongitudinalSessionRequestV1.model_validate(fixture["request"]),
        AnalysisContextV2.model_validate(context_payload),
    )


def _baseline() -> StrokesGainedBaseline:
    """Build the already-home baseline through P12's structural seam."""
    states = (
        BaselineState("fairway", "standard", "hole-1", 100.0, 2.8, 0.10),
        BaselineState("fairway", "standard", "hole-1", 200.0, 3.8, 0.14),
        BaselineState("green", "standard", "hole-1", 0.0, 0.0, 0.0),
        BaselineState("green", "standard", "hole-1", 20.0, 1.5, 0.08),
    )
    documents = [
        {
            "lie": state.lie,
            "context": state.context,
            "target": state.target,
            "distance_yards": state.distance_yards,
            "expected_strokes": state.expected_strokes,
            "standard_error": state.standard_error,
        }
        for state in states
    ]
    return StrokesGainedBaseline(
        baseline_id="synthetic-published-method",
        version="fixture-1",
        source_url="https://example.org/synthetic-expected-strokes-method",
        license="CC0-1.0 synthetic fixture",
        table_sha256=baseline_table_hash(documents),
        states=states,
    )


def _strokes_gained_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "shot_id": [f"sg-{index}" for index in range(3)],
            "source_id": [_SOURCE_ID] * 3,
            "start_lie": ["fairway"] * 3,
            "start_context": ["standard"] * 3,
            "finish_lie": ["green"] * 3,
            "finish_context": ["standard"] * 3,
            "target": ["hole-1"] * 3,
            "start_distance": [140, 150, 160],
            "finish_distance": [20, 15, 10],
        }
    )


def _strokes_gained_request(min_samples: int) -> StrokesGainedRequestV1:
    return StrokesGainedRequestV1(
        start=CourseStateColumnsV1(
            lie_column="start_lie",
            context_column="start_context",
            target_column="target",
            distance_column="start_distance",
            distance_unit="yd",
        ),
        finish=CourseStateColumnsV1(
            lie_column="finish_lie",
            context_column="finish_context",
            target_column="target",
            distance_column="finish_distance",
            distance_unit="yd",
        ),
        shot_id_column="shot_id",
        min_samples=min_samples,
    )


def _unit_map(values: Mapping[str, object]) -> dict[str, MetricUnitsV2]:
    units: dict[str, MetricUnitsV2] = {}
    for name, value in values.items():
        if isinstance(value, MetricUnitsV2):
            units[name] = value
        else:
            unit = str(value)
            units[name] = MetricUnitsV2(
                canonical_unit=unit,
                display_unit=unit,
                authority="source_declared",
            )
    return units


def _scenario(**values: object) -> LaunchMonitorConformanceScenarioV1:
    portable_values = portable_snapshot_value(values)
    if not isinstance(portable_values, dict):
        raise TypeError("scenario snapshot must remain a mapping")
    values = {str(key): item for key, item in portable_values.items()}
    values["scenario_sha256"] = launch_monitor_conformance_scenario_sha256(values)
    return LaunchMonitorConformanceScenarioV1.model_validate(values)


def _lineage_scenario_values(
    *, result: Any, context: AnalysisContextV2, units: Mapping[str, object]
) -> dict[str, object]:
    lineage = result.lineage
    return {
        "units": _unit_map(units),
        "player_identity": context.player_identity,
        "session_identity": context.session_identity,
        "order_evidence": context.order_evidence,
        "sources": lineage.sources,
        "backing_records": lineage.backing_records,
    }


def _analysis_scenarios() -> tuple[LaunchMonitorConformanceScenarioV1, ...]:
    frame = _analysis_frame()
    context = _context(frame)
    request = FlexibleAnalysisRequest(
        outcome="ball_speed",
        predictors=("club_speed",),
        analysis_mode="correlation",
        min_samples=4,
    )
    results = (
        analyze_variables_v2(frame, request, context=context),
        analyze_variables_v2(frame.iloc[:2], request, context=context),
    )
    return tuple(
        _scenario(
            scenario_id=f"analysis-v2-{result.status}",
            analysis_kind="analysis_v2",
            expected_status=result.status,
            description=f"Synthetic analysis-v2 {result.status} result.",
            claims=result.claims.model_dump(mode="json"),
            exclusions=result.missingness.excluded_by_reason,
            payload=result,
            **_lineage_scenario_values(
                result=result, context=context, units=result.units
            ),
        )
        for result in results
    )


def _covariation_scenarios() -> tuple[LaunchMonitorConformanceScenarioV1, ...]:
    frame = _covariation_frame()
    context = _covariation_context(frame)
    request = PlayerCovariationRequestV1(
        x_column="face_angle",
        y_column="club_path",
        player_column="player_id",
    )
    results = (
        analyze_player_covariation_v1(frame, request, context=context),
        analyze_player_covariation_v1(frame.iloc[:2], request, context=context),
    )
    return tuple(
        _scenario(
            scenario_id=f"player-covariation-{result.status}",
            analysis_kind="player_covariation",
            expected_status=result.status,
            description=f"Synthetic player-covariation {result.status} result.",
            claims=result.claims.model_dump(mode="json"),
            exclusions=result.missingness.excluded_by_reason,
            payload=result,
            **_lineage_scenario_values(
                result=result, context=context, units=result.units
            ),
        )
        for result in results
    )


def _longitudinal_scenarios() -> tuple[LaunchMonitorConformanceScenarioV1, ...]:
    frame, request, context = _longitudinal_inputs()
    unavailable_context = context.model_copy(
        update={
            "order_evidence": context.order_evidence.model_copy(
                update={"trust_level": "untrusted_inferred"}
            )
        }
    )
    pairs = (
        (analyze_longitudinal_sessions(frame, request, context=context), context),
        (
            analyze_longitudinal_sessions(frame, request, context=unavailable_context),
            unavailable_context,
        ),
    )
    scenarios = []
    for result, scenario_context in pairs:
        claims = result.claims.model_dump(mode="json")
        claims["causal_inference"] = False
        scenarios.append(
            _scenario(
                scenario_id=f"attested-longitudinal-{result.status}",
                analysis_kind="attested_longitudinal",
                expected_status=result.status,
                description=f"Synthetic attested-longitudinal {result.status} result.",
                claims=claims,
                exclusions=result.missingness.excluded_by_reason,
                payload=result,
                **_lineage_scenario_values(
                    result=result,
                    context=scenario_context,
                    units={
                        request.metric: scenario_context.source_units[request.metric]
                    },
                ),
            )
        )
    return tuple(scenarios)


def _derived_lineage(
    frame: pd.DataFrame, context: AnalysisContextV2
) -> dict[str, object]:
    lineage = build_analysis_lineage_v2(frame, context)
    return {
        "player_identity": context.player_identity,
        "session_identity": context.session_identity,
        "order_evidence": context.order_evidence,
        "sources": lineage.sources,
        "backing_records": lineage.backing_records,
    }


def _strokes_gained_scenarios() -> tuple[LaunchMonitorConformanceScenarioV1, ...]:
    frame = _strokes_gained_frame()
    context = _context(frame)
    results = (
        analyze_source_backed_strokes_gained(
            frame, _baseline(), _strokes_gained_request(3), context=context
        ),
        analyze_source_backed_strokes_gained(
            frame, _baseline(), _strokes_gained_request(4), context=context
        ),
    )
    return tuple(
        _scenario(
            scenario_id=f"source-backed-strokes-gained-{result.status}",
            analysis_kind="source_backed_strokes_gained",
            expected_status=result.status,
            description=(
                f"Synthetic source-backed strokes-gained {result.status} result."
            ),
            units=_unit_map(result.units),
            claims=result.claims.model_dump(mode="json"),
            exclusions=result.exclusions.by_reason,
            payload=result,
            **_derived_lineage(frame, context),
        )
        for result in results
    )


def _proxy_scenarios() -> tuple[LaunchMonitorConformanceScenarioV1, ...]:
    available_frame = pd.DataFrame(
        {
            "shot_id": ["proxy-0", "proxy-1"],
            "source_id": [_SOURCE_ID, _SOURCE_ID],
            "carry": [150.0, 155.0],
            "lateral": [-5.0, 2.0],
        }
    )
    unavailable_frame = pd.DataFrame(
        {
            "shot_id": ["proxy-0", "proxy-1"],
            "source_id": [_SOURCE_ID, _SOURCE_ID],
            "carry": [150.0, "missing"],
            "lateral": [-5.0, 2.0],
        }
    )
    requests = (
        OutcomeProxyRequestV1(
            carry_column="carry",
            lateral_column="lateral",
            carry_unit="yd",
            lateral_unit="yd",
            target_distance_yards=150,
            shot_id_column="shot_id",
            min_samples=1,
        ),
        OutcomeProxyRequestV1(
            carry_column="carry",
            lateral_column="lateral",
            carry_unit="yd",
            lateral_unit="yd",
            target_distance_yards=150,
            shot_id_column="shot_id",
            min_samples=2,
        ),
    )
    pairs = tuple(
        (analyze_outcome_proxy(frame, request), frame, _context(frame))
        for frame, request in zip(
            (available_frame, unavailable_frame), requests, strict=True
        )
    )
    return tuple(
        _scenario(
            scenario_id=f"distance-target-proxy-{result.status}",
            analysis_kind="distance_target_proxy",
            expected_status=result.status,
            description=f"Synthetic distance/target proxy {result.status} result.",
            units=_unit_map(result.units),
            claims=result.claims.model_dump(mode="json"),
            exclusions=result.exclusions.by_reason,
            payload=result,
            **_derived_lineage(frame, context),
        )
        for result, frame, context in pairs
    )


def build_conformance_bundle() -> LaunchMonitorConformanceBundleV1:
    """Return all deterministic consumer cases without embedding input rows."""
    scenarios = (
        *_analysis_scenarios(),
        *_covariation_scenarios(),
        *_longitudinal_scenarios(),
        *_strokes_gained_scenarios(),
        *_proxy_scenarios(),
    )
    values: dict[str, object] = {
        "bundle_version": LAUNCH_MONITOR_CONFORMANCE_BUNDLE_VERSION,
        "description": (
            "Synthetic, data-free consumer outputs for canonical launch-monitor "
            "analytics; no observed or private shot rows are embedded."
        ),
        "data_classification": "synthetic_contract_fixture_no_private_rows",
        "input_records_embedded": False,
        "scenarios": scenarios,
    }
    values["bundle_sha256"] = launch_monitor_conformance_bundle_sha256(values)
    return LaunchMonitorConformanceBundleV1.model_validate(values)


@pytest.fixture(scope="module")
def bundle() -> LaunchMonitorConformanceBundleV1:
    return build_conformance_bundle()


def _walk_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(map(_walk_keys, value.values())))
    if isinstance(value, list):
        return set().union(*(map(_walk_keys, value)))
    return set()


# ── UpstreamDrift's cases, travelling with the module ────────────────────


def test_bundle_spans_available_and_unavailable_cases_without_input_rows(
    bundle: LaunchMonitorConformanceBundleV1,
) -> None:
    scenarios = {
        (item.analysis_kind, item.expected_status) for item in bundle.scenarios
    }

    assert bundle.bundle_version == LAUNCH_MONITOR_CONFORMANCE_BUNDLE_VERSION
    assert bundle.data_classification == "synthetic_contract_fixture_no_private_rows"
    assert bundle.input_records_embedded is False
    assert scenarios == REQUIRED_CASES
    payload = bundle.model_dump(mode="json")
    assert "records" not in _walk_keys(payload)
    assert "restricted_internal" not in json.dumps(payload)


def test_scenarios_retain_units_claims_evidence_lineage_and_exclusions(
    bundle: LaunchMonitorConformanceBundleV1,
) -> None:
    for scenario in bundle.scenarios:
        assert scenario.units
        assert scenario.claims["causal_inference"] is False
        assert scenario.sources
        assert scenario.backing_records
        source_ids = {source.source_id for source in scenario.sources}
        assert all(
            record.source_id in source_ids for record in scenario.backing_records
        )
        assert all(
            len(record.record_sha256) == 64 for record in scenario.backing_records
        )
        assert sum(scenario.exclusions.values()) >= 0

    longitudinal = next(
        item
        for item in bundle.scenarios
        if item.scenario_id == "attested-longitudinal-available"
    )
    assert longitudinal.player_identity.trust_level == "explicit_user_attested"
    assert longitudinal.session_identity.trust_level == "source_reported"
    assert longitudinal.session_identity.evidence
    assert longitudinal.order_evidence.trust_level == "explicit_user_attested"
    assert longitudinal.payload.claims.causal_improvement is False

    strokes_gained = next(
        item
        for item in bundle.scenarios
        if item.scenario_id == "source-backed-strokes-gained-available"
    )
    proxy = next(
        item
        for item in bundle.scenarios
        if item.scenario_id == "distance-target-proxy-available"
    )
    assert strokes_gained.claims["is_strokes_gained"] is True
    assert strokes_gained.claims["source_backed"] is True
    assert proxy.claims["is_strokes_gained"] is False
    assert proxy.claims["source_backed"] is False


def test_payload_and_bundle_hashes_fail_closed_after_mutation() -> None:
    payload = build_conformance_bundle().model_dump(mode="json")
    payload["scenarios"][0]["description"] = "tampered scenario"

    with pytest.raises(ValidationError, match="scenario_sha256"):
        LaunchMonitorConformanceBundleV1.model_validate(payload)

    payload = build_conformance_bundle().model_dump(mode="json")
    payload["description"] = "tampered bundle"
    with pytest.raises(ValidationError, match="bundle_sha256"):
        LaunchMonitorConformanceBundleV1.model_validate(payload)


def test_numeric_snapshot_is_quantized_for_cross_platform_invariance(
    bundle: LaunchMonitorConformanceBundleV1,
) -> None:
    longitudinal = next(
        scenario
        for scenario in bundle.scenarios
        if scenario.scenario_id == "attested-longitudinal-available"
    )
    pooled = longitudinal.payload.pooled_association

    assert pooled is not None
    assert pooled.standard_error == 0.16183472
    assert pooled.confidence_interval_low == 0.9849697
    assert pooled.confidence_interval_high == 2.0150303
    assert pooled.p_value == 0.0026577006

    windows_tail = {
        "standard_error": 0.16183471874253738,
        "confidence_interval_low": 0.984969697271183,
        "confidence_interval_high": 2.0150303027288152,
        "p_value": 0.0026577005664792496,
    }
    linux_tail = {
        "standard_error": 0.1618347187425374,
        "confidence_interval_low": 0.9849696972710931,
        "confidence_interval_high": 2.0150303027289054,
        "p_value": 0.0026577005664792513,
    }
    assert portable_snapshot_value(windows_tail) == portable_snapshot_value(linux_tail)


# ── replacing the published-artifact comparison with direct assertions ───


def test_generated_schema_carries_the_full_conformance_obligation() -> None:
    """Replaces UpstreamDrift's committed-schema comparison."""
    schema = launch_monitor_conformance_bundle_json_schema()

    assert schema["properties"]["bundle_version"]["const"] == (
        "launch-monitor-analytics-conformance/1.0.0"
    )
    assert schema["properties"]["data_classification"]["const"] == (
        "synthetic_contract_fixture_no_private_rows"
    )
    assert schema["properties"]["input_records_embedded"]["const"] is False
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "description",
        "scenarios",
        "bundle_sha256",
    }
    assert schema["properties"]["scenarios"]["minItems"] == 10

    scenario = schema["$defs"]["LaunchMonitorConformanceScenarioV1"]
    assert scenario["additionalProperties"] is False
    assert scenario["properties"]["scenario_sha256"]["pattern"] == r"^[0-9a-f]{64}$"

    # The five-way payload union reaches the wire as a discriminated union,
    # keyed on contract_version - the arm P18 had to land before P17 could.
    payload = scenario["properties"]["payload"]
    assert payload["discriminator"]["propertyName"] == "contract_version"
    assert len(payload["oneOf"]) == 5
    assert {
        "LaunchMonitorAnalysisResultV2",
        "PlayerCovariationResultV1",
        "LongitudinalSessionResultV1",
        "StrokesGainedAnalysisResultV1",
        "OutcomeProxyResultV1",
    } <= set(schema["$defs"])


# ── refusal pins the port earns ──────────────────────────────────────────


def test_bundle_refuses_a_missing_required_case() -> None:
    """Nine of ten cases is not a conformance bundle."""
    payload = build_conformance_bundle().model_dump(mode="json")
    payload["scenarios"] = payload["scenarios"][:9]
    payload["bundle_sha256"] = launch_monitor_conformance_bundle_sha256(payload)

    with pytest.raises(ValidationError, match="at least 10 items"):
        LaunchMonitorConformanceBundleV1.model_validate(payload)


def test_bundle_refuses_a_duplicated_scenario_identity() -> None:
    payload = build_conformance_bundle().model_dump(mode="json")
    payload["scenarios"].append(payload["scenarios"][0])
    payload["bundle_sha256"] = launch_monitor_conformance_bundle_sha256(payload)

    with pytest.raises(ValidationError, match="scenario_id values must be unique"):
        LaunchMonitorConformanceBundleV1.model_validate(payload)


def test_scenario_refuses_a_backing_record_with_no_declared_source() -> None:
    payload = build_conformance_bundle().model_dump(mode="json")
    scenario = payload["scenarios"][0]
    scenario["backing_records"][0]["source_id"] = "not-a-declared-source"
    scenario["scenario_sha256"] = launch_monitor_conformance_scenario_sha256(scenario)
    payload["bundle_sha256"] = launch_monitor_conformance_bundle_sha256(payload)

    with pytest.raises(ValidationError, match="join to a declared source_id"):
        LaunchMonitorConformanceBundleV1.model_validate(payload)


def test_scenario_refuses_a_causal_claim() -> None:
    payload = build_conformance_bundle().model_dump(mode="json")
    scenario = payload["scenarios"][0]
    scenario["claims"]["causal_inference"] = True
    scenario["scenario_sha256"] = launch_monitor_conformance_scenario_sha256(scenario)
    payload["bundle_sha256"] = launch_monitor_conformance_bundle_sha256(payload)

    with pytest.raises(ValidationError, match="forbid causal inference"):
        LaunchMonitorConformanceBundleV1.model_validate(payload)


def test_scenario_refuses_a_payload_whose_kind_or_status_disagrees() -> None:
    payload = build_conformance_bundle().model_dump(mode="json")
    scenario = next(
        item
        for item in payload["scenarios"]
        if item["scenario_id"] == "analysis-v2-available"
    )
    scenario["expected_status"] = "unavailable"
    scenario["scenario_sha256"] = launch_monitor_conformance_scenario_sha256(scenario)
    payload["bundle_sha256"] = launch_monitor_conformance_bundle_sha256(payload)

    with pytest.raises(ValidationError, match="expected_status does not match"):
        LaunchMonitorConformanceBundleV1.model_validate(payload)


def test_scenario_refuses_a_negative_exclusion_count() -> None:
    payload = build_conformance_bundle().model_dump(mode="json")
    scenario = payload["scenarios"][0]
    scenario["exclusions"] = {"fabricated": -1}
    scenario["scenario_sha256"] = launch_monitor_conformance_scenario_sha256(scenario)
    payload["bundle_sha256"] = launch_monitor_conformance_bundle_sha256(payload)

    with pytest.raises(ValidationError, match="counts non-negative"):
        LaunchMonitorConformanceBundleV1.model_validate(payload)


def test_covariation_arm_carries_p18_union_and_ruling_fields(
    bundle: LaunchMonitorConformanceBundleV1,
) -> None:
    """P17's payload union resolves to the *canonical* covariation result."""
    scenario = next(
        item
        for item in bundle.scenarios
        if item.scenario_id == "player-covariation-available"
    )
    payload = scenario.payload

    # method_description is P18's folded-in rate_of_closure capability, and it
    # is required - so a bundle built before P18 could not have validated.
    assert payload.method_description
    # D22: the named threshold reaches the wire on the uncertainty block, and
    # the between-player estimate explains any absent interval rather than
    # returning a silent None.
    assert payload.uncertainty.between_player_interval_min_groups == (
        BETWEEN_PLAYER_INTERVAL_MIN_GROUPS
    )
    between = payload.between_player
    if between.state == "available":
        assert (between.ci_lower is None) == (
            between.interval_withheld_reason is not None
        )


def test_conformance_bundle_module_does_not_import_rate_of_closure() -> None:
    """The canonical layer never depends on the legacy package."""
    import ast
    from importlib.util import find_spec

    spec = find_spec("shared.python.launch_monitor.conformance_bundle")
    assert spec is not None and spec.origin is not None
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))

    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    assert not any(name.split(".")[0] == "rate_of_closure" for name in modules)
