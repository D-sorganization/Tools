"""Canonical source-backed strokes-gained and outcome-proxy contracts.

Travels with steps **P12** (``strokes_gained_types`` minus the baseline half,
plus ``_scoring_statistics``), **P13** (``outcome_proxy``) and **P14**
(``strokes_gained``) of the ADR-0046 G1 port plan, from UpstreamDrift
``tests/unit/launch_monitor/test_strokes_gained_contract.py``.

Two of UpstreamDrift's cases deliberately do **not** travel.

``test_baseline_hash_is_numeric_and_row_order_canonical`` and
``test_baseline_rejects_tamper_and_duplicate_course_state`` exercise the
expected-strokes baseline half, which the port plan names as the one
sub-module that is genuinely already home: G0's
``test_baseline_table_digest_agrees_across_stacks`` pins UpstreamDrift's
``baseline_table_sha256`` and this repository's ``baseline_table_hash`` to the
identical digest, and ``launch_monitor_strokes_gained_baseline`` additionally
carries a byte cap and source-URL validation UpstreamDrift lacks. Porting those
two cases would test a second copy of an authority this repository already
owns and already tests. What replaces them is
``test_already_home_baseline_satisfies_the_canonical_protocol``, which pins the
*seam* the port created instead — the thing that is genuinely new and could
genuinely break.

``test_published_strokes_gained_schema_matches_python_authority`` compares the
schema to UpstreamDrift's committed ``docs/api/contracts/`` artifact. That file
is UpstreamDrift's published API surface, not part of this model layer; a
second committed copy here would be a second thing to drift. Its obligation
travels as an assertion against the generated schema directly, which cannot go
stale behind an un-regenerated file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from rate_of_closure.launch_monitor_performance import (
    TargetErrorRequest,
    calculate_target_error,
)
from rate_of_closure.launch_monitor_strokes_gained_baseline import (
    CONTRACT_VERSION as ALREADY_HOME_BASELINE_CONTRACT_VERSION,
)
from rate_of_closure.launch_monitor_strokes_gained_baseline import (
    BaselineState,
    StrokesGainedBaseline,
    baseline_table_hash,
)
from shared.python.launch_monitor import (
    BASELINE_CONTRACT_VERSION,
    CourseStateColumnsV1,
    ExpectedStrokesBaselineLike,
    ExpectedStrokesStateLike,
    GroupingDimensionV1,
    LongitudinalDimensionV1,
    OutcomeProxyRequestV1,
    StrokesGainedRequestV1,
    analyze_outcome_proxy,
    analyze_source_backed_strokes_gained,
    strokes_gained_contract_json_schema,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

FIXTURE_DIR = Path(__file__).parent / "fixtures"
CROSS_STACK_FIXTURE = FIXTURE_DIR / "adr0046_cross_stack_session_v1.json"

# G0 pins, reproduced here against the canonical layer rather than assumed.
G0_SESSION_MEAN = 0.80592372152815683
G0_INCLUDED_ROWS = 160

# G1-D2: the shot-level fit UpstreamDrift shipped, preserved as a named variant.
G0_SHOT_LEVEL_P4_SLOPE = 0.075881035543697128
G0_SHOT_LEVEL_P4_R_SQUARED = 0.15450437016457175
G0_SHOT_LEVEL_P4_P_VALUE = 0.012104880151308768
G0_SHOT_LEVEL_SAMPLE_COUNT = 40

# G1-D2: the canonical session-cell fit that replaces it as the default.
G0_SESSION_CELL_P4_SLOPE = 0.075881035543697113
G0_SESSION_CELL_P4_R_SQUARED = 0.5682576505731145
G0_SESSION_CELL_P4_P_VALUE = 0.1410798565763777
G0_SESSION_CELL_SAMPLE_COUNT = 5


def _states() -> tuple[BaselineState, ...]:
    """Build UpstreamDrift's four benchmark points with the already-home type."""
    return (
        BaselineState("fairway", "standard", "hole-1", 100.0, 2.8, 0.10),
        BaselineState("fairway", "standard", "hole-1", 200.0, 3.8, 0.14),
        BaselineState("green", "standard", "hole-1", 0.0, 0.0, 0.0),
        BaselineState("green", "standard", "hole-1", 20.0, 1.5, 0.08),
    )


def _state_documents() -> list[dict[str, object]]:
    return [
        {
            "lie": state.lie,
            "context": state.context,
            "target": state.target,
            "distance_yards": state.distance_yards,
            "expected_strokes": state.expected_strokes,
            "standard_error": state.standard_error,
        }
        for state in _states()
    ]


def _baseline() -> StrokesGainedBaseline:
    states = _states()
    return StrokesGainedBaseline(
        baseline_id="licensed-test-baseline",
        version="2026.1",
        source_url="https://example.org/expected-strokes-method",
        license="test-only",
        table_sha256=baseline_table_hash(_state_documents()),
        states=states,
    )


def _request(*, min_samples: int = 3) -> StrokesGainedRequestV1:
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
            distance_column="finish_distance_m",
            distance_unit="m",
        ),
        shot_id_column="shot_id",
        confidence_level=0.95,
        min_samples=min_samples,
        summaries=(
            GroupingDimensionV1(
                dimension="player",
                column="player_id",
                trust_level="pseudonymous_stable",
                evidence="Stable study pseudonym supplied by the owner.",
            ),
            GroupingDimensionV1(
                dimension="session",
                column="session_id",
                trust_level="explicit_user_attested",
                evidence="The user attested the exported session identifier.",
            ),
            GroupingDimensionV1(
                dimension="club",
                column="club",
                trust_level="verified_external",
                evidence="Club identity was verified against capture metadata.",
            ),
        ),
        longitudinal=LongitudinalDimensionV1(
            order_column="session_order",
            order_unit="session",
            group_column="player_id",
            group_dimension="player",
            trust_level="pseudonymous_stable",
            evidence="Stable player pseudonym and chronological session order.",
            min_samples=3,
        ),
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "shot_id": [f"shot-{index}" for index in range(6)],
            "player_id": ["player-a"] * 3 + ["player-b"] * 3,
            "session_id": ["s1", "s2", "s3", "s1", "s2", "s3"],
            "session_order": [1, 2, 3, 1, 2, 3],
            "club": ["7i"] * 6,
            "start_lie": ["fairway"] * 6,
            "start_context": ["standard"] * 6,
            "target": ["hole-1"] * 6,
            "start_distance": [150.0, 160.0, 170.0, 150.0, 160.0, 170.0],
            "finish_lie": ["green"] * 6,
            "finish_context": ["standard"] * 6,
            "finish_distance_m": [18.288, 13.716, 9.144, 18.288, 13.716, 9.144],
        }
    )


@pytest.fixture(scope="module")
def cross_stack_payload() -> dict:
    """Return UpstreamDrift's committed ADR-0046 G0 cross-stack session."""
    return json.loads(CROSS_STACK_FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cross_stack_frame(cross_stack_payload: dict) -> pd.DataFrame:
    return pd.DataFrame.from_records(cross_stack_payload["records"])


@pytest.fixture(scope="module")
def cross_stack_baseline(cross_stack_payload: dict) -> StrokesGainedBaseline:
    document = cross_stack_payload["baseline"]
    return StrokesGainedBaseline(
        baseline_id=document["baseline_id"],
        version=document["version"],
        source_url=document["source_url"],
        license=document["license"],
        table_sha256=document["table_sha256"],
        states=tuple(
            BaselineState(
                state["lie"],
                state["context"],
                state["target"],
                float(state["distance_yards"]),
                float(state["expected_strokes"]),
                None
                if state.get("standard_error") is None
                else float(state["standard_error"]),
            )
            for state in document["states"]
        ),
    )


def _cross_stack_request(method: str) -> StrokesGainedRequestV1:
    return StrokesGainedRequestV1(
        start=CourseStateColumnsV1(
            lie_column="start_lie",
            context_column="start_context",
            target_column="target",
            distance_column="start_distance_yards",
            distance_unit="yd",
        ),
        finish=CourseStateColumnsV1(
            lie_column="finish_lie",
            context_column="finish_context",
            target_column="target",
            distance_column="finish_distance_metres",
            distance_unit="m",
        ),
        shot_id_column="shot_id",
        summaries=(
            GroupingDimensionV1(
                dimension="player",
                column="player_id",
                trust_level="pseudonymous_stable",
                evidence="Synthetic ADR-0046 G0 fixture pseudonym.",
            ),
        ),
        longitudinal=LongitudinalDimensionV1(
            order_column="session_order",
            order_unit="session",
            group_column="player_id",
            group_dimension="player",
            trust_level="pseudonymous_stable",
            evidence="Synthetic ADR-0046 G0 fixture session ordinal.",
            method=method,  # type: ignore[arg-type]
        ),
    )


# ── P12: the seam the retired baseline half left behind ──────────────────


def test_already_home_baseline_satisfies_the_canonical_protocol() -> None:
    """The already-home loader's artifact flows into the canonical analysis.

    This is the pin that replaces UpstreamDrift's two baseline-half cases. The
    port plan retires that half into
    ``rate_of_closure.launch_monitor_strokes_gained_baseline``; the risk it
    creates is not that the digest is wrong (G0 pinned it identical) but that
    the two halves stop composing. So this asserts composition directly.
    """
    baseline = _baseline()

    assert isinstance(baseline, ExpectedStrokesBaselineLike)
    assert all(isinstance(state, ExpectedStrokesStateLike) for state in baseline.states)
    assert BASELINE_CONTRACT_VERSION == ALREADY_HOME_BASELINE_CONTRACT_VERSION

    result = analyze_source_backed_strokes_gained(_frame(), baseline, _request())

    assert result.status == "available"
    assert result.baseline.table_sha256 == baseline.table_sha256
    assert result.baseline.contract_version == BASELINE_CONTRACT_VERSION


def test_canonical_layer_defines_no_second_baseline_authority() -> None:
    """P12 must not reintroduce the half the plan retired."""
    from shared.python.launch_monitor import strokes_gained_types

    retired = {
        "ExpectedStrokesStateV2",
        "ExpectedStrokesBaselineV2",
        "baseline_table_sha256",
    }

    assert not retired & set(dir(strokes_gained_types))
    assert not retired & set(strokes_gained_types.__all__)


# ── P14: ported UpstreamDrift cases ──────────────────────────────────────


def test_source_backed_sg_reports_traceable_rows_uncertainty_and_summaries() -> None:
    result = analyze_source_backed_strokes_gained(
        _frame(),
        _baseline(),
        _request(),
    )

    assert result.status == "available"
    assert result.metric_name == "source_backed_strokes_gained"
    assert result.unit == "strokes"
    assert result.value_summary.count == 6
    assert result.value_summary.mean == pytest.approx(1.275, abs=1e-4)
    assert result.value_summary.confidence_interval is not None
    assert result.uncertainty.benchmark_method == "interpolated-state-standard-errors"
    assert result.baseline.table_sha256 == _baseline().table_sha256
    assert result.row_results[0].expected_start == pytest.approx(3.3)
    assert result.row_results[0].expected_finish == pytest.approx(1.5)
    assert result.row_results[0].strokes_gained == pytest.approx(0.8)
    assert result.row_results[0].start.target == "hole-1"
    assert result.row_results[0].input_record_sha256 is not None
    assert result.exclusions.total_excluded == 0
    assert {summary.dimension for summary in result.group_summaries} == {
        "player",
        "session",
        "club",
    }
    assert len(result.longitudinal_summaries) == 2
    assert result.longitudinal_summaries[0].slope_unit == "strokes/session"
    assert result.claims.is_strokes_gained is True
    assert result.claims.source_backed is True
    assert result.claims.causal_inference is False
    assert "target/hole" in " ".join(result.limitations).lower()


def test_source_backed_sg_excludes_bad_rows_and_fails_minimum_closed() -> None:
    """G1-D3 in one case: exclude, audit, and still return a result."""
    frame = _frame().iloc[:3].copy()
    frame.loc[0, "start_context"] = ""
    frame.loc[1, "start_distance"] = 250.0

    partial = analyze_source_backed_strokes_gained(
        frame,
        _baseline(),
        _request(min_samples=1),
    )
    assert partial.status == "partial"
    assert partial.value_summary.count == 1
    assert partial.exclusions.by_reason == {
        "missing_course_state": 1,
        "outside_baseline": 1,
    }
    assert len(partial.excluded_rows) == 2

    unavailable = analyze_source_backed_strokes_gained(
        frame,
        _baseline(),
        _request(min_samples=2),
    )
    assert unavailable.status == "unavailable"
    assert unavailable.availability.reason_code == "insufficient_complete_rows"
    assert unavailable.availability.observed_count == 1
    assert unavailable.availability.required_count == 2
    assert unavailable.value_summary.count == 1
    assert unavailable.value_summary.mean is None
    assert unavailable.group_summaries == ()
    assert unavailable.longitudinal_summaries == ()


def test_grouped_and_longitudinal_summaries_require_explicit_evidence() -> None:
    with pytest.raises(ValidationError, match="trust_level"):
        GroupingDimensionV1(
            dimension="player",
            column="player_id",
            trust_level="untrusted_inferred",  # type: ignore[arg-type]
            evidence="Guessed from row order.",
        )
    with pytest.raises(ValidationError, match="evidence"):
        LongitudinalDimensionV1(
            order_column="session_order",
            order_unit="session",
            trust_level="explicit_user_attested",
            evidence="",
        )


def test_launch_monitor_proxy_is_never_labeled_strokes_gained() -> None:
    result = analyze_outcome_proxy(
        pd.DataFrame(
            {
                "carry_m": [137.16, 140.0],
                "lateral_m": [-9.144, 4.572],
            }
        ),
        OutcomeProxyRequestV1(
            carry_column="carry_m",
            lateral_column="lateral_m",
            carry_unit="m",
            lateral_unit="m",
            target_distance_yards=150.0,
        ),
    )

    assert result.metric_name == "expected_proximity_dispersion_proxy"
    assert result.unit == "yd"
    assert result.claims.is_strokes_gained is False
    assert result.claims.source_backed is False
    assert result.row_results[0].lateral_yards == pytest.approx(-10.0)
    assert "not strokes gained" in " ".join(result.limitations).lower()


def test_strokes_gained_schema_is_versioned() -> None:
    schema = strokes_gained_contract_json_schema()
    assert schema["title"] == "StrokesGainedAnalysisResultV1"
    assert schema["properties"]["contract_version"]["const"] == (
        "launch-monitor-strokes-gained-analysis/1.0.0"
    )


def test_generated_schema_is_the_python_authority() -> None:
    """Replaces UpstreamDrift's comparison against its published artifact."""
    schema = strokes_gained_contract_json_schema()
    definitions = schema["$defs"]

    assert schema["additionalProperties"] is False
    assert "method" in definitions["LongitudinalSummaryV1"]["required"]
    assert definitions["LongitudinalSummaryV1"]["properties"]["method"]["enum"] == [
        "session-cell-sg-trend/1",
        "shot-level-sg-trend/1",
    ]
    assert set(definitions["StrokesGainedClaimsV1"]["properties"]) == {
        "is_strokes_gained",
        "source_backed",
        "device_emulation",
        "device_certification",
        "causal_inference",
    }
    assert (
        definitions["StrokesGainedClaimsV1"]["properties"]["causal_inference"]["const"]
        is False
    )


# ── P14: decision G1-D2, shown as a delta ────────────────────────────────


def test_canonical_longitudinal_estimand_is_the_session_cell(
    cross_stack_frame: pd.DataFrame, cross_stack_baseline: StrokesGainedBaseline
) -> None:
    """G1-D2 on the exact fixture G0 measured, both estimands side by side.

    The point estimate barely moves — the fixture is balanced at eight shots
    per session, so the cell means carry the same slope to float noise. What
    moves is the *inference*: forty pseudo-replicated observations reported
    ``p = 0.0121`` where the five real session cells report ``p = 0.1411``.
    That is exactly the pseudo-replication UpstreamDrift's own
    ``longitudinal.py`` warns against and G1-D2 rules out.
    """
    shot_level = analyze_source_backed_strokes_gained(
        cross_stack_frame,
        cross_stack_baseline,
        _cross_stack_request("shot-level-sg-trend/1"),
    )
    session_cell = analyze_source_backed_strokes_gained(
        cross_stack_frame,
        cross_stack_baseline,
        _cross_stack_request("session-cell-sg-trend/1"),
    )

    assert shot_level.value_summary.mean == pytest.approx(G0_SESSION_MEAN, rel=1e-12)
    assert shot_level.exclusions.included_row_count == G0_INCLUDED_ROWS
    assert session_cell.value_summary.mean == shot_level.value_summary.mean

    old = {item.group_value: item for item in shot_level.longitudinal_summaries}
    new = {item.group_value: item for item in session_cell.longitudinal_summaries}
    assert set(old) == set(new) == {"P1", "P2", "P3", "P4"}

    assert all(item.method == "shot-level-sg-trend/1" for item in old.values())
    assert all(item.method == "session-cell-sg-trend/1" for item in new.values())
    assert all(item.sample_count == G0_SHOT_LEVEL_SAMPLE_COUNT for item in old.values())
    assert all(
        item.sample_count == G0_SESSION_CELL_SAMPLE_COUNT for item in new.values()
    )

    assert old["P4"].slope == pytest.approx(G0_SHOT_LEVEL_P4_SLOPE, rel=1e-9)
    assert old["P4"].r_squared == pytest.approx(G0_SHOT_LEVEL_P4_R_SQUARED, rel=1e-9)
    assert old["P4"].p_value == pytest.approx(G0_SHOT_LEVEL_P4_P_VALUE, rel=1e-9)

    assert new["P4"].slope == pytest.approx(G0_SESSION_CELL_P4_SLOPE, rel=1e-9)
    assert new["P4"].r_squared == pytest.approx(G0_SESSION_CELL_P4_R_SQUARED, rel=1e-9)
    assert new["P4"].p_value == pytest.approx(G0_SESSION_CELL_P4_P_VALUE, rel=1e-9)

    # The slope survives; the significance verdict does not.
    assert new["P4"].slope == pytest.approx(old["P4"].slope, rel=1e-12)
    assert old["P4"].p_value < 0.05 < new["P4"].p_value


def test_session_cell_is_the_default_estimand() -> None:
    """A request that says nothing about method gets the canonical one."""
    dimension = LongitudinalDimensionV1(
        order_column="session_order",
        order_unit="session",
        trust_level="explicit_user_attested",
        evidence="The user attested chronological session order.",
    )

    assert dimension.method == "session-cell-sg-trend/1"

    result = analyze_source_backed_strokes_gained(_frame(), _baseline(), _request())
    assert all(
        item.method == "session-cell-sg-trend/1"
        for item in result.longitudinal_summaries
    )


def test_shot_level_variant_is_reachable_and_never_silently_substituted() -> None:
    """G1-D2 preserves the old estimand; it does not delete it."""
    with pytest.raises(ValidationError):
        LongitudinalDimensionV1(
            order_column="session_order",
            order_unit="session",
            trust_level="explicit_user_attested",
            evidence="Attested order.",
            method="unnamed-trend",  # type: ignore[arg-type]
        )


def test_unequal_session_weights_collapse_before_the_slope_is_fitted() -> None:
    """One heavily-sampled session must not outvote the others."""
    frame = pd.DataFrame(
        {
            "shot_id": [f"shot-{index}" for index in range(8)],
            "player_id": ["player-a"] * 8,
            "session_id": ["s1"] * 5 + ["s2", "s3", "s4"],
            "session_order": [1, 1, 1, 1, 1, 2, 3, 4],
            "club": ["7i"] * 8,
            "start_lie": ["fairway"] * 8,
            "start_context": ["standard"] * 8,
            "target": ["hole-1"] * 8,
            "start_distance": [150.0] * 8,
            "finish_lie": ["green"] * 8,
            "finish_context": ["standard"] * 8,
            "finish_distance_m": [18.288, 16.0, 12.0, 14.0, 10.0]
            + [13.716, 9.144, 4.572],
        }
    )
    request = _request().model_copy(
        update={
            "summaries": (),
            "longitudinal": LongitudinalDimensionV1(
                order_column="session_order",
                order_unit="session",
                group_column="player_id",
                group_dimension="player",
                trust_level="pseudonymous_stable",
                evidence="Stable pseudonym and chronological session order.",
            ),
        }
    )

    session_cell = analyze_source_backed_strokes_gained(frame, _baseline(), request)
    shot_level = analyze_source_backed_strokes_gained(
        frame,
        _baseline(),
        request.model_copy(
            update={
                "longitudinal": request.longitudinal.model_copy(  # type: ignore[union-attr]
                    update={"method": "shot-level-sg-trend/1"}
                )
            }
        ),
    )

    assert session_cell.longitudinal_summaries[0].sample_count == 4
    assert shot_level.longitudinal_summaries[0].sample_count == 8
    assert session_cell.longitudinal_summaries[0].slope != pytest.approx(
        shot_level.longitudinal_summaries[0].slope, rel=1e-9
    )


# ── P13: the new target-error gate the plan requires ─────────────────────


def test_outcome_proxy_target_error_gate() -> None:
    """P13's gate: the canonical proxy and its twin agree to delta 0.0.

    The port plan classified ``outcome_proxy`` ``port-up`` on a *reading* —
    "the closed form is character-for-character the same statistic" — while
    still requiring a gate before the port landed. This is that gate. Both
    stacks now live in this repository, so unlike the three G0 files it needs
    no vendored submodule.
    """
    frame = pd.DataFrame(
        {
            "carry_m": [137.16, 140.0, 128.0, 151.4, 133.3],
            "lateral_m": [-9.144, 4.572, 0.0, -2.5, 11.7],
        }
    )

    canonical = analyze_outcome_proxy(
        frame,
        OutcomeProxyRequestV1(
            carry_column="carry_m",
            lateral_column="lateral_m",
            carry_unit="m",
            lateral_unit="m",
            target_distance_yards=150.0,
        ),
    )
    twin = calculate_target_error(
        frame,
        TargetErrorRequest(
            carry_column="carry_m",
            lateral_column="lateral_m",
            carry_unit="m",
            lateral_unit="m",
            target_distance_yards=150.0,
        ),
    )

    canonical_values = [row.radial_error_yards for row in canonical.row_results]
    assert len(canonical_values) == len(twin.values) == len(frame)
    deltas = [
        abs(left - right)
        for left, right in zip(canonical_values, twin.values, strict=True)
    ]
    assert max(deltas) == 0.0
    assert canonical.value_summary.mean is not None
    assert canonical.value_summary.mean - twin.mean == 0.0

    # The superset the plan documented: exclusion accounting, uncertainty,
    # availability, and an explicit "this is not strokes gained" block, none
    # of which the twin's ``ScoreResult`` can express.
    assert canonical.value_summary.confidence_interval is not None
    assert canonical.exclusions.input_row_count == len(frame)
    assert canonical.claims.is_strokes_gained is False
    assert not hasattr(twin, "claims")
    assert not hasattr(twin, "exclusions")


def test_outcome_proxy_excludes_non_numeric_rows_and_audits_them() -> None:
    result = analyze_outcome_proxy(
        pd.DataFrame(
            {
                "carry_yd": [150.0, None, float("inf"), 148.0],
                "lateral_yd": [3.0, 1.0, 2.0, True],
            }
        ),
        OutcomeProxyRequestV1(
            carry_column="carry_yd",
            lateral_column="lateral_yd",
            carry_unit="yd",
            lateral_unit="yd",
            target_distance_yards=150.0,
        ),
    )

    assert result.status == "partial"
    assert result.exclusions.included_row_count == 1
    assert result.exclusions.by_reason == {"missing_or_non_numeric_outcome": 3}


def test_outcome_proxy_is_unavailable_below_the_declared_minimum() -> None:
    result = analyze_outcome_proxy(
        pd.DataFrame({"carry_yd": [150.0], "lateral_yd": [3.0]}),
        OutcomeProxyRequestV1(
            carry_column="carry_yd",
            lateral_column="lateral_yd",
            carry_unit="yd",
            lateral_unit="yd",
            target_distance_yards=150.0,
            min_samples=5,
        ),
    )

    assert result.status == "unavailable"
    assert result.exclusions.included_row_count == 1


# ── Design-by-contract refusal pins ──────────────────────────────────────


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda: analyze_source_backed_strokes_gained(
                [{"start_lie": "fairway"}],  # type: ignore[arg-type]
                _baseline(),
                _request(),
            ),
            id="strokes_gained_rejects_non_frame",
        ),
        pytest.param(
            lambda: analyze_outcome_proxy(
                {"carry_yd": [1.0]},  # type: ignore[arg-type]
                OutcomeProxyRequestV1(
                    carry_column="carry_yd",
                    lateral_column="lateral_yd",
                    carry_unit="yd",
                    lateral_unit="yd",
                    target_distance_yards=150.0,
                ),
            ),
            id="outcome_proxy_rejects_non_frame",
        ),
    ],
)
def test_analyses_refuse_a_non_dataframe(call) -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        call()


def test_analyses_refuse_absent_columns_by_name() -> None:
    with pytest.raises(ValueError, match="Columns not present"):
        analyze_source_backed_strokes_gained(
            _frame().drop(columns=["finish_distance_m"]), _baseline(), _request()
        )
    with pytest.raises(ValueError, match="Columns not present"):
        analyze_outcome_proxy(
            pd.DataFrame({"carry_yd": [150.0]}),
            OutcomeProxyRequestV1(
                carry_column="carry_yd",
                lateral_column="absent",
                carry_unit="yd",
                lateral_unit="yd",
                target_distance_yards=150.0,
            ),
        )


def test_request_refuses_duplicate_summary_dimensions() -> None:
    with pytest.raises(ValidationError, match="unique"):
        StrokesGainedRequestV1(
            start=CourseStateColumnsV1(
                lie_column="a",
                context_column="b",
                target_column="c",
                distance_column="d",
                distance_unit="yd",
            ),
            finish=CourseStateColumnsV1(
                lie_column="e",
                context_column="f",
                target_column="c",
                distance_column="g",
                distance_unit="yd",
            ),
            summaries=(
                GroupingDimensionV1(
                    dimension="player",
                    column="p1",
                    trust_level="pseudonymous_stable",
                    evidence="one",
                ),
                GroupingDimensionV1(
                    dimension="player",
                    column="p2",
                    trust_level="pseudonymous_stable",
                    evidence="two",
                ),
            ),
        )


def test_longitudinal_dimension_refuses_a_half_declared_group() -> None:
    with pytest.raises(ValidationError, match="must be paired"):
        LongitudinalDimensionV1(
            order_column="session_order",
            order_unit="session",
            group_column="player_id",
            trust_level="explicit_user_attested",
            evidence="Attested order.",
        )


def test_analysis_never_mutates_the_caller_frame() -> None:
    frame = _frame()
    before = frame.copy(deep=True)

    analyze_source_backed_strokes_gained(frame, _baseline(), _request())

    pd.testing.assert_frame_equal(frame, before)


def test_outside_baseline_rows_fail_closed_rather_than_extrapolate() -> None:
    frame = _frame()
    frame.loc[:, "start_distance"] = 400.0

    result = analyze_source_backed_strokes_gained(
        frame, _baseline(), _request(min_samples=1)
    )

    assert result.status == "unavailable"
    assert result.exclusions.by_reason == {"outside_baseline": 6}
    assert all(row.reason_code == "outside_baseline" for row in result.excluded_rows)
