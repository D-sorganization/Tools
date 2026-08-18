"""Numerical tests for the pure Cohen-Coon PID tuning math (pid_tuning.py).

These guard the safety-critical tuning arithmetic that recommends PID gains
pushed to the P1AM PLC. They pin the Cohen-Coon coefficients against the
reference formulas and verify FOPDT identification recovers a known plant.
"""

from __future__ import annotations

import math

import pytest
from pid_tuning import (
    CC_KC_BASE,
    CC_KC_RATIO,
    CC_TD_DEN_BASE,
    CC_TD_DEN_RATIO,
    CC_TD_NUM,
    CC_TI_DEN_BASE,
    CC_TI_DEN_RATIO,
    CC_TI_NUM_BASE,
    CC_TI_NUM_RATIO,
    MIN_TIME_PARAM,
    TuningResult,
    cohen_coon_pid,
    identify_fopdt_and_tune,
)


def _reference_cohen_coon(
    kp: float, tau: float, theta: float
) -> tuple[float, float, float]:
    """Independent re-derivation of the Cohen-Coon PID gains."""
    r = theta / tau
    kc = (1.0 / kp) * (tau / theta) * (CC_KC_BASE + CC_KC_RATIO * r)
    ti = (
        theta
        * (CC_TI_NUM_BASE + CC_TI_NUM_RATIO * r)
        / (CC_TI_DEN_BASE + CC_TI_DEN_RATIO * r)
    )
    td = theta * CC_TD_NUM / (CC_TD_DEN_BASE + CC_TD_DEN_RATIO * r)
    return kc, kc / ti, kc * td


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kp", "tau", "theta"),
    [
        (2.0, 10.0, 2.0),
        (1.0, 5.0, 1.0),
        (0.5, 20.0, 4.0),
        (-1.5, 8.0, 3.0),
    ],
)
def test_cohen_coon_matches_reference(kp: float, tau: float, theta: float) -> None:
    """Cohen-Coon gains equal the canonical FOPDT formulas to full precision."""
    kc, ki, kd = cohen_coon_pid(kp, tau, theta)
    ref_kc, ref_ki, ref_kd = _reference_cohen_coon(kp, tau, theta)
    assert kc == pytest.approx(ref_kc, rel=1e-12)
    assert ki == pytest.approx(ref_ki, rel=1e-12)
    assert kd == pytest.approx(ref_kd, rel=1e-12)


@pytest.mark.unit
def test_cohen_coon_pinned_values() -> None:
    """Pin a concrete known plant so a constant typo is caught.

    Kp=2, tau=10, theta=2 -> ratio=0.2.
    Kc = (1/2)*(10/2)*(1.333 + 0.25*0.2) = 2.5 * 1.383 = 3.4575
    Ti = 2*(32 + 6*0.2)/(13 + 8*0.2) = 2*33.2/14.6 = 4.547945...
    Td = 2*4/(11 + 2*0.2) = 8/11.4 = 0.701754...
    """
    kc, ki, kd = cohen_coon_pid(2.0, 10.0, 2.0)
    assert kc == pytest.approx(3.4575, abs=1e-9)
    assert ki == pytest.approx(3.4575 / 4.547945205479452, rel=1e-9)
    assert kd == pytest.approx(3.4575 * 0.7017543859649122, rel=1e-9)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kp", "tau", "theta"),
    [(0.0, 10.0, 2.0), (2.0, 0.0, 2.0), (2.0, 10.0, 0.0)],
)
def test_cohen_coon_rejects_degenerate(kp: float, tau: float, theta: float) -> None:
    with pytest.raises(ValueError):
        cohen_coon_pid(kp, tau, theta)


def _synthetic_step_history(
    kp: float,
    tau: float,
    theta: float,
    *,
    initial_pv: float,
    delta_cv: float,
    dt: float = 0.5,
    duration: float = 100.0,
) -> list[tuple[float, float, float]]:
    """Sample a noise-free FOPDT step response into (t, cv, pv) triples."""
    final_cv = delta_cv
    history: list[tuple[float, float, float]] = []
    n = int(duration / dt)
    for i in range(n):
        t = i * dt
        if t < theta:
            pv = initial_pv
        else:
            pv = initial_pv + kp * delta_cv * (1.0 - math.exp(-(t - theta) / tau))
        history.append((t, final_cv, pv))
    return history


@pytest.mark.unit
def test_identify_recovers_process_gain() -> None:
    """A clean first-order step recovers Kp exactly and gives positive gains."""
    kp, tau, theta = 2.0, 10.0, 3.0
    initial_pv, delta_cv = 20.0, 5.0
    history = _synthetic_step_history(
        kp, tau, theta, initial_pv=initial_pv, delta_cv=delta_cv
    )

    result = identify_fopdt_and_tune(
        history,
        step_triggered=True,
        initial_pv=initial_pv,
        initial_cv=0.0,
        final_cv=delta_cv,
        step_time=0.0,
    )

    assert result.status == "success"
    # Process gain is the most robustly identified parameter.
    assert result.kp == pytest.approx(kp, rel=0.02)
    # Two-point (28.3%/63.2%) estimate recovers tau and theta within ~50%.
    assert result.tau == pytest.approx(tau, rel=0.5)
    assert result.theta == pytest.approx(theta, abs=2.0)
    # Recommended gains must be positive and match Cohen-Coon for the
    # *identified* parameters (i.e. the route applies the right formula).
    ref_kc, ref_ki, ref_kd = cohen_coon_pid(result.kp, result.tau, result.theta)
    assert result.rec_kp == pytest.approx(ref_kc, rel=1e-9)
    assert result.rec_ki == pytest.approx(ref_ki, rel=1e-9)
    assert result.rec_kd == pytest.approx(ref_kd, rel=1e-9)
    assert result.rec_kp > 0.0


@pytest.mark.unit
def test_identify_no_step_returns_warning() -> None:
    result = identify_fopdt_and_tune(
        [(0.0, 0.0, 20.0)],
        step_triggered=False,
        initial_pv=20.0,
        initial_cv=0.0,
        final_cv=0.0,
        step_time=0.0,
    )
    assert result.status == "warning"
    assert result.rec_kp == 0.0
    assert result.rec_ki == 0.0
    assert result.rec_kd == 0.0


@pytest.mark.unit
def test_identify_empty_history_returns_warning() -> None:
    result = identify_fopdt_and_tune(
        [],
        step_triggered=True,
        initial_pv=20.0,
        initial_cv=0.0,
        final_cv=5.0,
        step_time=0.0,
    )
    assert result.status == "warning"


@pytest.mark.unit
def test_response_shape_is_stable() -> None:
    """as_response always emits the same key structure."""
    result = identify_fopdt_and_tune(
        [],
        step_triggered=False,
        initial_pv=0.0,
        initial_cv=0.0,
        final_cv=0.0,
        step_time=0.0,
    )
    payload = result.as_response()
    assert set(payload["recommended_pid"]) == {"kp", "ki", "kd"}
    assert set(payload["parameters"]) == {"kp", "tau", "theta"}


@pytest.mark.unit
def test_response_preserves_negative_gains() -> None:
    """as_response must NOT clamp negative gains to zero.

    A reverse-acting process legitimately yields negative Cohen-Coon gains.
    Clamping them to zero silently converts the recommendation into an
    open-loop controller (kp=ki=kd=0) while still rendering a result, so the
    negative values must survive the response mapping untouched.
    """
    result = TuningResult(
        status="warning",
        message=(
            "Reverse-acting process identified (Kp < 0); configure the loop "
            "reverse-acting before applying these gains."
        ),
        kp=-2.0,
        tau=9.75,
        theta=3.25,
        rec_kp=-2.125,
        rec_ki=-0.4,
        rec_kd=-1.5,
    )
    payload = result.as_response()

    assert payload["recommended_pid"]["kp"] == pytest.approx(-2.125)
    assert payload["recommended_pid"]["ki"] == pytest.approx(-0.4)
    assert payload["recommended_pid"]["kd"] == pytest.approx(-1.5)
    assert payload["status"] == "warning"
    assert "reverse-acting" in payload["message"].lower()


@pytest.mark.unit
def test_reverse_acting_process_is_flagged_not_zeroed() -> None:
    """A negative identified Kp yields negative gains and a warning status."""
    kp, tau, theta = -2.0, 10.0, 3.0
    initial_pv, delta_cv = 20.0, 5.0
    history = _synthetic_step_history(
        kp, tau, theta, initial_pv=initial_pv, delta_cv=delta_cv
    )

    result = identify_fopdt_and_tune(
        history,
        step_triggered=True,
        initial_pv=initial_pv,
        initial_cv=0.0,
        final_cv=delta_cv,
        step_time=0.0,
    )

    assert result.kp < 0.0
    assert result.rec_kp < 0.0
    assert result.rec_ki < 0.0
    assert result.rec_kd < 0.0
    # An open-loop process must never be reported as "tuned successfully".
    assert result.status == "warning"
    assert "reverse-acting" in result.message.lower()
    # And the negative gains survive into the wire response.
    assert result.as_response()["recommended_pid"]["kp"] < 0.0


@pytest.mark.unit
def test_two_point_identification_uses_28_percent_pair() -> None:
    """The 28.3%/63.2% pair recovers tau and theta without the 10% bias.

    For an FOPDT plant the 28.3% and 63.2% crossings sit at ``theta+tau/3``
    and ``theta+tau``, giving ``tau = 1.5*(t63-t28)`` and ``theta = t63-tau``.
    The old 10% pair instead biased theta high and tau low by ~0.105*tau: for
    this plant it reported theta=4.5 (true 3.0, +50%) and tau=8.5 (-15%).
    """
    kp, tau, theta = 2.0, 10.0, 3.0
    initial_pv, delta_cv = 20.0, 5.0
    history = _synthetic_step_history(
        kp, tau, theta, initial_pv=initial_pv, delta_cv=delta_cv
    )

    result = identify_fopdt_and_tune(
        history,
        step_triggered=True,
        initial_pv=initial_pv,
        initial_cv=0.0,
        final_cv=delta_cv,
        step_time=0.0,
    )

    assert result.status == "success"
    # Tight tolerances the biased 10% pair cannot meet.
    assert result.tau == pytest.approx(tau, rel=0.05)
    assert result.theta == pytest.approx(theta, abs=0.4)


@pytest.mark.unit
def test_noise_spike_at_step_sample_is_rejected() -> None:
    """A first crossing inside 2 sample intervals must not become a tuning.

    Noise landing the first threshold crossing on the step sample itself
    drives the identified dead time onto the MIN_TIME_PARAM floor. Because
    Kc is proportional to tau/theta, the recommendation then explodes -- this
    plant needs Kc~2.1 but the floored theta yields Kc~87, previously
    returned as "success" and offered for download to the PLC.
    """
    kp, tau, theta = 2.0, 10.0, 3.0
    initial_pv, delta_cv = 20.0, 5.0
    history = _synthetic_step_history(
        kp, tau, theta, initial_pv=initial_pv, delta_cv=delta_cv
    )
    # Single-sample noise spike at the instant of the step: 24.0 already sits
    # above both the 10% (21.0) and 28.3% (22.83) thresholds.
    history[0] = (history[0][0], history[0][1], 24.0)

    result = identify_fopdt_and_tune(
        history,
        step_triggered=True,
        initial_pv=initial_pv,
        initial_cv=0.0,
        final_cv=delta_cv,
        step_time=0.0,
    )

    assert result.status == "warning"
    assert result.rec_kp == 0.0
    assert result.rec_ki == 0.0
    assert result.rec_kd == 0.0
    assert "sample interval" in result.message.lower()


@pytest.mark.unit
def test_floored_dead_time_downgrades_to_warning() -> None:
    """theta landing on the MIN_TIME_PARAM floor can never be a success.

    The two crossings are far enough apart that ``tau = 1.5*(t63-t28)``
    overshoots ``t63``, driving the two-point dead time negative. The floor
    then makes Kc an upper bound rather than a measurement, so the result must
    not be presented as a successful tuning.
    """
    history: list[tuple[float, float, float]] = []
    for i in range(41):
        t = i * 0.5
        pv = 20.0 if t < 1.0 else (22.0 if t < 10.0 else 25.0)
        history.append((t, 5.0, pv))

    result = identify_fopdt_and_tune(
        history,
        step_triggered=True,
        initial_pv=20.0,
        initial_cv=0.0,
        final_cv=5.0,
        step_time=0.0,
    )
    assert result.theta == pytest.approx(MIN_TIME_PARAM)
    assert result.status == "warning"
    assert "floor" in result.message.lower()


@pytest.mark.unit
def test_implausible_gain_is_not_reported_as_success() -> None:
    """A lag-dominant plant outside Cohen-Coon validity warns, not succeeds.

    Cohen-Coon is published for dead-time ratios roughly 0.1 <= theta/tau
    <= 1. Well below that the tau/theta term inflates Kc without bound, so
    the result must not be presented as ready to apply.
    """
    kp, tau, theta = 2.0, 40.0, 0.5
    initial_pv, delta_cv = 20.0, 5.0
    history = _synthetic_step_history(
        kp, tau, theta, initial_pv=initial_pv, delta_cv=delta_cv, duration=200.0
    )

    result = identify_fopdt_and_tune(
        history,
        step_triggered=True,
        initial_pv=initial_pv,
        initial_cv=0.0,
        final_cv=delta_cv,
        step_time=0.0,
    )

    assert result.theta / result.tau < 0.1
    assert result.status == "warning"
    # The gains are still reported so an engineer can inspect them.
    assert result.rec_kp != 0.0


@pytest.mark.unit
def test_no_threshold_crossing_is_rejected() -> None:
    """A PV that never responds must not fabricate a tuning."""
    history = [(i * 0.5, 5.0, 20.0) for i in range(40)]
    result = identify_fopdt_and_tune(
        history,
        step_triggered=True,
        initial_pv=20.0,
        initial_cv=0.0,
        final_cv=5.0,
        step_time=0.0,
    )
    assert result.status == "warning"
    assert result.rec_kp == 0.0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("initial_pv", "20"),
        ("initial_cv", None),
        ("final_cv", [5.0]),
        ("step_time", "0"),
        ("step_triggered", 1),
    ],
)
def test_identify_rejects_wrong_types(field: str, value: object) -> None:
    kwargs: dict[str, object] = {
        "step_triggered": True,
        "initial_pv": 20.0,
        "initial_cv": 0.0,
        "final_cv": 5.0,
        "step_time": 0.0,
    }
    kwargs[field] = value
    with pytest.raises(TypeError):
        identify_fopdt_and_tune([(0.0, 0.0, 20.0)], **kwargs)


@pytest.mark.unit
def test_identify_rejects_malformed_history() -> None:
    with pytest.raises(TypeError):
        identify_fopdt_and_tune(
            [(0.0, 0.0)],
            step_triggered=True,
            initial_pv=20.0,
            initial_cv=0.0,
            final_cv=5.0,
            step_time=0.0,
        )


@pytest.mark.unit
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_identify_rejects_non_finite(bad: float) -> None:
    with pytest.raises(ValueError):
        identify_fopdt_and_tune(
            [(0.0, 0.0, 20.0)],
            step_triggered=True,
            initial_pv=bad,
            initial_cv=0.0,
            final_cv=5.0,
            step_time=0.0,
        )


@pytest.mark.unit
def test_tuning_result_rejects_unknown_status() -> None:
    with pytest.raises(ValueError):
        TuningResult(
            status="ok",
            message="x",
            kp=1.0,
            tau=1.0,
            theta=1.0,
            rec_kp=1.0,
            rec_ki=1.0,
            rec_kd=1.0,
        )
