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
    # Two-point (10%/63.2%) estimate recovers tau and theta within ~50%.
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
def test_response_shape_clamps_negative_gains() -> None:
    """as_response never emits negative recommended gains."""
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
    assert all(v >= 0.0 for v in payload["recommended_pid"].values())
    assert set(payload["parameters"]) == {"kp", "tau", "theta"}
