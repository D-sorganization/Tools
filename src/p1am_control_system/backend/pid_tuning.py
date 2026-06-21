"""Pure FOPDT identification and Cohen-Coon PID tuning math.

This module contains the safety-critical control-tuning arithmetic that was
previously embedded inline in the ``stop_pid_tuning`` FastAPI route in
``main.py``. Extracting it here makes the math importable and unit-testable
with no FastAPI/HTTPException dependency, and gives every Cohen-Coon coefficient
a named, documented constant so a typo cannot silently ship wrong gains to the
P1AM PLC.

The Cohen-Coon PID tuning rules for a first-order-plus-dead-time (FOPDT) plant
with steady-state gain ``Kp``, time constant ``tau`` and dead time ``theta``
(ratio ``r = theta / tau``) are::

    Kc = (1 / Kp) * (tau / theta) * (1.333 + 0.25 * r)
    Ti = theta * (32 + 6 * r) / (13 + 8 * r)
    Td = theta * 4 / (11 + 2 * r)

which are the canonical Cohen-Coon PID coefficients (Cohen & Coon, 1953).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# --- FOPDT step-response identification constants ---------------------------
# Fraction of the total PV change used to mark the "process started moving"
# point. The dead time ``theta`` is estimated as the time from the step to
# this 10% crossing.
PV_LOW_FRACTION = 0.10
# Fraction of the total PV change that marks one process time constant for a
# first-order response (1 - 1/e = 0.632). ``tau`` is estimated from the time
# between the 10% and 63.2% crossings.
PV_TAU_FRACTION = 0.632
# Minimum |delta CV| treated as a real step; below this the CV change is
# considered numerically negligible and normalised to 1.0.
MIN_DELTA_CV = 0.01
# Minimum |Kp| for which a tuning recommendation is produced. Below this the
# identified process gain is too small to invert safely.
MIN_PROCESS_GAIN = 0.001
# Lower bound applied to identified theta/tau (seconds) to avoid divide-by-zero.
MIN_TIME_PARAM = 0.1
# Number of trailing history samples averaged to estimate the final PV.
FINAL_PV_SAMPLE_COUNT = 10

# --- Cohen-Coon PID coefficients --------------------------------------------
CC_KC_BASE = 1.333
CC_KC_RATIO = 0.25
CC_TI_NUM_BASE = 32.0
CC_TI_NUM_RATIO = 6.0
CC_TI_DEN_BASE = 13.0
CC_TI_DEN_RATIO = 8.0
CC_TD_NUM = 4.0
CC_TD_DEN_BASE = 11.0
CC_TD_DEN_RATIO = 2.0

# History samples are ``(time_offset, cv, pv)`` triples.
HistorySample = tuple[float, float, float]


@dataclass(frozen=True)
class TuningResult:
    """Identified FOPDT parameters and recommended PID gains.

    ``status`` is ``"success"`` when a tuning recommendation was produced,
    ``"warning"`` when there was insufficient data (no step / empty history).
    The route layer maps this onto its HTTP response shape.
    """

    status: str
    message: str
    kp: float
    tau: float
    theta: float
    rec_kp: float
    rec_ki: float
    rec_kd: float

    def as_response(self) -> dict[str, Any]:
        """Render this result into the route's JSON response shape."""
        return {
            "status": self.status,
            "message": self.message,
            "parameters": {
                "kp": round(self.kp, 3),
                "tau": round(self.tau, 2),
                "theta": round(self.theta, 2),
            },
            "recommended_pid": {
                "kp": max(0.0, round(self.rec_kp, 3)),
                "ki": max(0.0, round(self.rec_ki, 3)),
                "kd": max(0.0, round(self.rec_kd, 3)),
            },
        }


def cohen_coon_pid(kp: float, tau: float, theta: float) -> tuple[float, float, float]:
    """Return Cohen-Coon ``(Kc, Ki, Kd)`` gains for an FOPDT plant.

    Parameters
    ----------
    kp:
        Identified steady-state process gain. Must be non-zero.
    tau:
        Identified process time constant (seconds). Must be positive.
    theta:
        Identified process dead time (seconds). Must be positive.

    Returns
    -------
    tuple of float
        ``(Kc, Ki, Kd)`` where ``Ki = Kc / Ti`` and ``Kd = Kc * Td`` follow the
        parallel-form PID gains used by the controller.
    """
    if kp == 0.0:
        raise ValueError("process gain kp must be non-zero")
    if tau <= 0.0:
        raise ValueError("process time constant tau must be positive")
    if theta <= 0.0:
        raise ValueError("process dead time theta must be positive")

    ratio = theta / tau
    kc = (1.0 / kp) * (tau / theta) * (CC_KC_BASE + CC_KC_RATIO * ratio)
    ti = (
        theta
        * (CC_TI_NUM_BASE + CC_TI_NUM_RATIO * ratio)
        / (CC_TI_DEN_BASE + CC_TI_DEN_RATIO * ratio)
    )
    td = theta * CC_TD_NUM / (CC_TD_DEN_BASE + CC_TD_DEN_RATIO * ratio)

    kc_gain = kc
    ki_gain = kc / ti
    kd_gain = kc * td
    return kc_gain, ki_gain, kd_gain


def identify_fopdt_and_tune(
    history: list[HistorySample],
    *,
    step_triggered: bool,
    initial_pv: float,
    initial_cv: float,
    final_cv: float,
    step_time: float,
) -> TuningResult:
    """Identify FOPDT parameters from a step response and tune via Cohen-Coon.

    This is the pure extraction of the math previously inline in the
    ``stop_pid_tuning`` route. It performs a two-point (10% / 63.2%) FOPDT
    identification on the recorded step-response ``history`` and applies the
    Cohen-Coon PID rules.

    Parameters
    ----------
    history:
        Recorded ``(time_offset, cv, pv)`` samples for the tuning session.
    step_triggered:
        Whether a step change was actually executed during the session.
    initial_pv:
        Process value at the moment the step was applied.
    initial_cv, final_cv:
        Control value before and after the step change.
    step_time:
        Time offset (seconds, relative to session start) at which the step
        was applied.
    """
    if not history or not step_triggered:
        return TuningResult(
            status="warning",
            message=(
                "Tuning stopped, but no step change was executed or history is empty."
            ),
            kp=0.0,
            tau=0.0,
            theta=0.0,
            rec_kp=0.0,
            rec_ki=0.0,
            rec_kd=0.0,
        )

    delta_cv = final_cv - initial_cv
    if abs(delta_cv) < MIN_DELTA_CV:
        delta_cv = 1.0

    n_samples = len(history)
    last_samples = history[max(0, n_samples - FINAL_PV_SAMPLE_COUNT) :]
    final_pv = sum(h[2] for h in last_samples) / len(last_samples)
    delta_pv = final_pv - initial_pv

    kp_ident = delta_pv / delta_cv

    threshold_10 = initial_pv + PV_LOW_FRACTION * delta_pv
    threshold_63 = initial_pv + PV_TAU_FRACTION * delta_pv

    t_10: float | None = None
    t_63: float | None = None

    for time_offset, _, pv_val in history:
        if time_offset < step_time:
            continue
        if t_10 is None and (
            (delta_pv > 0 and pv_val >= threshold_10)
            or (delta_pv < 0 and pv_val <= threshold_10)
        ):
            t_10 = time_offset
        if t_63 is None and (
            (delta_pv > 0 and pv_val >= threshold_63)
            or (delta_pv < 0 and pv_val <= threshold_63)
        ):
            t_63 = time_offset

    if t_10 is None:
        t_10 = step_time + 1.0
    if t_63 is None:
        t_63 = t_10 + 2.0

    theta_ident = max(MIN_TIME_PARAM, t_10 - step_time)
    tau_ident = max(MIN_TIME_PARAM, t_63 - t_10)

    if abs(kp_ident) > MIN_PROCESS_GAIN:
        rec_kp, rec_ki, rec_kd = cohen_coon_pid(kp_ident, tau_ident, theta_ident)
    else:
        rec_kp = rec_ki = rec_kd = 0.0

    return TuningResult(
        status="success",
        message="Tuning parameters identified successfully.",
        kp=kp_ident,
        tau=tau_ident,
        theta=theta_ident,
        rec_kp=rec_kp,
        rec_ki=rec_ki,
        rec_kd=rec_kd,
    )
