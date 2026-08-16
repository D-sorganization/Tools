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

Identification uses the published two-point method: the 28.3% and 63.2%
crossings of the total PV change fall at ``theta + tau/3`` and ``theta + tau``
respectively, so ``tau = 1.5 * (t63 - t28)`` and ``theta = t63 - tau``. This
is unbiased, unlike a 10%/63.2% pair, whose crossing sits at
``theta + 0.105*tau`` and therefore inflates theta and deflates tau by roughly
``0.105 * tau`` each.

Because ``Kc`` is proportional to ``tau / theta``, an under-resolved dead time
makes the recommendation explode. Every identification is therefore guarded:
a result is only reported as ``"success"`` when the crossings are resolvable
at the recorded sample rate, the dead time is genuinely measured rather than
floored, and the outcome sits inside the Cohen-Coon validity band.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median
from typing import Any

# --- FOPDT step-response identification constants ---------------------------
# Lower of the two published two-point fractions. The 28.3% crossing of the
# total PV change occurs at t = theta + tau/3 for a first-order response.
PV_LOW_FRACTION = 0.283
# Fraction of the total PV change that marks one process time constant for a
# first-order response (1 - 1/e = 0.632), i.e. t = theta + tau.
PV_TAU_FRACTION = 0.632
# Two-point solution constants: tau = 1.5 * (t63 - t28), theta = t63 - tau.
TWO_POINT_TAU_FACTOR = 1.5
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
# The first threshold crossing must be at least this many sample intervals
# after the step. A crossing on (or within one sample of) the step itself is
# noise, not dead time, and would drive theta onto MIN_TIME_PARAM.
MIN_IDENT_SAMPLE_SPANS = 2
# Cohen-Coon is published for dead-time ratios in roughly this band. Outside
# it the tau/theta term inflates Kc without physical justification.
COHEN_COON_MIN_RATIO = 0.1
COHEN_COON_MAX_RATIO = 1.0
# Absolute sanity bound on any emitted gain. Nothing beyond this is a
# plausible recommendation for a thermal/power plant on this rig.
MAX_REC_GAIN = 100.0

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

# Valid ``TuningResult.status`` values.
VALID_STATUSES = frozenset({"success", "warning"})

# History samples are ``(time_offset, cv, pv)`` triples.
HistorySample = tuple[float, float, float]


def require_real_number(value: object, name: str) -> float:
    """Return ``value`` as a finite float.

    Shared input guard for the control math in this package.

    Raises
    ------
    TypeError
        If ``value`` is not an ``int`` or ``float`` (``bool`` is rejected).
    ValueError
        If ``value`` is NaN or infinite.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite, got {numeric!r}")
    return numeric


def require_positive_int(value: object, name: str) -> int:
    """Return ``value`` as an ``int`` of at least 1.

    Raises
    ------
    TypeError
        If ``value`` is not an ``int`` (``bool`` is rejected).
    ValueError
        If ``value`` is less than 1.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def _require_bool(value: object, name: str) -> bool:
    """Return ``value`` as a bool, rejecting truthy non-bool input."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _validate_history(history: object) -> list[HistorySample]:
    """Return ``history`` normalised to a list of finite ``(t, cv, pv)`` triples.

    Raises
    ------
    TypeError
        If ``history`` is not a sequence of 3-element numeric samples.
    ValueError
        If any sample value is NaN or infinite.
    """
    if isinstance(history, (str, bytes)) or not isinstance(history, Sequence):
        raise TypeError(
            f"history must be a sequence of samples, got {type(history).__name__}"
        )
    validated: list[HistorySample] = []
    for index, sample in enumerate(history):
        if isinstance(sample, (str, bytes)) or not isinstance(sample, Sequence):
            raise TypeError(
                f"history[{index}] must be a (time, cv, pv) triple, "
                f"got {type(sample).__name__}"
            )
        if len(sample) != 3:
            raise TypeError(
                f"history[{index}] must have exactly 3 elements, got {len(sample)}"
            )
        validated.append(
            (
                require_real_number(sample[0], f"history[{index}].time"),
                require_real_number(sample[1], f"history[{index}].cv"),
                require_real_number(sample[2], f"history[{index}].pv"),
            )
        )
    return validated


@dataclass(frozen=True)
class TuningResult:
    """Identified FOPDT parameters and recommended PID gains.

    ``status`` is ``"success"`` only when a recommendation was produced from a
    resolvable step response that sits inside the Cohen-Coon validity band.
    It is ``"warning"`` for insufficient data, an unresolvable identification,
    a reverse-acting process, or an implausible result. The route layer maps
    this onto its HTTP response shape.

    Recommended gains are reported exactly as computed. They are **not**
    clamped to be non-negative: a reverse-acting plant (identified ``Kp < 0``)
    genuinely tunes to negative Cohen-Coon gains, and zeroing them would
    present an open-loop controller as a valid recommendation.
    """

    status: str
    message: str
    kp: float
    tau: float
    theta: float
    rec_kp: float
    rec_ki: float
    rec_kd: float

    def __post_init__(self) -> None:
        """Validate the result invariants.

        Raises
        ------
        TypeError
            If ``status``/``message`` are not strings or a gain is not numeric.
        ValueError
            If ``status`` is not one of ``VALID_STATUSES`` or a value is
            non-finite.
        """
        if not isinstance(self.status, str):
            raise TypeError(f"status must be a str, got {type(self.status).__name__}")
        if self.status not in VALID_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(VALID_STATUSES)}, got {self.status!r}"
            )
        if not isinstance(self.message, str):
            raise TypeError(f"message must be a str, got {type(self.message).__name__}")
        for field_name in ("kp", "tau", "theta", "rec_kp", "rec_ki", "rec_kd"):
            require_real_number(getattr(self, field_name), field_name)

    def as_response(self) -> dict[str, Any]:
        """Render this result into the route's JSON response shape.

        Gains are rounded for display only. Sign is preserved so a
        reverse-acting recommendation reaches the operator intact.
        """
        return {
            "status": self.status,
            "message": self.message,
            "parameters": {
                "kp": round(self.kp, 3),
                "tau": round(self.tau, 2),
                "theta": round(self.theta, 2),
            },
            "recommended_pid": {
                "kp": round(self.rec_kp, 3),
                "ki": round(self.rec_ki, 3),
                "kd": round(self.rec_kd, 3),
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


def _rejected(message: str, kp: float = 0.0) -> TuningResult:
    """Return a warning result carrying no gain recommendation."""
    return TuningResult(
        status="warning",
        message=message,
        kp=kp,
        tau=0.0,
        theta=0.0,
        rec_kp=0.0,
        rec_ki=0.0,
        rec_kd=0.0,
    )


def _find_crossings(
    history: list[HistorySample],
    *,
    step_time: float,
    threshold_low: float,
    threshold_tau: float,
    rising: bool,
) -> tuple[float | None, float | None]:
    """Return the first post-step ``(t_low, t_tau)`` threshold crossing times.

    ``t_tau`` is only accepted at or after ``t_low`` so that a noisy sample
    cannot order the two crossings backwards.
    """
    t_low: float | None = None
    t_tau: float | None = None
    for time_offset, _, pv_val in history:
        if time_offset < step_time:
            continue
        crossed_low = pv_val >= threshold_low if rising else pv_val <= threshold_low
        crossed_tau = pv_val >= threshold_tau if rising else pv_val <= threshold_tau
        if t_low is None and crossed_low:
            t_low = time_offset
        if t_low is not None and t_tau is None and crossed_tau:
            t_tau = time_offset
    return t_low, t_tau


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

    Performs a published two-point (28.3% / 63.2%) FOPDT identification on the
    recorded step-response ``history`` and applies the Cohen-Coon PID rules.

    The identification is *rejected* (``status="warning"``, zero gains) when it
    cannot be trusted: fewer than two samples, no threshold crossing, a first
    crossing within ``MIN_IDENT_SAMPLE_SPANS`` sample intervals of the step, or
    both crossings landing on the same sample. It is *downgraded* to a warning
    while still reporting gains when the process is reverse-acting, the dead
    time was floored, the step was too small to measure, the dead-time ratio
    sits outside the Cohen-Coon validity band, or a gain exceeds
    ``MAX_REC_GAIN``.

    Parameters
    ----------
    history:
        Recorded ``(time_offset, cv, pv)`` samples for the tuning session.
        Every element must be a 3-element sequence of finite numbers.
    step_triggered:
        Whether a step change was actually executed during the session.
    initial_pv:
        Process value at the moment the step was applied. Must be finite.
    initial_cv, final_cv:
        Control value before and after the step change. Must be finite.
    step_time:
        Time offset (seconds, relative to session start) at which the step
        was applied. Must be finite.

    Raises
    ------
    TypeError
        If any argument has the wrong type.
    ValueError
        If any numeric argument is NaN or infinite.
    """
    samples = _validate_history(history)
    step_triggered = _require_bool(step_triggered, "step_triggered")
    initial_pv = require_real_number(initial_pv, "initial_pv")
    initial_cv = require_real_number(initial_cv, "initial_cv")
    final_cv = require_real_number(final_cv, "final_cv")
    step_time = require_real_number(step_time, "step_time")

    if not samples or not step_triggered:
        return _rejected(
            "Tuning stopped, but no step change was executed or history is empty."
        )

    warnings: list[str] = []

    delta_cv = final_cv - initial_cv
    if abs(delta_cv) < MIN_DELTA_CV:
        warnings.append(
            f"Step size |delta CV| = {abs(delta_cv):.4f} is below the "
            f"{MIN_DELTA_CV} measurable minimum and was normalised to 1.0, so "
            "the identified process gain is not trustworthy."
        )
        delta_cv = 1.0

    n_samples = len(samples)
    last_samples = samples[max(0, n_samples - FINAL_PV_SAMPLE_COUNT) :]
    final_pv = sum(h[2] for h in last_samples) / len(last_samples)
    delta_pv = final_pv - initial_pv

    kp_ident = delta_pv / delta_cv

    if n_samples < 2:
        return _rejected(
            "Tuning stopped, but the step response has too few samples to "
            "identify dead time.",
            kp=kp_ident,
        )

    intervals = [
        later - earlier
        for earlier, later in zip(
            [h[0] for h in samples], [h[0] for h in samples[1:]], strict=False
        )
        if later > earlier
    ]
    if not intervals:
        return _rejected(
            "Tuning stopped, but the step-response timestamps do not advance.",
            kp=kp_ident,
        )
    sample_interval = median(intervals)

    if delta_pv == 0.0:
        return _rejected(
            "Tuning stopped, but the process value never responded to the step.",
            kp=kp_ident,
        )

    threshold_low = initial_pv + PV_LOW_FRACTION * delta_pv
    threshold_tau = initial_pv + PV_TAU_FRACTION * delta_pv

    t_low, t_tau = _find_crossings(
        samples,
        step_time=step_time,
        threshold_low=threshold_low,
        threshold_tau=threshold_tau,
        rising=delta_pv > 0,
    )

    if t_low is None or t_tau is None:
        return _rejected(
            "Tuning stopped, but the process value never crossed the "
            f"{PV_LOW_FRACTION:.1%}/{PV_TAU_FRACTION:.1%} identification "
            "thresholds; no gains can be recommended.",
            kp=kp_ident,
        )

    min_dead_time = MIN_IDENT_SAMPLE_SPANS * sample_interval
    if (t_low - step_time) < min_dead_time:
        return _rejected(
            "Tuning rejected: the process crossed the "
            f"{PV_LOW_FRACTION:.1%} threshold {t_low - step_time:.2f} s after "
            f"the step, less than {MIN_IDENT_SAMPLE_SPANS} sample interval "
            f"({sample_interval:.2f} s) spans. Dead time cannot be resolved "
            "at this sample rate, and an under-resolved dead time inflates "
            "the recommended gain without bound. Increase the sample rate or "
            "the step size and repeat the test.",
            kp=kp_ident,
        )

    if (t_tau - t_low) < sample_interval:
        return _rejected(
            "Tuning rejected: the "
            f"{PV_LOW_FRACTION:.1%} and {PV_TAU_FRACTION:.1%} thresholds were "
            "crossed on the same sample, so the time constant cannot be "
            "resolved.",
            kp=kp_ident,
        )

    # Published two-point solution: t28 = theta + tau/3, t63 = theta + tau.
    tau_raw = TWO_POINT_TAU_FACTOR * (t_tau - t_low)
    theta_raw = (t_tau - step_time) - tau_raw

    tau_ident = max(MIN_TIME_PARAM, tau_raw)
    theta_ident = max(MIN_TIME_PARAM, theta_raw)

    if theta_raw < MIN_TIME_PARAM:
        warnings.append(
            f"Identified dead time ({theta_raw:.2f} s) fell to the "
            f"{MIN_TIME_PARAM} s floor, so the recommended gain is an upper "
            "bound rather than a measurement."
        )

    if abs(kp_ident) <= MIN_PROCESS_GAIN:
        return _rejected(
            f"Tuning rejected: identified process gain |Kp| = {abs(kp_ident):.5f} "
            f"is at or below the {MIN_PROCESS_GAIN} minimum and cannot be "
            "inverted safely.",
            kp=kp_ident,
        )

    rec_kp, rec_ki, rec_kd = cohen_coon_pid(kp_ident, tau_ident, theta_ident)

    if kp_ident < 0.0:
        warnings.append(
            "Reverse-acting process identified (Kp < 0), so the Cohen-Coon "
            "gains are negative. Configure the loop as reverse-acting before "
            "applying them; do not enter negative gains into a direct-acting "
            "loop."
        )

    dead_time_ratio = theta_ident / tau_ident
    if not COHEN_COON_MIN_RATIO <= dead_time_ratio <= COHEN_COON_MAX_RATIO:
        warnings.append(
            f"Dead-time ratio theta/tau = {dead_time_ratio:.3f} is outside the "
            f"Cohen-Coon validity band "
            f"{COHEN_COON_MIN_RATIO}-{COHEN_COON_MAX_RATIO}; the recommendation "
            "is indicative only and must be verified before use."
        )

    largest_gain = max(abs(rec_kp), abs(rec_ki), abs(rec_kd))
    if largest_gain > MAX_REC_GAIN:
        warnings.append(
            f"Recommended gain magnitude {largest_gain:.1f} exceeds the "
            f"{MAX_REC_GAIN} sanity bound; treat the identification as failed."
        )

    if warnings:
        return TuningResult(
            status="warning",
            message=" ".join(warnings),
            kp=kp_ident,
            tau=tau_ident,
            theta=theta_ident,
            rec_kp=rec_kp,
            rec_ki=rec_ki,
            rec_kd=rec_kd,
        )

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
