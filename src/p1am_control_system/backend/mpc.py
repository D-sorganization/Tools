"""Dynamic Matrix Control (DMC) move computation and PID-vs-MPC simulation.

The MPC used by ``/api/mpc/simulate`` is a textbook unconstrained DMC: over a
prediction horizon ``P`` the predicted process value is

    y_pred = free_response + G @ delta_u

where ``free_response`` is the trajectory the process would follow if the
current control value were simply *held*, ``G`` is the step-response dynamic
matrix, and ``delta_u`` is the vector of ``C`` future control *moves*. The
decision variable is therefore the move vector, never the absolute control
value -- ``free_response`` already contains the full effect of holding
``last_cv``, so optimising over absolute CV counts the current input twice.
"""

from __future__ import annotations

import math
from typing import Any, Protocol

# Numeric input guards live in pid_tuning (already this module's dependency)
# so the control package validates its arithmetic inputs in exactly one place.
from pid_tuning import cohen_coon_pid, require_positive_int, require_real_number

# --- Simulation constants ----------------------------------------------------
# Fixed simulation sample interval (seconds) and horizon for the comparison
# chart rendered by the frontend.
SIM_DT = 0.5
SIM_STEPS = 50

# --- Control value limits ----------------------------------------------------
# The P1AM analog output is a 0-100% control value.
CV_MIN = 0.0
CV_MAX = 100.0
# Largest single control move (percent of CV span) the optimiser may request.
# Bounding the *move* is what keeps a DMC solution physically realisable; the
# absolute CV limits are enforced separately when the move is integrated.
MAX_MOVE = 100.0

# --- Gradient-descent solver constants ---------------------------------------
# The quadratic DMC objective is minimised by projected gradient descent.
GD_ITERATIONS = 100
GD_STEP_SCALE = 0.01


class MPCSimulationPayload(Protocol):
    prediction_horizon: int
    control_horizon: int
    setpoint: float
    rho: float
    process_gain: float
    process_tau: float
    process_delay: float


def solve_dmc_move(
    *,
    pv: float,
    last_cv: float,
    setpoint: float,
    process_gain: float,
    process_tau: float,
    process_delay: float,
    dt: float,
    prediction_horizon: int,
    control_horizon: int,
    rho: float,
) -> float:
    """Return the first control *move* (delta-u) of the optimal DMC solution.

    The returned value is an increment to be added to ``last_cv`` by the
    caller, which is also responsible for enforcing the absolute CV limits.
    At steady state -- ``pv == setpoint`` and ``last_cv`` already holding the
    process there -- the optimal move is zero.

    Parameters
    ----------
    pv:
        Current process value.
    last_cv:
        Control value currently applied and assumed held over the horizon.
    setpoint:
        Target process value over the whole prediction horizon.
    process_gain, process_tau, process_delay:
        FOPDT model used to build the step-response dynamic matrix. ``tau``
        must be positive; ``process_delay`` must be non-negative.
    dt:
        Sample interval in seconds. Must be positive.
    prediction_horizon, control_horizon:
        Number of predicted samples and free moves. Both must be >= 1 and
        ``control_horizon`` must not exceed ``prediction_horizon``.
    rho:
        Move-suppression weight. Must be non-negative.

    Raises
    ------
    TypeError
        If any argument has the wrong type.
    ValueError
        If any argument is outside its documented range.
    """
    pv = require_real_number(pv, "pv")
    last_cv = require_real_number(last_cv, "last_cv")
    setpoint = require_real_number(setpoint, "setpoint")
    process_gain = require_real_number(process_gain, "process_gain")
    process_tau = require_real_number(process_tau, "process_tau")
    process_delay = require_real_number(process_delay, "process_delay")
    dt = require_real_number(dt, "dt")
    rho = require_real_number(rho, "rho")
    prediction_horizon = require_positive_int(prediction_horizon, "prediction_horizon")
    control_horizon = require_positive_int(control_horizon, "control_horizon")

    if process_tau <= 0.0:
        raise ValueError(f"process_tau must be positive, got {process_tau}")
    if process_delay < 0.0:
        raise ValueError(f"process_delay must be non-negative, got {process_delay}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if rho < 0.0:
        raise ValueError(f"rho must be non-negative, got {rho}")
    if control_horizon > prediction_horizon:
        raise ValueError(
            "control_horizon must not exceed prediction_horizon "
            f"({control_horizon} > {prediction_horizon})"
        )

    # Unit step-response coefficients g[j] = y(t_j) for a unit CV step.
    step_coeffs = []
    for j in range(1, prediction_horizon + 1):
        t_eval = j * dt - process_delay
        if t_eval <= 0:
            step_coeffs.append(0.0)
        else:
            step_coeffs.append(process_gain * (1.0 - math.exp(-t_eval / process_tau)))

    dynamic_matrix = [
        [
            step_coeffs[row - col] if row >= col else 0.0
            for col in range(control_horizon)
        ]
        for row in range(prediction_horizon)
    ]

    # Trajectory the process follows if ``last_cv`` is simply held. This
    # already accounts for the current input, which is why the decision
    # variable below is a move vector and not an absolute CV.
    free_response = []
    for j in range(1, prediction_horizon + 1):
        t_eval = j * dt - process_delay
        if t_eval <= 0:
            free_response.append(pv)
        else:
            free_response.append(
                pv
                + (process_gain * last_cv - pv)
                * (1.0 - math.exp(-t_eval / process_tau))
            )

    target = [setpoint] * prediction_horizon

    gtg = [[0.0] * control_horizon for _ in range(control_horizon)]
    for row_idx in range(control_horizon):
        for col_idx in range(control_horizon):
            gtg[row_idx][col_idx] = sum(
                dynamic_matrix[p_idx][row_idx] * dynamic_matrix[p_idx][col_idx]
                for p_idx in range(prediction_horizon)
            )

    hessian = [
        [2.0 * gtg[row_idx][col_idx] for col_idx in range(control_horizon)]
        for row_idx in range(control_horizon)
    ]
    for i in range(control_horizon):
        hessian[i][i] += 2.0 * rho

    gradient_bias = [0.0] * control_horizon
    for col_idx in range(control_horizon):
        gradient_bias[col_idx] = 2.0 * sum(
            dynamic_matrix[p_idx][col_idx] * (free_response[p_idx] - target[p_idx])
            for p_idx in range(prediction_horizon)
        )

    # Decision variable is the MOVE vector: the free response above already
    # carries the effect of holding ``last_cv``, so starting from the absolute
    # CV would count the current input twice. Zero moves is the correct origin.
    delta_u = [0.0] * control_horizon
    alpha = GD_STEP_SCALE / (
        2.0 * (sum(sum(abs(x) for x in row) for row in gtg) + rho + 1.0)
    )

    for _ in range(GD_ITERATIONS):
        grad = [0.0] * control_horizon
        for row_idx in range(control_horizon):
            grad[row_idx] = (
                sum(
                    hessian[row_idx][col_idx] * delta_u[col_idx]
                    for col_idx in range(control_horizon)
                )
                + gradient_bias[row_idx]
            )

        # Project onto the move limits, not the absolute CV limits.
        for i in range(control_horizon):
            delta_u[i] = max(-MAX_MOVE, min(MAX_MOVE, delta_u[i] - alpha * grad[i]))

    return delta_u[0]


def simulate_pid_vs_mpc(payload: MPCSimulationPayload) -> dict[str, Any]:
    """Simulate and compare PID and MPC response for a first-order process."""
    kp_process = payload.process_gain
    tau = payload.process_tau
    theta = payload.process_delay
    dt = SIM_DT
    steps = SIM_STEPS

    # Reuse the canonical Cohen-Coon coefficients from pid_tuning so the MPC
    # comparison baseline can never drift from the live tuning recommendation.
    pid_kp, pid_ki, pid_kd = cohen_coon_pid(kp_process, max(0.5, tau), max(0.1, theta))

    pid_pv = [0.0] * steps
    pid_cv = [0.0] * steps
    pid_integral = 0.0
    pid_prev_err = 0.0
    cv_hist_pid = [0.0] * steps

    for k in range(1, steps):
        err = payload.setpoint - pid_pv[k - 1]
        pid_integral = max(-100.0, min(100.0, pid_integral + err * dt))
        deriv = (err - pid_prev_err) / dt
        pid_prev_err = err

        cv = pid_kp * err + pid_ki * pid_integral + pid_kd * deriv
        cv = max(CV_MIN, min(CV_MAX, cv))
        pid_cv[k] = cv
        cv_hist_pid[k] = cv

        delay_idx = k - int(theta / dt)
        delayed_cv = cv_hist_pid[max(0, delay_idx)]
        dy = (kp_process * delayed_cv - pid_pv[k - 1]) * (dt / tau)
        pid_pv[k] = max(0.0, pid_pv[k - 1] + dy)

    mpc_pv = [0.0] * steps
    mpc_cv = [0.0] * steps
    cv_hist_mpc = [0.0] * steps

    for k in range(1, steps):
        last_u = mpc_cv[k - 1]
        move = solve_dmc_move(
            pv=mpc_pv[k - 1],
            last_cv=last_u,
            setpoint=payload.setpoint,
            process_gain=kp_process,
            process_tau=tau,
            process_delay=theta,
            dt=dt,
            prediction_horizon=payload.prediction_horizon,
            control_horizon=payload.control_horizon,
            rho=payload.rho,
        )
        mpc_cv[k] = max(CV_MIN, min(CV_MAX, last_u + move))
        cv_hist_mpc[k] = mpc_cv[k]

        delay_idx = k - int(theta / dt)
        delayed_cv = cv_hist_mpc[max(0, delay_idx)]
        dy = (kp_process * delayed_cv - mpc_pv[k - 1]) * (dt / tau)
        mpc_pv[k] = max(0.0, mpc_pv[k - 1] + dy)

    time_series = [round(i * dt, 1) for i in range(steps)]
    return {
        "status": "success",
        "time": time_series,
        "pid": {
            "pv": [round(x, 2) for x in pid_pv],
            "cv": [round(x, 2) for x in pid_cv],
        },
        "mpc": {
            "pv": [round(x, 2) for x in mpc_pv],
            "cv": [round(x, 2) for x in mpc_cv],
        },
    }
