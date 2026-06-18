import math
from typing import Any, Protocol


class MPCSimulationPayload(Protocol):
    prediction_horizon: int
    control_horizon: int
    setpoint: float
    rho: float
    process_gain: float
    process_tau: float
    process_delay: float


def simulate_pid_vs_mpc(payload: MPCSimulationPayload) -> dict[str, Any]:
    """Simulate and compare PID and MPC response for a first-order process."""
    kp_process = payload.process_gain
    tau = payload.process_tau
    theta = payload.process_delay
    dt = 0.5
    steps = 50

    ratio = max(0.1, theta) / max(0.5, tau)
    kc = (1.0 / kp_process) * (tau / max(0.1, theta)) * (1.333 + 0.25 * ratio)
    ti = max(0.1, theta) * (32.0 + 6.0 * ratio) / (13.0 + 8.0 * ratio)
    td = max(0.1, theta) * 4.0 / (11.0 + 2.0 * ratio)
    pid_kp = kc
    pid_ki = kc / ti
    pid_kd = kc * td

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
        cv = max(0.0, min(100.0, cv))
        pid_cv[k] = cv
        cv_hist_pid[k] = cv

        delay_idx = k - int(theta / dt)
        delayed_cv = cv_hist_pid[max(0, delay_idx)]
        dy = (kp_process * delayed_cv - pid_pv[k - 1]) * (dt / tau)
        pid_pv[k] = max(0.0, pid_pv[k - 1] + dy)

    mpc_pv = [0.0] * steps
    mpc_cv = [0.0] * steps
    cv_hist_mpc = [0.0] * steps

    prediction_horizon = payload.prediction_horizon
    control_horizon = payload.control_horizon

    for k in range(1, steps):
        g = []
        for j in range(1, prediction_horizon + 1):
            t_eval = j * dt - theta
            if t_eval <= 0:
                g.append(0.0)
            else:
                g.append(kp_process * (1.0 - math.exp(-t_eval / tau)))

        dynamic_matrix = [
            [g[row - col] if row >= col else 0.0 for col in range(control_horizon)]
            for row in range(prediction_horizon)
        ]

        free_response = []
        last_u = mpc_cv[k - 1]
        for j in range(1, prediction_horizon + 1):
            t_eval = j * dt - theta
            if t_eval <= 0:
                free_response.append(mpc_pv[k - 1])
            else:
                free_response.append(
                    mpc_pv[k - 1]
                    + (kp_process * last_u - mpc_pv[k - 1])
                    * (1.0 - math.exp(-t_eval / tau))
                )

        target = [payload.setpoint] * prediction_horizon

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
            hessian[i][i] += 2.0 * payload.rho

        gradient_bias = [0.0] * control_horizon
        for col_idx in range(control_horizon):
            gradient_bias[col_idx] = 2.0 * sum(
                dynamic_matrix[p_idx][col_idx] * (free_response[p_idx] - target[p_idx])
                for p_idx in range(prediction_horizon)
            )

        u_opt = [last_u] * control_horizon
        alpha = 0.01 / (
            2.0 * (sum(sum(abs(x) for x in row) for row in gtg) + payload.rho + 1.0)
        )

        for _ in range(100):
            grad = [0.0] * control_horizon
            for row_idx in range(control_horizon):
                grad[row_idx] = (
                    sum(
                        hessian[row_idx][col_idx] * u_opt[col_idx]
                        for col_idx in range(control_horizon)
                    )
                    + gradient_bias[row_idx]
                )

            for i in range(control_horizon):
                u_opt[i] = max(0.0, min(100.0, u_opt[i] - alpha * grad[i]))

        mpc_cv[k] = u_opt[0]
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
