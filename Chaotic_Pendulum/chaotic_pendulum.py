import matplotlib
import numpy as np

matplotlib.use("TkAgg")
import argparse
import logging
import sys
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse
from scipy.integrate import solve_ivp

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def equations_of_motion(
    t: float, state: list[float], m1: float, m2: float, l1: float, l2: float, g: float
) -> list[float]:
    """
    Computes derivatives for the double pendulum using Lagrangian mechanics.
    state: [theta1, omega1, theta2, omega2]
    """
    theta1, omega1, theta2, omega2 = state

    delta = theta1 - theta2

    # Lagrangian derivatives
    den1 = l1 * (2 * m1 + m2 - m2 * np.cos(2 * theta1 - 2 * theta2))
    domega1 = (
        -g * (2 * m1 + m2) * np.sin(theta1)
        - m2 * g * np.sin(theta1 - 2 * theta2)
        - 2 * np.sin(delta) * m2 * (omega2**2 * l2 + omega1**2 * l1 * np.cos(delta))
    ) / den1

    den2 = l2 * (2 * m1 + m2 - m2 * np.cos(2 * theta1 - 2 * theta2))
    domega2 = (
        2
        * np.sin(delta)
        * (
            omega1**2 * l1 * (m1 + m2)
            + g * (m1 + m2) * np.cos(theta1)
            + omega2**2 * l2 * m2 * np.cos(delta)
        )
    ) / den2

    return [omega1, domega1, omega2, domega2]


def simulate(
    t_max: float,
    dt: float,
    initial_state: list[float],
    m1: float,
    m2: float,
    l1: float,
    l2: float,
    g: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Integrates Lagrangian equations over time and calculates physical forces."""
    t_eval = np.arange(0, t_max, dt)
    res = solve_ivp(
        fun=lambda t, y: equations_of_motion(t, y, m1, m2, l1, l2, g),
        t_span=[0, t_max],
        y0=initial_state,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-8,
    )

    if not res.success:
        logging.error(f"Integration failed: {res.message}")
        raise RuntimeError("ODE integration failed.")

    theta1 = res.y[0, :]
    omega1 = res.y[1, :]
    theta2 = res.y[2, :]
    omega2 = res.y[3, :]

    # Recalculate angular accelerations to derive forces
    alpha1 = np.zeros_like(theta1)
    alpha2 = np.zeros_like(theta2)

    for i in range(len(theta1)):
        derivs = equations_of_motion(
            0, [theta1[i], omega1[i], theta2[i], omega2[i]], m1, m2, l1, l2, g
        )
        alpha1[i] = derivs[1]
        alpha2[i] = derivs[3]

    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)

    # Calculate cartesian accelerations
    a1_x = l1 * (alpha1 * np.cos(theta1) - omega1**2 * np.sin(theta1))
    a1_y = l1 * (alpha1 * np.sin(theta1) + omega1**2 * np.cos(theta1))

    a2_x = a1_x + l2 * (alpha2 * np.cos(theta2) - omega2**2 * np.sin(theta2))
    a2_y = a1_y + l2 * (alpha2 * np.sin(theta2) + omega2**2 * np.cos(theta2))

    # Unconstrained net force (F = ma) allowing vectors in 2D independent of rod tension
    F1_x = m1 * a1_x
    F1_y = m1 * a1_y

    F2_x = m2 * a2_x
    F2_y = m2 * a2_y

    return x1, y1, x2, y2, F1_x, F1_y, F2_x, F2_y, alpha1, alpha2, t_eval


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chaotic Lagrangian Double Pendulum Screensaver"
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Path for video output (e.g., saver.mp4 or saver.gif).",
        default=None,
    )
    parser.add_argument(
        "--fps", type=int, default=60, help="Frames per second. Default 60."
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Simulation time in seconds. Default 30.",
    )
    parser.add_argument("--m1", type=float, default=1.0, help="Mass 1.")
    parser.add_argument("--m2", type=float, default=2.5, help="Mass 2 (End effector).")
    parser.add_argument("--l1", type=float, default=1.0, help="Length 1.")
    parser.add_argument("--l2", type=float, default=1.0, help="Length 2.")
    parser.add_argument("--gravity", type=float, default=9.81, help="Gravity.")
    args = parser.parse_args()

    dt = 1.0 / args.fps

    # Start with high energy (elevated angles and initial velocities)
    initial_state = [np.pi / 1.1, 2.5, np.pi / 1.5, 4.0]

    logging.info("Calculating Lagrangian mechanics & Force Tensors...")
    try:
        x1, y1, x2, y2, F1_x, F1_y, F2_x, F2_y, alpha1, alpha2, t_eval = simulate(
            args.duration,
            dt,
            initial_state,
            args.m1,
            args.m2,
            args.l1,
            args.l2,
            args.gravity,
        )
    except Exception as e:
        logging.error(f"Failed to simulate: {e}")
        sys.exit(1)

    # Physical derivations for active plotting
    F1_mag = np.hypot(F1_x, F1_y)
    F2_mag = np.hypot(F2_x, F2_y)
    tau1 = args.m1 * args.l1**2 * alpha1
    tau2 = args.m2 * args.l2**2 * alpha2

    max_F = max(np.max(F1_mag), np.max(F2_mag)) * 1.05 + 1e-3
    max_T = max(np.max(np.abs(tau1)), np.max(np.abs(tau2))) * 1.05 + 1e-3

    logging.info("Preparing visual environment...")

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(14, 8), dpi=120)
    fig.patch.set_facecolor("#0B0C10")

    gs = gridspec.GridSpec(2, 2, width_ratios=[1.2, 1])

    # ==========================================
    # Viewport 1: Pendulum Simulation
    # ==========================================
    ax = fig.add_subplot(gs[:, 0])
    ax.set_facecolor("#0B0C10")
    ax.axis("off")

    max_len = args.l1 + args.l2
    ax.set_xlim(-max_len * 1.5, max_len * 1.5)
    ax.set_ylim(-max_len * 1.5, max_len * 1.5)
    ax.set_aspect("equal")

    # Visual elements
    (trail,) = ax.plot([], [], "-", lw=1.5, color="#45A29E", alpha=0.4, zorder=1)

    arm1 = Ellipse(
        (0, 0),
        width=args.l1,
        height=0.10,
        angle=0,
        color="#66FCF1",
        alpha=0.8,
        zorder=2,
    )
    arm2 = Ellipse(
        (0, 0),
        width=args.l2,
        height=0.10,
        angle=0,
        color="#45A29E",
        alpha=0.8,
        zorder=2,
    )
    ax.add_patch(arm1)
    ax.add_patch(arm2)

    joint0 = Ellipse((0, 0), width=0.08, height=0.08, color="#FFFFFF", zorder=4)
    joint1 = Ellipse((0, 0), width=0.12, height=0.12, color="#FFFFFF", zorder=4)
    end_effector = Ellipse(
        (0, 0), width=0.18 * args.m2, height=0.18 * args.m2, color="#4A90E2", zorder=4
    )
    ax.add_patch(joint0)
    ax.add_patch(joint1)
    ax.add_patch(end_effector)

    force_scale = 0.015
    (f1_line,) = ax.plot([], [], "-", color="#FF0055", lw=2.5, zorder=5)  # F1
    (f2_line,) = ax.plot([], [], "-", color="#FFAA00", lw=2.5, zorder=5)  # F2

    # Adds the legend specifically for this plotting box
    ax.plot([], [], "-", color="#FF0055", lw=2.5, label=r"$\mathbf{F}_{net, 1}$")
    ax.plot([], [], "-", color="#FFAA00", lw=2.5, label=r"$\mathbf{F}_{net, 2}$")
    ax.legend(
        loc="upper right",
        facecolor="#0B0C10",
        edgecolor="none",
        fontsize=12,
        labelcolor="white",
    )

    # ==========================================
    # Viewport 2: Rolling Force Tensors Plot
    # ==========================================
    history_sec = 10.0

    ax_f = fig.add_subplot(gs[0, 1])
    ax_f.set_facecolor("#1F2833")
    ax_f.set_xlim(-history_sec, 0)
    ax_f.set_ylim(0, max_F)
    ax_f.set_title(
        r"Net Force Magnitude ($\|\mathbf{F}\|$)",
        color="#66FCF1",
        fontsize=14,
        weight="bold",
    )
    ax_f.tick_params(colors="#C5C6C7", labelsize=10)
    ax_f.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
    ax_f.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax_f.yaxis.get_offset_text().set_color("#C5C6C7")
    for spine in ax_f.spines.values():
        spine.set_color("#45A29E")
    ax_f.grid(True, color="#0B0C10", alpha=0.6, linestyle="--")

    (f1_curve,) = ax_f.plot([], [], color="#FF0055", lw=2.0, label=r"$F_1$")
    (f2_curve,) = ax_f.plot([], [], color="#FFAA00", lw=2.0, label=r"$F_2$")
    ax_f.legend(
        loc="upper left",
        facecolor="#0B0C10",
        edgecolor="none",
        labelcolor="white",
        fontsize=11,
    )

    # ==========================================
    # Viewport 3: Rolling Torques Plot
    # ==========================================
    ax_t = fig.add_subplot(gs[1, 1])
    ax_t.set_facecolor("#1F2833")
    ax_t.set_xlim(-history_sec, 0)
    ax_t.set_ylim(-max_T, max_T)
    ax_t.set_title(
        r"Nodal Torques ($\tau$)", color="#4A90E2", fontsize=14, weight="bold"
    )
    ax_t.tick_params(colors="#C5C6C7", labelsize=10)
    ax_t.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
    ax_t.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax_t.yaxis.get_offset_text().set_color("#C5C6C7")
    for spine in ax_t.spines.values():
        spine.set_color("#45A29E")
    ax_t.grid(True, color="#0B0C10", alpha=0.6, linestyle="--")

    (t1_curve,) = ax_t.plot([], [], color="#66FCF1", lw=2.0, label=r"$\tau_1$")
    (t2_curve,) = ax_t.plot([], [], color="#4A90E2", lw=2.0, label=r"$\tau_2$")
    ax_t.legend(
        loc="upper left",
        facecolor="#0B0C10",
        edgecolor="none",
        labelcolor="white",
        fontsize=11,
    )

    trail_length = int(args.fps * history_sec)

    def init() -> tuple[Any, ...]:
        trail.set_data([], [])
        f1_line.set_data([], [])
        f2_line.set_data([], [])
        arm1.center = (0, 0)
        arm2.center = (0, 0)
        joint1.center = (0, 0)
        end_effector.center = (0, 0)
        f1_curve.set_data([], [])
        f2_curve.set_data([], [])
        t1_curve.set_data([], [])
        t2_curve.set_data([], [])
        return (
            trail,
            arm1,
            arm2,
            joint0,
            joint1,
            end_effector,
            f1_line,
            f2_line,
            f1_curve,
            f2_curve,
            t1_curve,
            t2_curve,
        )

    def animate(i: int) -> tuple[Any, ...]:
        start_idx = max(0, i - trail_length)

        # Pendulum updates
        trail.set_data(x2[start_idx : i + 1], y2[start_idx : i + 1])

        arm1.center = (x1[i] / 2, y1[i] / 2)
        arm2.center = (x1[i] + (x2[i] - x1[i]) / 2, y1[i] + (y2[i] - y1[i]) / 2)
        arm1.angle = np.degrees(np.arctan2(y1[i], x1[i]))
        arm2.angle = np.degrees(np.arctan2(y2[i] - y1[i], x2[i] - x1[i]))

        joint1.center = (x1[i], y1[i])
        end_effector.center = (x2[i], y2[i])

        fx1 = x1[i] + F1_x[i] * force_scale
        fy1 = y1[i] + F1_y[i] * force_scale
        f1_line.set_data([x1[i], fx1], [y1[i], fy1])

        fx2 = x2[i] + F2_x[i] * force_scale
        fy2 = y2[i] + F2_y[i] * force_scale
        f2_line.set_data([x2[i], fx2], [y2[i], fy2])

        # Rolling plot sliding updates
        window_t = t_eval[start_idx : i + 1]
        active_t = window_t - t_eval[i]

        f1_curve.set_data(active_t, F1_mag[start_idx : i + 1])
        f2_curve.set_data(active_t, F2_mag[start_idx : i + 1])

        t1_curve.set_data(active_t, tau1[start_idx : i + 1])
        t2_curve.set_data(active_t, tau2[start_idx : i + 1])

        return (
            trail,
            arm1,
            arm2,
            joint0,
            joint1,
            end_effector,
            f1_line,
            f2_line,
            f1_curve,
            f2_curve,
            t1_curve,
            t2_curve,
        )

    logging.info("Building animation sequence...")
    anim = FuncAnimation(
        fig,
        animate,
        frames=len(x1),
        init_func=init,
        interval=dt * 1000,
        blit=True,
        repeat=not bool(args.save),
    )

    if args.save:
        logging.info(f"Saving video to {args.save} ...")
        if args.save.endswith(".gif"):
            writer = PillowWriter(fps=args.fps)
        else:
            writer = FFMpegWriter(
                fps=args.fps, metadata=dict(artist="AI Agent"), bitrate=3000
            )

        try:
            anim.save(args.save, writer=writer)
            logging.info(f"Successfully saved {args.save}")
        except Exception as e:
            logging.error(f"Failed to output video. Error: {e}")
    else:
        logging.info("Spawning live display panel (Close window to exit)...")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
