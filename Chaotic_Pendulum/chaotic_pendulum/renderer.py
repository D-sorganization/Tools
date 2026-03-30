import matplotlib

matplotlib.use("TkAgg")
import logging
import time
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse
from matplotlib.widgets import CheckButtons

from .config import PhysicsConfig, RenderConfig


class PendulumRenderer:
    """DbC focused renderer isolating Matplotlib spaghetti logic from physics engine."""

    def __init__(
        self, render_cfg: RenderConfig, phys_cfg: PhysicsConfig, data: dict[str, Any]
    ):
        self.r_cfg = render_cfg
        self.p_cfg = phys_cfg
        self.data = data
        self.t_eval = data["t_eval"]
        self.dt = 1.0 / self.r_cfg.fps

        # Unpack
        self.x1 = data["pos"]["x1"]
        self.y1 = data["pos"]["y1"]
        self.x2 = data["pos"]["x2"]
        self.y2 = data["pos"]["y2"]

        self.f1_tot = data["v1"]["total"]
        self.f1_cf = data["v1"]["centrifugal"]
        self.f1_cor = data["v1"]["coriolis"]

        self.f2_tot = data["v2"]["total"]
        self.f2_cf = data["v2"]["centrifugal"]
        self.f2_cor = data["v2"]["coriolis"]

        # Precompute Mag
        self.mag_tot1 = np.hypot(self.f1_tot[0], self.f1_tot[1])
        self.mag_tot2 = np.hypot(self.f2_tot[0], self.f2_tot[1])
        self.mag_cf1 = np.hypot(self.f1_cf[0], self.f1_cf[1])
        self.mag_cf2 = np.hypot(self.f2_cf[0], self.f2_cf[1])
        self.mag_cor1 = np.hypot(self.f1_cor[0], self.f1_cor[1])
        self.mag_cor2 = np.hypot(self.f2_cor[0], self.f2_cor[1])

        self.max_F = (
            max(
                np.max(self.mag_tot1),
                np.max(self.mag_tot2),
                np.max(self.mag_cf1),
                np.max(self.mag_cf2),
                np.max(self.mag_cor2),
            )
            * 1.05
            + 1e-3
        )

    def render(self) -> None:
        """Launches drawing environment."""
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(15, 8), dpi=120)
        self.fig.patch.set_facecolor("#0B0C10")

        gs = gridspec.GridSpec(2, 2, width_ratios=[1.2, 1])

        self.setup_pendulum_axes(gs)
        self.setup_force_axes(gs)
        self.setup_torque_axes(gs)
        self.setup_widgets()

        trail_length = int(self.r_cfg.fps * self.r_cfg.history_sec)

        # State tracking via class namespace (avoids mypy list generic hack)
        self.start_time_ref: float = -1.0

        def live_frames() -> Any:
            while True:
                yield 0

        # Main closure overrides
        def init() -> tuple[Any, ...]:
            self.start_time_ref = -1.0
            self.trail.set_data([], [])
            self.time_text.set_text("")
            return self.trail, self.time_text

        def animate(frame_i: int) -> tuple[Any, ...]:
            if self.r_cfg.save_path:
                i = frame_i
            else:
                if self.start_time_ref < 0:
                    self.start_time_ref = time.time()
                elapsed = time.time() - self.start_time_ref
                i = int(elapsed / self.dt)
                if i >= len(self.x1):
                    self.start_time_ref = time.time()
                    i = 0
                i = min(i, len(self.x1) - 1)

            start_idx = max(0, i - trail_length)
            self._update_elements(i, start_idx)
            return self.trail, self.time_text

        anim = FuncAnimation(
            self.fig,
            animate,
            frames=len(self.x1) if self.r_cfg.save_path else live_frames(),
            init_func=init,
            interval=self.dt * 1000,
            blit=False,
            repeat=not bool(self.r_cfg.save_path),
        )

        if self.r_cfg.save_path:
            logging.info(f"Saving video to {self.r_cfg.save_path} ...")
            writer: Any
            if self.r_cfg.save_path.endswith(".gif"):
                writer = PillowWriter(fps=self.r_cfg.fps)
            else:
                writer = FFMpegWriter(
                    fps=self.r_cfg.fps, metadata=dict(artist="AI Agent"), bitrate=3000
                )

            try:
                anim.save(self.r_cfg.save_path, writer=writer)
                logging.info(f"Successfully saved {self.r_cfg.save_path}")
            except Exception as e:
                logging.error(f"Failed to output video. Error: {e}")
        else:
            logging.info("Spawning live display panel (Close window to exit)...")
            plt.tight_layout()
            plt.show()

    def _update_elements(self, i: int, start: int) -> None:
        """Internal updater to keep logic DRY."""
        self.trail.set_data(self.x2[start : i + 1], self.y2[start : i + 1])

        x1, y1 = self.x1[i], self.y1[i]
        x2, y2 = self.x2[i], self.y2[i]

        # Ellipse placements
        self.arm1.center = (x1 / 2, y1 / 2)
        self.arm2.center = (x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2)
        self.arm1.angle = np.degrees(np.arctan2(y1, x1))
        self.arm2.angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        self.joint1.center = (x1, y1)
        self.end_effector.center = (x2, y2)

        # Force Vectors Drawing Logic
        status = self.check.get_status()
        show_tot, show_cf, show_cor = status

        fscale = 0.012
        # Totals
        if show_tot:
            self.line_f1_tot.set_data(
                [x1, x1 + self.f1_tot[0][i] * fscale],
                [y1, y1 + self.f1_tot[1][i] * fscale],
            )
            self.line_f2_tot.set_data(
                [x2, x2 + self.f2_tot[0][i] * fscale],
                [y2, y2 + self.f2_tot[1][i] * fscale],
            )
        else:
            self.line_f1_tot.set_data([], [])
            self.line_f2_tot.set_data([], [])

        # Centrifugal
        if show_cf:
            self.line_f1_cf.set_data(
                [x1, x1 + self.f1_cf[0][i] * fscale],
                [y1, y1 + self.f1_cf[1][i] * fscale],
            )
            self.line_f2_cf.set_data(
                [x2, x2 + self.f2_cf[0][i] * fscale],
                [y2, y2 + self.f2_cf[1][i] * fscale],
            )
        else:
            self.line_f1_cf.set_data([], [])
            self.line_f2_cf.set_data([], [])

        # Coriolis
        if show_cor:
            self.line_f1_cor.set_data(
                [x1, x1 + self.f1_cor[0][i] * fscale],
                [y1, y1 + self.f1_cor[1][i] * fscale],
            )
            self.line_f2_cor.set_data(
                [x2, x2 + self.f2_cor[0][i] * fscale],
                [y2, y2 + self.f2_cor[1][i] * fscale],
            )
        else:
            self.line_f1_cor.set_data([], [])
            self.line_f2_cor.set_data([], [])

        # UI Time
        self.time_text.set_text(f"T: {self.t_eval[i]:.2f}s")

        # Rolling Forces Graph Update
        window_t = self.t_eval[start : i + 1]
        active_t = window_t - self.t_eval[i]

        self.plot_f1_tot.set_data(
            (active_t, self.mag_tot1[start : i + 1]) if show_tot else ([], [])
        )
        self.plot_f2_tot.set_data(
            (active_t, self.mag_tot2[start : i + 1]) if show_tot else ([], [])
        )
        self.plot_f1_cf.set_data(
            (active_t, self.mag_cf1[start : i + 1]) if show_cf else ([], [])
        )
        self.plot_f2_cf.set_data(
            (active_t, self.mag_cf2[start : i + 1]) if show_cf else ([], [])
        )
        self.plot_f1_cor.set_data(
            (active_t, self.mag_cor1[start : i + 1]) if show_cor else ([], [])
        )
        self.plot_f2_cor.set_data(
            (active_t, self.mag_cor2[start : i + 1]) if show_cor else ([], [])
        )

        # Torque Graph Update
        self.plot_t1.set_data(active_t, self.data["tau1"][start : i + 1])
        self.plot_t2.set_data(active_t, self.data["tau2"][start : i + 1])

    def setup_pendulum_axes(self, gs: Any) -> None:
        ax = self.fig.add_subplot(gs[:, 0])
        ax.set_facecolor("#0B0C10")
        ax.axis("off")

        max_len = self.p_cfg.l1 + self.p_cfg.l2
        ax.set_xlim(-max_len * 1.5, max_len * 1.5)
        ax.set_ylim(-max_len * 1.5, max_len * 1.5)
        ax.set_aspect("equal")

        (self.trail,) = ax.plot(
            [], [], "-", lw=1.5, color="#45A29E", alpha=0.4, zorder=1
        )
        self.arm1 = Ellipse(
            (0, 0),
            width=self.p_cfg.l1,
            height=0.10,
            angle=0,
            color="#66FCF1",
            alpha=0.8,
            zorder=2,
        )
        self.arm2 = Ellipse(
            (0, 0),
            width=self.p_cfg.l2,
            height=0.10,
            angle=0,
            color="#45A29E",
            alpha=0.8,
            zorder=2,
        )
        ax.add_patch(self.arm1)
        ax.add_patch(self.arm2)

        self.joint0 = Ellipse(
            (0, 0), width=0.08, height=0.08, color="#FFFFFF", zorder=4
        )
        self.joint1 = Ellipse(
            (0, 0), width=0.12, height=0.12, color="#FFFFFF", zorder=4
        )
        self.end_effector = Ellipse(
            (0, 0),
            width=0.18 * self.p_cfg.m2,
            height=0.18 * self.p_cfg.m2,
            color="#4A90E2",
            zorder=4,
        )
        ax.add_patch(self.joint0)
        ax.add_patch(self.joint1)
        ax.add_patch(self.end_effector)

        # Vectors rendering logic
        (self.line_f1_tot,) = ax.plot([], [], "-", color="#FF0055", lw=2.5, zorder=5)
        (self.line_f2_tot,) = ax.plot([], [], "-", color="#FFAA00", lw=2.5, zorder=5)
        (self.line_f1_cf,) = ax.plot([], [], "--", color="#11FF00", lw=2.0, zorder=5)
        (self.line_f2_cf,) = ax.plot([], [], "--", color="#11AA00", lw=2.0, zorder=5)
        (self.line_f1_cor,) = ax.plot([], [], ":", color="#CC00FF", lw=2.0, zorder=5)
        (self.line_f2_cor,) = ax.plot([], [], ":", color="#AA00FF", lw=2.0, zorder=5)

        self.time_text = ax.text(
            0.04,
            0.94,
            "",
            transform=ax.transAxes,
            color="#45A29E",
            fontsize=12,
            weight="bold",
            fontfamily="monospace",
        )

    def setup_force_axes(self, gs: Any) -> None:
        ax = self.fig.add_subplot(gs[0, 1])
        ax.set_facecolor("#1F2833")
        ax.set_xlim(-self.r_cfg.history_sec, 0)
        ax.set_ylim(0, self.max_F)
        ax.set_title(
            r"Force Magnitude Logs", color="#66FCF1", fontsize=14, weight="bold"
        )
        ax.tick_params(colors="#C5C6C7", labelsize=10)
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.ScalarFormatter(useMathText=True)
        )
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_color("#C5C6C7")
        for spine in ax.spines.values():
            spine.set_color("#45A29E")
        ax.grid(True, color="#0B0C10", alpha=0.6, linestyle="--")

        (self.plot_f1_tot,) = ax.plot([], [], color="#FF0055", lw=2.0, label=r"Tot$_1$")
        (self.plot_f2_tot,) = ax.plot([], [], color="#FFAA00", lw=2.0, label=r"Tot$_2$")
        (self.plot_f1_cf,) = ax.plot(
            [], [], "--", color="#11FF00", lw=1.5, alpha=0.8, label=r"CF$_1$"
        )
        (self.plot_f2_cf,) = ax.plot(
            [], [], "--", color="#11AA00", lw=1.5, alpha=0.8, label=r"CF$_2$"
        )
        (self.plot_f1_cor,) = ax.plot(
            [], [], ":", color="#CC00FF", lw=1.5, alpha=0.8, label=r"Cor$_1$"
        )
        (self.plot_f2_cor,) = ax.plot(
            [], [], ":", color="#AA00FF", lw=1.5, alpha=0.8, label=r"Cor$_2$"
        )

        ax.legend(
            loc="upper left",
            facecolor="#0B0C10",
            edgecolor="none",
            labelcolor="white",
            fontsize=10,
            ncol=3,
        )

    def setup_torque_axes(self, gs: Any) -> None:
        ax = self.fig.add_subplot(gs[1, 1])
        ax.set_facecolor("#1F2833")
        ax.set_xlim(-self.r_cfg.history_sec, 0)
        max_T = (
            max(np.max(np.abs(self.data["tau1"])), np.max(np.abs(self.data["tau2"])))
            * 1.05
            + 1e-3
        )
        ax.set_ylim(-max_T, max_T)
        ax.set_title(
            r"Nodal Torques ($\tau$)", color="#4A90E2", fontsize=14, weight="bold"
        )
        ax.tick_params(colors="#C5C6C7", labelsize=10)
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.ScalarFormatter(useMathText=True)
        )
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_color("#C5C6C7")
        for spine in ax.spines.values():
            spine.set_color("#45A29E")
        ax.grid(True, color="#0B0C10", alpha=0.6, linestyle="--")

        (self.plot_t1,) = ax.plot([], [], color="#66FCF1", lw=2.0, label=r"$\tau_1$")
        (self.plot_t2,) = ax.plot([], [], color="#4A90E2", lw=2.0, label=r"$\tau_2$")
        ax.legend(
            loc="upper left",
            facecolor="#0B0C10",
            edgecolor="none",
            labelcolor="white",
            fontsize=11,
        )

    def setup_widgets(self) -> None:
        """Sets up UI check buttons."""
        ax_check = self.fig.add_axes((0.02, 0.85, 0.15, 0.12))
        # Making UI Dark to match scheme
        ax_check.set_facecolor("#1F2833")
        ax_check.patch.set_alpha(0.8)  # Transparent background

        labels = ["Total Forces", "Centrifugal Out", "Coriolis Ortho"]
        visibility = [True, True, True]
        self.check = CheckButtons(ax_check, labels, visibility)

        # Style checkboxes
        for t in self.check.labels:
            t.set_color("white")
            t.set_fontsize(10)
            t.set_fontfamily("monospace")
