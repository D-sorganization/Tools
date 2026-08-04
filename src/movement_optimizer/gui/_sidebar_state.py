# Copyright (c) 2024-2026 D-sorganization
# SPDX-License-Identifier: MIT
#
# Movement-Optimizer — see LICENSE for full terms.
#

"""State management helpers for ParameterSidebar."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

from ..constants import (
    PROGRESS_EVAL_SCALE,
    PROGRESS_MAX_PCT,
    PROGRESS_PHASE_BOUNDARY_EVALS,
    STALL_HINT_ELAPSED_S,
)
from ..models import BodyModel
from ..trajectory import ProgressReport

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from PyQt6.QtWidgets import QLabel, QProgressBar, QPushButton

    from .labelled_slider import LabelledSlider

logger = logging.getLogger(__name__)


class SidebarStateContract(Protocol):
    """Widgets and sliders used by sidebar state helpers."""

    opt_btn: QPushButton
    both_btn: QPushButton
    cancel_btn: QPushButton
    stall_label: QLabel
    progress: QProgressBar
    prog_label: QLabel
    iter_label: QLabel
    cost_label: QLabel
    improve_label: QLabel
    elapsed_label: QLabel
    conv_ax: Axes
    conv_canvas: FigureCanvas
    conv_fig: Figure
    mass_slider: LabelledSlider
    height_slider: LabelledSlider
    ll_slider: LabelledSlider
    ul_slider: LabelledSlider
    to_slider: LabelledSlider
    bar_slider: LabelledSlider
    bar_depth_slider: LabelledSlider
    bar_height_slider: LabelledSlider
    dur_slider: LabelledSlider
    smooth_slider: LabelledSlider


def show_optimizing(sidebar: SidebarStateContract) -> None:
    sidebar.opt_btn.setEnabled(False)
    sidebar.opt_btn.setToolTip("Optimization currently in progress. Please wait or cancel.")
    sidebar.both_btn.setEnabled(False)
    sidebar.both_btn.setToolTip("Optimization currently in progress. Please wait or cancel.")
    sidebar.cancel_btn.setVisible(True)
    sidebar.cancel_btn.setToolTip("Cancel the currently running optimization (Esc)")
    sidebar.stall_label.setVisible(False)
    sidebar.stall_label.setText("")
    sidebar.progress.setValue(0)
    _clear_progress_labels(sidebar)


def show_idle(sidebar: SidebarStateContract) -> None:
    sidebar.opt_btn.setEnabled(True)
    sidebar.opt_btn.setToolTip(
        "Start trajectory optimization for the currently selected exercise tab (Ctrl+R)"
    )
    sidebar.both_btn.setEnabled(True)
    sidebar.both_btn.setToolTip(
        "Start trajectory optimization sequentially for all exercise tabs (Ctrl+Shift+R)"
    )
    # Restore cancel button to its default text/state before hiding it so that
    # the next optimization run starts with a clean button label.
    sidebar.cancel_btn.setText("Cancel")
    sidebar.cancel_btn.setEnabled(True)
    sidebar.cancel_btn.setVisible(False)


def update_progress(sidebar: SidebarStateContract, report: ProgressReport) -> None:
    n_evals = report.iteration
    pct = min(
        PROGRESS_MAX_PCT,
        int(PROGRESS_MAX_PCT * (1 - 1 / (1 + n_evals / PROGRESS_EVAL_SCALE))),
    )
    sidebar.progress.setValue(pct)
    phase = "Converging" if n_evals > PROGRESS_PHASE_BOUNDARY_EVALS else "Exploring"
    sidebar.prog_label.setText(f"{phase}...")
    sidebar.iter_label.setText(f"Evaluations: {report.iteration}")
    sidebar.cost_label.setText(f"Cost: {report.cost:.1f}  (best: {report.best_cost:.1f})")
    sidebar.improve_label.setText(f"Improvement: {report.improvement_pct:+.3f}%")
    elapsed = report.elapsed_s
    time_str = f"{elapsed:.1f}s" if elapsed < 60 else f"{int(elapsed // 60)}m {elapsed % 60:.0f}s"
    sidebar.elapsed_label.setText(f"Elapsed: {time_str}")

    if report.is_stalled:
        sidebar.stall_label.setText(f"\u26a0 STALLED: {report.stall_reason}")
        sidebar.stall_label.setVisible(True)
    elif elapsed > STALL_HINT_ELAPSED_S:
        sidebar.stall_label.setText(
            "\u26a0 Taking longer than expected. Consider cancelling and adjusting parameters."
        )
        sidebar.stall_label.setVisible(True)
    else:
        sidebar.stall_label.setVisible(False)

    _update_conv_plot(sidebar, report.cost_history)


def _update_conv_plot(sidebar: SidebarStateContract, history: list[float]) -> None:
    ax = sidebar.conv_ax
    ax.clear()
    _style_conv_ax(sidebar)
    if len(history) > 1:
        clean = [c for c in history if c < 1e30]
        if len(clean) > 1:
            ax.plot(range(len(clean)), clean, color="C0", lw=1.2)
            ax.set_yscale("log")
    sidebar.conv_canvas.draw_idle()


def _style_conv_ax(sidebar: SidebarStateContract) -> None:
    from ..rendering import Palette

    ax = sidebar.conv_ax
    ax.set_facecolor(Palette.BG_PLOT)
    ax.tick_params(colors=Palette.FG_DIM, which="both", labelsize=6)
    for sp in ("bottom", "left"):
        ax.spines[sp].set_color(Palette.FG_DIM)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("eval", fontsize=6, color=Palette.FG_DIM)
    ax.set_ylabel("cost", fontsize=6, color=Palette.FG_DIM)
    sidebar.conv_fig.tight_layout(pad=0.3)


def _clear_progress_labels(sidebar: SidebarStateContract) -> None:
    sidebar.prog_label.setText("")
    sidebar.iter_label.setText("")
    sidebar.cost_label.setText("")
    sidebar.improve_label.setText("")
    sidebar.elapsed_label.setText("")
    sidebar.conv_ax.clear()
    _style_conv_ax(sidebar)
    sidebar.conv_canvas.draw_idle()


def get_body_model(sidebar: SidebarStateContract) -> BodyModel:
    return BodyModel(
        body_mass=sidebar.mass_slider.value(),
        height=sidebar.height_slider.value(),
        seg_multipliers={
            "lower_leg": sidebar.ll_slider.value(),
            "upper_leg": sidebar.ul_slider.value(),
            "torso": sidebar.to_slider.value(),
        },
        squat_bar_depth=sidebar.bar_depth_slider.value(),
        squat_bar_height=sidebar.bar_height_slider.value(),
    )


def reset_defaults(sidebar: SidebarStateContract) -> None:
    sidebar.mass_slider.set_value(75.0)
    sidebar.height_slider.set_value(1.75)
    sidebar.ll_slider.set_value(1.00)
    sidebar.ul_slider.set_value(1.00)
    sidebar.to_slider.set_value(1.00)
    sidebar.bar_slider.set_value(60.0)
    sidebar.bar_depth_slider.set_value(0.0)
    sidebar.bar_height_slider.set_value(0.0)
    sidebar.dur_slider.set_value(2.0)
    sidebar.smooth_slider.set_value(1.0)
