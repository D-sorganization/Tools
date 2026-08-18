"""Shared animation and plot-view scaffolding for motion-analysis tabs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, cast

from PyQt6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .motion_analysis_panel import MotionAnalysisPanel
from .vector_overlay import OverlayScene


class _MotionCanvas(Protocol):
    """Structural boundary used by the shared view scaffolding."""

    LAYERS: tuple[tuple[str, str], ...]

    def is_layer_visible(self, name: str) -> bool: ...

    def set_layer_visible(self, name: str, visible: bool) -> None: ...

    def set_scene(
        self,
        chain_nodes: list[tuple[float, float]],
        body_points: dict[str, tuple[float, float]] | None = None,
    ) -> None: ...

    def set_overlays(self, scene: OverlayScene) -> None: ...


class MotionViewMixin:
    """Build shared Animation/Plots subtabs and their appearance controls."""

    canvas: _MotionCanvas
    analysis_panel: MotionAnalysisPanel
    _layer_toggles: dict[str, QCheckBox]
    _plot_legend_toggle: QCheckBox

    def _build_animation_view(self) -> QWidget:
        """Return the animation subtab with full vertical canvas room."""
        view = QWidget()
        view_layout = QVBoxLayout(view)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.addWidget(cast(QWidget, self.canvas))
        return view

    def _build_plots_view(self) -> QWidget:
        """Return roomy analysis plots with a legend visibility control."""
        view = QWidget()
        view_layout = QVBoxLayout(view)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.setSpacing(6)
        appearance = QHBoxLayout()
        self._plot_legend_toggle = QCheckBox("Show plot legends")
        self._plot_legend_toggle.setChecked(True)
        self._plot_legend_toggle.setToolTip(
            "Show or hide the legends on the analysis plots so they do not "
            "obscure the plotted curves."
        )
        self._plot_legend_toggle.stateChanged.connect(self._refresh_plot_legends)
        appearance.addWidget(self._plot_legend_toggle)
        appearance.addStretch()
        view_layout.addLayout(appearance)
        plot_scroll = QScrollArea()
        plot_scroll.setWidgetResizable(True)
        plot_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        plot_scroll.setWidget(self.analysis_panel)
        view_layout.addWidget(plot_scroll, stretch=1)
        return view

    def _build_layers_group(self, layer_keys: Sequence[str] | None = None) -> QGroupBox:
        """Build the animation-layer checklist for the supplied layer keys."""
        allowed = set(layer_keys) if layer_keys is not None else None
        group = QGroupBox("Show in animation")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        tips = {
            "grid": "Background reference grid.",
            "chain": "Swing chain polyline.",
            "rider": "Articulated rider body segments.",
            "markers": "Anchor and seat pivot markers.",
            "forces": "All force and torque vector overlays.",
        }
        for key, label in self.canvas.LAYERS:
            if allowed is not None and key not in allowed:
                continue
            checkbox = QCheckBox(label)
            checkbox.setChecked(self.canvas.is_layer_visible(key))
            checkbox.setToolTip(tips.get(key, ""))
            checkbox.stateChanged.connect(
                lambda _state, name=key, box=checkbox: self.canvas.set_layer_visible(
                    name, box.isChecked()
                )
            )
            self._layer_toggles[key] = checkbox
            layout.addWidget(checkbox)
        return group

    def _apply_plot_legend_visibility(self) -> None:
        """Match analysis-plot legend visibility to the appearance toggle."""
        self.analysis_panel.set_legends_visible(self._plot_legend_toggle.isChecked())

    def _refresh_plot_legends(self, _state: int | None = None) -> None:
        """Apply the legend toggle and redraw the analysis panel."""
        self._apply_plot_legend_visibility()
        self.analysis_panel.draw()
