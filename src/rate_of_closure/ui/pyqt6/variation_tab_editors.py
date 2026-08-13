"""Noise-row lifecycle and locus-context policy for the PyQt Variation tab."""

from __future__ import annotations

from typing import cast

from PyQt6.QtWidgets import QLabel, QVBoxLayout

from rate_of_closure.simulation import SimulationConfig
from rate_of_closure.ui.pyqt6.variation_rows import NoiseRow
from shared.python.swing_sim.integration_grid import effective_rk4_duration
from shared.python.swing_sim.variation import PerturbationGroup

__all__ = ["VariationTabEditorsMixin"]


class VariationTabEditorsMixin:
    """Own row creation/removal and one source-aware locus context decision."""

    _base_simulation_config: SimulationConfig
    _loaded_base: dict[str, float]
    _loaded_groups: tuple[PerturbationGroup, ...]
    _rows: list[NoiseRow]
    _rows_layout: QVBoxLayout
    _status: QLabel

    def mode(self) -> str:
        """Return the concrete tab's current pipeline mode."""
        raise NotImplementedError

    def _add_row(self) -> NoiseRow:
        row = NoiseRow(
            self.mode(),
            self._remove_row,
            localized_enabled=self._localized_authoring_enabled(),
            duration_s=self._localized_duration_s(),
            on_change=self._on_noise_editor_changed,
        )
        self._rows.append(row)
        self._rows_layout.insertWidget(self._rows_layout.count() - 1, row)
        return row

    def _remove_row(self, row: NoiseRow) -> None:
        if len(self._rows) <= 1:
            self._status.setText("At least one noise row is required.")
            return
        self._rows.remove(row)
        row.setParent(None)
        row.deleteLater()
        self._on_noise_editor_changed()

    def _on_mode_changed(self, *_args: object) -> None:
        self._loaded_base.clear()
        self._loaded_groups = ()
        self._refresh_row_contexts()
        self._on_noise_editor_changed()

    def _localized_authoring_enabled(self, mode: str | None = None) -> bool:
        """Return whether a mode/source pair can execute authored torque loci."""
        selected_mode = self.mode() if mode is None else mode
        return (
            selected_mode == "swing"
            and self._base_simulation_config.source_kind == "double_pendulum"
        )

    def _localized_duration_s(self) -> float:
        """Return the source's exact fixed-step duration authority."""
        return cast(
            float,
            effective_rk4_duration(self._base_simulation_config.swing_duration_s),
        )

    def _refresh_row_contexts(self) -> None:
        """Update every row from one source/mode locus-authoring decision."""
        enabled = self._localized_authoring_enabled()
        duration_s = self._localized_duration_s()
        for row in self._rows:
            row.set_context(self.mode(), enabled, duration_s)

    def _on_noise_editor_changed(self) -> None:
        """Invalidate paired evidence after an authored source changes."""
        if hasattr(self, "_attribution_generation"):
            self._invalidate_attribution(
                "Variation plan changed; prior paired authority was cleared."
            )

    def _invalidate_attribution(self, reason: str) -> None:
        """Invalidate paired evidence; provided by the concrete controller."""
        raise NotImplementedError
