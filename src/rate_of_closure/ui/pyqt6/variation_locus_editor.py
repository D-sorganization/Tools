"""Localized torque locus editor and precision-preserving merge authority."""

from __future__ import annotations

from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QWidget

from rate_of_closure.ui.pyqt6.variation_editor_widgets import make_spin
from shared.python.swing_sim.variation import (
    LOCALIZED_TORQUE_VARIABLE_JOINTS,
    NoiseSpec,
)

__all__ = ["LocalizedLocusEditor"]


class LocalizedLocusEditor(QWidget):
    """Author the one valid temporal/topological locus for a torque variable."""

    def __init__(self, duration_s: float) -> None:
        super().__init__()
        self._duration_s = duration_s
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Half-open window [start, end)"))
        self.window_start = make_spin(0.0, duration_s, 0.0, 9)
        self.window_start.setAccessibleName("Localized torque window start")
        self.window_start.setToolTip(
            "Inclusive start time [s] for the required half-open [start, end) window."
        )
        self.window_end = make_spin(0.0, duration_s, min(0.1, duration_s), 9)
        self.window_end.setAccessibleName("Localized torque window end")
        self.window_end.setToolTip(
            "Exclusive end time [s] for the required half-open [start, end) window."
        )
        self.joint_selector = QComboBox()
        self.joint_selector.setAccessibleName("Localized torque topological joint")
        self.joint_selector.setToolTip(
            "Stable topological torque joint. joint.* IDs are not spatial swing.* "
            "trace point IDs. The selected variable fixes this value."
        )
        self.joint_selector.setEnabled(False)
        for widget in (self.window_start, self.window_end):
            widget.setSuffix(" s")
            layout.addWidget(widget)
        layout.addWidget(self.joint_selector)

    @property
    def duration_s(self) -> float:
        """Return the current source-duration bound."""
        return self._duration_s

    def set_duration(self, duration_s: float) -> None:
        """Update the source-duration bound without coercing current values."""
        self._duration_s = duration_s
        for widget in (self.window_start, self.window_end):
            widget.setMaximum(duration_s)

    def set_variable(self, key: str | None) -> None:
        """Select the variable's sole stable joint and initialize its window."""
        joint_id = LOCALIZED_TORQUE_VARIABLE_JOINTS.get(key)
        self.setVisible(joint_id is not None)
        self.joint_selector.clear()
        if joint_id is not None:
            self.joint_selector.addItem(joint_id, joint_id)
            self.window_start.setValue(0.0)
            self.window_end.setValue(min(0.1, self._duration_s))

    def load_spec(self, spec: NoiseSpec) -> None:
        """Load an already validated localized window without changing its joint."""
        if spec.variable_key not in LOCALIZED_TORQUE_VARIABLE_JOINTS:
            return
        assert spec.time_window_s is not None
        self.window_start.setValue(spec.time_window_s[0])
        self.window_end.setValue(spec.time_window_s[1])

    def state(self) -> tuple[float, float, object]:
        """Return representable state used to detect which endpoint changed."""
        return (
            self.window_start.value(),
            self.window_end.value(),
            self.joint_selector.currentData(),
        )

    def accepts(
        self,
        spec: NoiseSpec,
        *,
        localized_enabled: bool,
        duration_s: float,
    ) -> bool:
        """Return whether a context can author the spec without substitution."""
        expected = LOCALIZED_TORQUE_VARIABLE_JOINTS.get(spec.variable_key)
        if expected is None:
            return True
        window = spec.time_window_s
        return (
            localized_enabled
            and window is not None
            and spec.point_ids == (expected,)
            and 0.0 <= window[0] < window[1] <= duration_s
        )

    def merged_locus(
        self,
        key: str,
        loaded: NoiseSpec | None,
        prior_state: tuple[float, float, object] | None,
        *,
        reset: bool,
    ) -> tuple[tuple[float, float] | None, tuple[str, ...]]:
        """Merge each visible endpoint with untouched exact loaded authority."""
        expected = LOCALIZED_TORQUE_VARIABLE_JOINTS.get(key)
        if expected is None:
            if reset:
                return None, ()
            return (
                None if loaded is None else loaded.time_window_s,
                () if loaded is None else loaded.point_ids,
            )
        start, end, _joint = self.state()
        if not 0.0 <= start < end <= self._duration_s:
            raise ValueError(
                "localized torque time window requires 0 <= start < end <= "
                f"double-pendulum duration {self._duration_s:g} s"
            )
        if (
            loaded is not None
            and loaded.variable_key == key
            and not reset
            and prior_state is not None
        ):
            assert loaded.time_window_s is not None
            loaded_start, loaded_end = loaded.time_window_s
            start = loaded_start if start == prior_state[0] else start
            end = loaded_end if end == prior_state[1] else end
        return (start, end), (expected,)
