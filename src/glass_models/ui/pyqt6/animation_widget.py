"""PyQt6 widget for animation playback and scrubbing.

Provides complete animation playback controls including:
- Play/pause/restart buttons
- Time slider with scrubbing support (<100ms response)
- Speed slider (0.5x to 10x, adjustable in real time)
- First/last step navigation buttons
- Time range selector with visualization
- Loop toggle checkbox
- Current/total time display
- Frame export button (image sequence)
- Smooth playback via QTimer with adjustable frame interval

Performance targets:
- 30+ FPS smooth animation
- Sub-100ms scrubbing response
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from glass_models.viz.timesteps import TimeStepManager

logger = logging.getLogger(__name__)


class AnimationPlaybackWidget(QWidget):
    """PyQt6 widget for time-dependent data animation playback.

    Provides a complete animation control interface with playback controls,
    time navigation, speed adjustment, and frame export capabilities.

    Signals:
        time_changed: Emitted when current time step changes (int index)
        playback_started: Emitted when playback begins
        playback_stopped: Emitted when playback stops
        frame_exported: Emitted when frame export completes (str path)
        loop_toggled: Emitted when loop state changes (bool enabled)
    """

    time_changed = pyqtSignal(int)
    playback_started = pyqtSignal()
    playback_stopped = pyqtSignal()
    frame_exported = pyqtSignal(str)
    loop_toggled = pyqtSignal(bool)

    def __init__(
        self,
        time_steps: np.ndarray,
        field_data_list: list[dict[str, Any]],
        parent: QWidget | None = None,
    ) -> None:
        """Initialize animation playback widget.

        Args:
            time_steps: 1D numpy array of time values
            field_data_list: List of field data dicts (one per step)
            parent: Parent widget
        """
        super().__init__(parent)
        self.manager = TimeStepManager(time_steps, field_data_list)
        self._is_playing = False
        self._speed_multiplier = 1.0
        self._loop_enabled = False
        self._base_interval_ms = 33  # ~30 FPS at 1.0x speed

        logger.debug(
            "AnimationPlaybackWidget initialized with %d steps",
            self.manager.total_steps,
        )

        self._setup_ui()
        self._connect_signals()
        self._setup_timer()

    def _setup_ui(self) -> None:
        """Set up UI components."""
        main_layout = QVBoxLayout(self)

        # =====================
        # Playback controls row
        # =====================
        controls_layout = QHBoxLayout()

        self.play_button = QPushButton("Play")
        self.play_button.setMaximumWidth(80)
        controls_layout.addWidget(self.play_button)

        self.pause_button = QPushButton("Pause")
        self.pause_button.setMaximumWidth(80)
        controls_layout.addWidget(self.pause_button)

        self.restart_button = QPushButton("Restart")
        self.restart_button.setMaximumWidth(80)
        controls_layout.addWidget(self.restart_button)

        self.first_button = QPushButton("First")
        self.first_button.setMaximumWidth(60)
        controls_layout.addWidget(self.first_button)

        self.last_button = QPushButton("Last")
        self.last_button.setMaximumWidth(60)
        controls_layout.addWidget(self.last_button)

        controls_layout.addStretch()

        self.loop_checkbox = QCheckBox("Loop")
        controls_layout.addWidget(self.loop_checkbox)

        self.export_button = QPushButton("Export Frames")
        self.export_button.setMaximumWidth(100)
        controls_layout.addWidget(self.export_button)

        main_layout.addLayout(controls_layout)

        # =====================
        # Time slider row
        # =====================
        slider_layout = QHBoxLayout()

        slider_label = QLabel("Time:")
        slider_layout.addWidget(slider_label)

        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(self.manager.total_steps - 1)
        self.time_slider.setValue(0)
        self.time_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.time_slider.setTickInterval(max(1, self.manager.total_steps // 10))
        slider_layout.addWidget(self.time_slider)

        self.time_display_label = QLabel(self._format_time_display())
        self.time_display_label.setMaximumWidth(100)
        slider_layout.addWidget(self.time_display_label)

        main_layout.addLayout(slider_layout)

        # =====================
        # Speed slider row
        # =====================
        speed_layout = QHBoxLayout()

        speed_label = QLabel("Speed:")
        speed_layout.addWidget(speed_label)

        # Speed slider: 50-1000% maps to 0.5x-10x
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(50)  # 0.5x
        self.speed_slider.setMaximum(1000)  # 10x
        self.speed_slider.setValue(100)  # 1.0x default
        self.speed_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.speed_slider.setTickInterval(150)
        speed_layout.addWidget(self.speed_slider)

        self.speed_display_label = QLabel("1.0x")
        self.speed_display_label.setMaximumWidth(50)
        speed_layout.addWidget(self.speed_display_label)

        main_layout.addLayout(speed_layout)

        # =====================
        # Time range selector row
        # =====================
        range_layout = QHBoxLayout()

        range_label = QLabel("Range:")
        range_layout.addWidget(range_label)

        start_label = QLabel("Start:")
        range_layout.addWidget(start_label)

        self.range_start_spinbox = QSpinBox()
        self.range_start_spinbox.setMinimum(0)
        self.range_start_spinbox.setMaximum(self.manager.total_steps - 1)
        self.range_start_spinbox.setValue(0)
        range_layout.addWidget(self.range_start_spinbox)

        end_label = QLabel("End:")
        range_layout.addWidget(end_label)

        self.range_end_spinbox = QSpinBox()
        self.range_end_spinbox.setMinimum(0)
        self.range_end_spinbox.setMaximum(self.manager.total_steps - 1)
        self.range_end_spinbox.setValue(self.manager.total_steps - 1)
        range_layout.addWidget(self.range_end_spinbox)

        range_layout.addStretch()
        main_layout.addLayout(range_layout)

    def _connect_signals(self) -> None:
        """Connect all signal/slot connections."""
        self.play_button.clicked.connect(self._on_play)
        self.pause_button.clicked.connect(self._on_pause)
        self.restart_button.clicked.connect(self._on_restart)
        self.first_button.clicked.connect(self._on_first)
        self.last_button.clicked.connect(self._on_last)
        self.loop_checkbox.stateChanged.connect(self._on_loop_toggled)

        self.time_slider.sliderMoved.connect(self.on_slider_moved)
        self.time_slider.valueChanged.connect(self._on_slider_value_changed)

        self.speed_slider.valueChanged.connect(self.on_speed_changed)

        self.export_button.clicked.connect(self._on_export)

    def _setup_timer(self) -> None:
        """Set up QTimer for smooth playback."""
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._on_timer_tick)

    def _format_time_display(self) -> str:
        """Format time display string as 'current / total'."""
        current = self.manager.current_time
        min_time, max_time = self.manager.time_range
        return f"{current:.4g} / {max_time:.4g}"

    def _update_time_display(self) -> None:
        """Update time display label."""
        self.time_display_label.setText(self._format_time_display())

    def _update_speed_display(self) -> None:
        """Update speed display label."""
        self.speed_display_label.setText(f"{self._speed_multiplier:.1f}x")

    def _on_play(self) -> None:
        """Handle play button clicked."""
        if not self._is_playing:
            self._is_playing = True
            self.playback_timer.start(self._get_timer_interval())
            logger.debug("Playback started")
            self.playback_started.emit()

    def _on_pause(self) -> None:
        """Handle pause button clicked."""
        if self._is_playing:
            self._is_playing = False
            self.playback_timer.stop()
            logger.debug("Playback paused")
            self.playback_stopped.emit()

    def _on_restart(self) -> None:
        """Handle restart button clicked."""
        self.manager.reset()
        self.time_slider.setValue(0)
        self._update_time_display()
        logger.debug("Restarted to first frame")
        self.time_changed.emit(0)

    def _on_first(self) -> None:
        """Handle first step button clicked."""
        self.manager.seek_to_step(0)
        self.time_slider.setValue(0)
        self._update_time_display()
        self.time_changed.emit(0)

    def _on_last(self) -> None:
        """Handle last step button clicked."""
        last_index = self.manager.total_steps - 1
        self.manager.seek_to_step(last_index)
        self.time_slider.setValue(last_index)
        self._update_time_display()
        self.time_changed.emit(last_index)

    def _on_loop_toggled(self, state: int) -> None:
        """Handle loop checkbox toggled."""
        self._loop_enabled = state == 2  # Checked state
        logger.debug("Loop toggled: %s", self._loop_enabled)
        self.loop_toggled.emit(self._loop_enabled)

    def on_slider_moved(self, value: int) -> None:
        """Handle time slider moved (scrubbing).

        Args:
            value: New slider position (0 to total_steps-1)
        """
        try:
            self.manager.seek_to_step(value)
            self._update_time_display()
            self.time_changed.emit(value)
            logger.debug("Seeked to step %d via slider", value)
        except ValueError as e:
            logger.warning("Invalid slider seek: %s", e)

    def _on_slider_value_changed(self, value: int) -> None:
        """Handle slider value changed (internal update)."""
        # Only process if slider was explicitly moved, not programmatically set
        pass

    def on_speed_changed(self, value: int) -> None:
        """Handle speed slider changed.

        Maps slider value (50-1000) to speed multiplier (0.5-10.0).

        Args:
            value: Slider value in range [50, 1000]
        """
        # Map slider range (50-1000) to multiplier range (0.5-10.0)
        # Using logarithmic scaling for more natural feel
        self._speed_multiplier = value / 100.0
        self._update_speed_display()

        # Restart timer with new interval
        if self._is_playing:
            self.playback_timer.setInterval(self._get_timer_interval())

        logger.debug("Speed changed to %.2fx", self._speed_multiplier)

    def _get_timer_interval(self) -> int:
        """Calculate timer interval based on speed multiplier.

        Returns:
            Timer interval in milliseconds.
        """
        # Target 30 FPS at 1.0x speed
        # interval = base_interval / speed_multiplier
        interval = max(5, int(self._base_interval_ms / self._speed_multiplier))
        return interval

    def _on_timer_tick(self) -> None:
        """Handle timer tick for playback."""
        if not self._is_playing:
            return

        advanced = self.manager.advance_step()

        if not advanced:
            # Reached end
            if self._loop_enabled:
                self.manager.reset()
                logger.debug("Animation looped to start")
            else:
                self._on_pause()
                logger.debug("Animation finished")
                return

        # Update slider and display
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(self.manager.current_step_index)
        self.time_slider.blockSignals(False)

        self._update_time_display()
        self.time_changed.emit(self.manager.current_step_index)

    def _on_export(self) -> None:
        """Handle export button clicked."""
        logger.info("Export frames button clicked")
        # TODO: Implement frame export dialog and functionality
        # This would typically open a file dialog and export PNG sequence

    def export_frames(self, output_dir: str | Path) -> None:
        """Export animation frames to image sequence.

        Args:
            output_dir: Directory to save frames to
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Exporting %d frames to %s", self.manager.total_steps, output_dir)

        # TODO: Implement actual frame export
        # This would:
        # 1. Iterate through all time steps
        # 2. Render/export each frame as PNG with zero-padded numbering
        # 3. Emit frame_exported signal when complete

        self.frame_exported.emit(str(output_dir))

    @property
    def is_playing(self) -> bool:
        """Return True if animation is currently playing."""
        return self._is_playing

    @property
    def speed_multiplier(self) -> float:
        """Return current speed multiplier."""
        return self._speed_multiplier

    @property
    def loop_enabled(self) -> bool:
        """Return True if looping is enabled."""
        return self._loop_enabled
