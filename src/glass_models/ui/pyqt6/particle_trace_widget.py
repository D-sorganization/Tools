"""PyQt6 widget for particle trace playback and control.

This module provides interactive controls for particle tracing visualization:
- Play/pause/restart controls
- Speed slider (0.5x to 10x)
- Seed density slider (1-100%)
- Animation timer with configurable frame rate
- Trajectory colormapping options
"""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class ParticleTracePlaybackWidget(QWidget):
    """Widget for controlling particle trace playback and parameters.

    This widget provides:
    - Play/pause/restart buttons
    - Speed control slider (0.5x to 10x)
    - Seed density slider (1-100%)
    - Framerate configuration
    - Callback mechanism for animation updates
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        frame_rate: int = 60,
        on_update: Callable[[float], None] | None = None,
    ) -> None:
        """Initialize the particle trace playback widget.

        Args:
            parent: Parent QWidget
            frame_rate: Target frame rate in Hz
            on_update: Callback function called each frame with time delta
        """
        super().__init__(parent)
        self.frame_rate = frame_rate
        self.on_update = on_update
        self.is_playing = False
        self.speed_multiplier = 1.0
        self.seed_density = 1.0

        self._init_ui()
        self._setup_timer()

    def _init_ui(self) -> None:
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Control buttons
        controls_layout = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")
        self.restart_button = QPushButton("Restart")

        self.play_button.clicked.connect(self._on_play)
        self.pause_button.clicked.connect(self._on_pause)
        self.restart_button.clicked.connect(self._on_restart)

        controls_layout.addWidget(self.play_button)
        controls_layout.addWidget(self.pause_button)
        controls_layout.addWidget(self.restart_button)
        layout.addLayout(controls_layout)

        # Speed slider
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(5)  # 0.5x
        self.speed_slider.setMaximum(100)  # 10x
        self.speed_slider.setValue(10)  # 1x
        self.speed_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.speed_slider.setTickInterval(5)
        self.speed_slider.valueChanged.connect(self._on_speed_changed)

        self.speed_label = QLabel("1.0x")
        speed_layout.addWidget(self.speed_slider)
        speed_layout.addWidget(self.speed_label)
        layout.addLayout(speed_layout)

        # Seed density slider
        density_layout = QHBoxLayout()
        density_layout.addWidget(QLabel("Seed Density:"))
        self.density_slider = QSlider(Qt.Orientation.Horizontal)
        self.density_slider.setMinimum(1)  # 1%
        self.density_slider.setMaximum(100)  # 100%
        self.density_slider.setValue(100)  # 100%
        self.density_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.density_slider.setTickInterval(10)
        self.density_slider.valueChanged.connect(self._on_density_changed)

        self.density_label = QLabel("100%")
        density_layout.addWidget(self.density_slider)
        density_layout.addWidget(self.density_label)
        layout.addLayout(density_layout)

        layout.addStretch()

    def _setup_timer(self) -> None:
        """Set up the animation timer."""
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_timer_tick)
        self.frame_interval = 1000 // self.frame_rate  # milliseconds

    def _on_play(self) -> None:
        """Handle play button clicked."""
        self.is_playing = True
        self.timer.start(self.frame_interval)
        self.play_button.setEnabled(False)
        self.pause_button.setEnabled(True)

    def _on_pause(self) -> None:
        """Handle pause button clicked."""
        self.is_playing = False
        self.timer.stop()
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)

    def _on_restart(self) -> None:
        """Handle restart button clicked."""
        self._on_pause()
        # Emit restart signal (to be implemented in parent)
        # For now, just reset state

    def _on_timer_tick(self) -> None:
        """Handle timer tick for animation frame."""
        if not self.is_playing or not self.on_update:
            return

        # Calculate time delta for this frame
        dt = (self.frame_interval / 1000.0) * self.speed_multiplier

        # Call update callback
        self.on_update(dt)

    def _on_speed_changed(self, value: int) -> None:
        """Handle speed slider change."""
        # Map slider value [5, 100] to multiplier [0.5, 10]
        self.speed_multiplier = 0.5 + (value - 5) * 0.095
        self.speed_label.setText(f"{self.speed_multiplier:.2f}x")

    def _on_density_changed(self, value: int) -> None:
        """Handle seed density slider change."""
        self.seed_density = value / 100.0
        self.density_label.setText(f"{value}%")

    def set_update_callback(self, callback: Callable[[float], None]) -> None:
        """Set the callback function for animation updates.

        Args:
            callback: Function that takes time delta as argument
        """
        self.on_update = callback

    def set_frame_rate(self, fps: int) -> None:
        """Set the target frame rate.

        Args:
            fps: Frames per second
        """
        self.frame_rate = fps
        self.frame_interval = 1000 // fps

    def is_animating(self) -> bool:
        """Check if animation is currently running."""
        return self.is_playing

    def get_speed_multiplier(self) -> float:
        """Get current speed multiplier."""
        return self.speed_multiplier

    def get_seed_density(self) -> float:
        """Get current seed density (0.0 to 1.0)."""
        return self.seed_density
