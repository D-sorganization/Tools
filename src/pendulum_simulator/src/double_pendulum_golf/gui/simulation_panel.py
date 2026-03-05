"""
Shared simulation panel for double and triple pendulum tabs.
"""

import csv
import os
import shutil
import subprocess
import tempfile
from typing import Callable

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QMessageBox, QSplitter,
    QFileDialog, QApplication,
)


class SimulationPanel(QWidget):
    """Reusable panel that hosts controls, pendulum, and matrix widgets."""

    ANIMATION_INTERVAL_MS = 16  # ~60 fps

    def __init__(
        self,
        controls: QWidget,
        pendulum: QWidget,
        matrix: QWidget,
        params_builder: Callable[[dict], object],
        torque_builder: Callable[[dict], Callable],
        state_builder: Callable[[dict], np.ndarray],
        run_simulation: Callable,
        torque_history: QWidget | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.controls = controls
        self.pendulum = pendulum
        self.matrix = matrix
        self.torque_history = torque_history
        self._params_builder = params_builder
        self._torque_builder = torque_builder
        self._state_builder = state_builder
        self._run_simulation = run_simulation

        self._result = None
        self._anim_idx = 0
        self._playback_speed = 1.0

        self._build_ui()
        self._connect_signals()
        self._setup_timer()

    def _build_ui(self) -> None:
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.controls)
        splitter.addWidget(self.pendulum)
        splitter.addWidget(self.matrix)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 1)

        if self.torque_history is not None:
            splitter.addWidget(self.torque_history)
            splitter.setStretchFactor(3, 1)
            splitter.setSizes([320, 520, 280, 280])
        else:
            splitter.setSizes([320, 520, 360])

        main_layout.addWidget(splitter)

    def _connect_signals(self) -> None:
        self.controls.run_requested.connect(self._on_run)
        self.controls.reset_requested.connect(self._on_reset)
        self.controls.play_toggled.connect(self._on_play_toggle)
        self.controls.speed_changed.connect(self._on_speed_change)
        self.controls.frame_changed.connect(self._on_frame_change)
        self.controls.export_data_requested.connect(self._on_export_data)
        self.controls.export_video_requested.connect(self._on_export_video)

    def _setup_timer(self) -> None:
        self._timer = QTimer(self)
        self._timer.setInterval(self.ANIMATION_INTERVAL_MS)
        self._timer.timeout.connect(self._advance_frame)

    def _on_run(self) -> None:
        try:
            p = self.controls.get_params()
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", str(e))
            return

        try:
            params = self._params_builder(p)
        except AssertionError as e:
            QMessageBox.warning(self, "Parameter Error", str(e))
            return

        if p["t_end"] <= 0:
            QMessageBox.warning(self, "Input Error", "Duration must be positive")
            return

        initial_state = self._state_builder(p)
        torque_func = self._torque_builder(p)

        self.controls.btn_run.setEnabled(False)

        try:
            result = self._run_simulation(
                params=params,
                initial_state=initial_state,
                t_end=p["t_end"],
                torque_func=torque_func,
                dt=0.005,
            )
        except Exception as e:
            QMessageBox.critical(self, "Simulation Error", str(e))
            self.controls.btn_run.setEnabled(True)
            return

        self._result = result
        self._anim_idx = 0

        self.pendulum.set_simulation(result)
        self.matrix.set_simulation(result)
        if self.torque_history is not None:
            self.torque_history.set_simulation(result)
        self.controls.set_slider_range(result.n_steps - 1)
        self.controls.set_slider_value(0)
        self._display_frame(0)
        self.controls.btn_run.setEnabled(True)

        # Auto-play the simulation
        self.controls.btn_play.setChecked(True)

    def _on_reset(self) -> None:
        self._timer.stop()
        self._result = None
        self._anim_idx = 0
        self.pendulum.clear()
        self.matrix.clear()
        if self.torque_history is not None:
            self.torque_history.clear()
        self.controls.stop_playback()
        self.controls.set_slider_value(0)

    def _on_play_toggle(self, playing: bool) -> None:
        if self._result is None:
            self.controls.stop_playback()
            return
        if playing:
            if self._anim_idx >= self._result.n_steps - 1:
                self._anim_idx = 0
                if hasattr(self.pendulum, "_trail"):
                    self.pendulum._trail.clear()
            self._timer.start()
        else:
            self._timer.stop()

    def _on_speed_change(self, speed: float) -> None:
        self._playback_speed = speed

    def _on_frame_change(self, frame: int) -> None:
        if self._result is None:
            return
        self._anim_idx = frame
        if hasattr(self.pendulum, "_trail"):
            self.pendulum._trail.clear()
            trail_start = max(0, frame - getattr(self.pendulum, "TRAIL_LENGTH", 0))
            for i in range(trail_start, frame + 1):
                pos = self._result.positions_at(i)
                tip = pos.get("tip")
                if tip is not None:
                    self.pendulum._trail.append(tip)
        self._display_frame(frame)

    def _advance_frame(self) -> None:
        if self._result is None:
            self._timer.stop()
            return

        frames_per_tick = max(1, int(self._playback_speed * 3))
        self._anim_idx += frames_per_tick

        if self._anim_idx >= self._result.n_steps:
            self._anim_idx = self._result.n_steps - 1
            self._timer.stop()
            self.controls.stop_playback()

        self._display_frame(self._anim_idx)
        self.controls.set_slider_value(self._anim_idx)

    def _display_frame(self, idx: int) -> None:
        assert self._result is not None
        idx = max(0, min(idx, self._result.n_steps - 1))
        self.pendulum.set_frame(idx)
        self.matrix.set_frame(idx)
        if self.torque_history is not None:
            self.torque_history.set_frame(idx)

    def _on_export_data(self) -> None:
        if self._result is None:
            QMessageBox.information(self, "Export Data", "Run a simulation first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Data", "", "CSV Files (*.csv)"
        )
        if not path:
            return

        headers = ["t"]
        if self._result.states.shape[1] == 4:
            headers += [
                "tau_drive_1", "tau_drive_2",
                "tau_friction_1", "tau_friction_2",
                "tau_total_1", "tau_total_2",
                "shoulder_fx", "shoulder_fy",
                "wrist_fx", "wrist_fy",
            ]
        else:
            headers += [
                "tau_drive_1", "tau_drive_2", "tau_drive_3",
                "tau_friction_1", "tau_friction_2", "tau_friction_3",
                "tau_total_1", "tau_total_2", "tau_total_3",
                "shoulder_fx", "shoulder_fy",
                "wrist1_fx", "wrist1_fy",
                "wrist2_fx", "wrist2_fy",
            ]

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                for i in range(self._result.n_steps):
                    t = self._result.t[i]
                    tau_drive = self._result.torques_at(i)
                    forces = self._result.joint_forces_at(i)

                    if self._result.states.shape[1] == 4:
                        tau_friction = self._result.friction_torques_at(i)
                        tau_total = self._result.total_torques_at(i)
                        row = [
                            t,
                            tau_drive[0], tau_drive[1],
                            tau_friction[0], tau_friction[1],
                            tau_total[0], tau_total[1],
                            forces["shoulder"][0], forces["shoulder"][1],
                            forces["wrist"][0], forces["wrist"][1],
                        ]
                    else:
                        row = [
                            t,
                            tau_drive[0], tau_drive[1], tau_drive[2],
                            0.0, 0.0, 0.0,  # friction not yet in triple model
                            tau_drive[0], tau_drive[1], tau_drive[2],
                            forces["shoulder"][0], forces["shoulder"][1],
                            forces["wrist1"][0], forces["wrist1"][1],
                            forces["wrist2"][0], forces["wrist2"][1],
                        ]
                    writer.writerow(row)

        except OSError as e:
            QMessageBox.critical(self, "Export Data", f"Failed to write file: {e}")
            return

        QMessageBox.information(self, "Export Data", f"Saved data to:\n{path}")

    def _on_export_video(self) -> None:
        if self._result is None:
            QMessageBox.information(self, "Export Video", "Run a simulation first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Video", "", "MP4 Video (*.mp4);;GIF (*.gif)"
        )
        if not path:
            return

        ffmpeg_path = shutil.which("ffmpeg")
        was_playing = self._timer.isActive()
        self._timer.stop()

        tmp_dir = tempfile.mkdtemp(prefix="pendulum_frames_")
        try:
            for i in range(self._result.n_steps):
                self._display_frame(i)
                QApplication.processEvents()
                pix = self.pendulum.grab()
                frame_path = os.path.join(tmp_dir, f"frame_{i:05d}.png")
                pix.save(frame_path)

            if ffmpeg_path is None:
                out_dir = os.path.splitext(path)[0] + "_frames"
                os.makedirs(out_dir, exist_ok=True)
                for name in os.listdir(tmp_dir):
                    shutil.move(os.path.join(tmp_dir, name), os.path.join(out_dir, name))
                QMessageBox.warning(
                    self,
                    "Export Video",
                    "ffmpeg not found. Exported PNG frames instead:\n" + out_dir,
                )
                return

            fps = int(1000 / self.ANIMATION_INTERVAL_MS)
            cmd = [
                ffmpeg_path,
                "-y",
                "-framerate",
                str(fps),
                "-i",
                os.path.join(tmp_dir, "frame_%05d.png"),
                "-pix_fmt",
                "yuv420p",
                path,
            ]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if result.returncode != 0:
                QMessageBox.critical(
                    self,
                    "Export Video",
                    "ffmpeg failed. Check your ffmpeg installation.",
                )
                return

            QMessageBox.information(self, "Export Video", f"Saved video to:\n{path}")
        finally:
            if was_playing:
                self._timer.start()
            shutil.rmtree(tmp_dir, ignore_errors=True)
