"""
Lower Body Model - PyQt6/PySide6 GUI Launcher
"""

import logging
import sys
import threading

import mujoco
from mujoco import viewer
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from lower_body_model.builder import build_lower_body_xml
from lower_body_model.simulator import LowerBodySimulator


class ControlPanel(QMainWindow):
    """
    Control Panel for Lower Body Simulator.
    Allows real-time tweaking of initial posture and applies basic PD stability.
    """

    def __init__(self, sim: LowerBodySimulator, mujoco_viewer: viewer.Handle) -> None:
        super().__init__()
        self.sim = sim
        self.viewer = mujoco_viewer

        self.setWindowTitle("Lower Body Control Panel")
        self.setMinimumWidth(350)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Posture Controls
        posture_group = QGroupBox("Initial Posture Parameters (Degrees)")
        posture_layout = QFormLayout(posture_group)

        # 1. Anterior Tilt
        self.tilt_slider = QSlider(Qt.Orientation.Horizontal)
        self.tilt_slider.setMinimum(-50)
        self.tilt_slider.setMaximum(50)
        self.tilt_slider.setValue(30)
        self.tilt_lbl = QLabel("30")
        self.tilt_slider.valueChanged.connect(lambda v: self.tilt_lbl.setText(str(v)))
        posture_layout.addRow("Hip Anterior Tilt:", self.tilt_slider)
        posture_layout.addRow("", self.tilt_lbl)

        # 2. Knee Flexion
        self.knee_slider = QSlider(Qt.Orientation.Horizontal)
        self.knee_slider.setMinimum(0)
        self.knee_slider.setMaximum(150)
        self.knee_slider.setValue(120)
        self.knee_lbl = QLabel("120")
        self.knee_slider.valueChanged.connect(lambda v: self.knee_lbl.setText(str(v)))
        posture_layout.addRow("Knee Flexion:", self.knee_slider)
        posture_layout.addRow("", self.knee_lbl)

        # 3. Foot Angle
        self.foot_slider = QSlider(Qt.Orientation.Horizontal)
        self.foot_slider.setMinimum(-45)
        self.foot_slider.setMaximum(45)
        self.foot_slider.setValue(20)
        self.foot_lbl = QLabel("20")
        self.foot_slider.valueChanged.connect(lambda v: self.foot_lbl.setText(str(v)))
        posture_layout.addRow("Foot Extern Rot:", self.foot_slider)
        posture_layout.addRow("", self.foot_lbl)

        # PD Gains
        pd_group = QGroupBox("Control Gains")
        pd_layout = QFormLayout(pd_group)

        self.kp_slider = QSlider(Qt.Orientation.Horizontal)
        self.kp_slider.setMinimum(0)
        self.kp_slider.setMaximum(2000)
        self.kp_slider.setValue(int(self.sim.kp_stability))
        self.kp_lbl = QLabel(str(int(self.sim.kp_stability)))
        self.kp_slider.valueChanged.connect(self.update_gains)
        pd_layout.addRow("Kp Stiffness:", self.kp_slider)
        pd_layout.addRow("", self.kp_lbl)

        self.kd_slider = QSlider(Qt.Orientation.Horizontal)
        self.kd_slider.setMinimum(0)
        self.kd_slider.setMaximum(300)
        self.kd_slider.setValue(int(self.sim.kd_stability))
        self.kd_lbl = QLabel(str(int(self.sim.kd_stability)))
        self.kd_slider.valueChanged.connect(self.update_gains)
        pd_layout.addRow("Kd Damping:", self.kd_slider)
        pd_layout.addRow("", self.kd_lbl)

        layout.addWidget(pd_group)

        # Playback Controls
        play_group = QGroupBox("Playback")
        play_layout = QVBoxLayout(play_group)

        self.is_playing = False
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        play_layout.addWidget(self.play_btn)

        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(0)
        self.timeline_slider.setEnabled(True)
        self.timeline_slider.valueChanged.connect(self.scrub_timeline)
        play_layout.addWidget(QLabel("Scrub History:"))
        play_layout.addWidget(self.timeline_slider)

        layout.addWidget(play_group)

        # Re-apply Button
        apply_btn = QPushButton("Apply Initial Stance & Reset")
        apply_btn.clicked.connect(self.apply_stance)
        layout.addWidget(apply_btn)

        # IAA Demo Button
        iaa_btn = QPushButton("Run Induced Acceleration Trigger (Console)")
        iaa_btn.clicked.connect(self.run_iaa)
        layout.addWidget(iaa_btn)

        # Setup timer for simulator stepping (40 Hz = 25ms)
        self.timer = QTimer()
        self.timer.timeout.connect(self.step_sim)
        self.timer.start(25)

        # Lock for sim access
        self.sim_lock = threading.Lock()

        self.ui_update_counter = 0

        # Initial apply
        self.apply_stance()

    def update_gains(self) -> None:
        val_kp = self.kp_slider.value()
        val_kd = self.kd_slider.value()
        self.kp_lbl.setText(str(val_kp))
        self.kd_lbl.setText(str(val_kd))
        with self.sim_lock:
            self.sim.kp_stability = float(val_kp)
            self.sim.kd_stability = float(val_kd)

    def toggle_play(self) -> None:
        self.is_playing = not self.is_playing
        self.play_btn.setText("Pause" if self.is_playing else "Play")
        self.timeline_slider.setEnabled(not self.is_playing)
        with self.sim_lock:
            if not self.is_playing and self.sim.history:
                self.timeline_slider.setMaximum(len(self.sim.history) - 1)
                self.timeline_slider.setValue(len(self.sim.history) - 1)

    def scrub_timeline(self) -> None:
        if self.is_playing:
            return
        idx = self.timeline_slider.value()
        with self.sim_lock:
            self.sim.restore_frame(idx)
            self.viewer.sync()

    def apply_stance(self) -> None:
        tilt = self.tilt_slider.value()
        knee = self.knee_slider.value()
        foot = self.foot_slider.value()

        with self.sim_lock:
            mujoco.mj_resetData(self.sim.model, self.sim.data)
            self.sim.clear_history()

            self.sim.setup_initial_pose(
                hip_anterior_tilt=tilt,
                knee_flexion=knee,
                foot_angle=foot,
            )

            # Reset timeline state
            if not self.is_playing:
                self.timeline_slider.setMaximum(0)
                self.timeline_slider.setValue(0)

    def run_iaa(self) -> None:
        with self.sim_lock:
            logging.info("--- Induced Acceleration Analysis ---")
            iaa = self.sim.analyze_induced_acceleration("act_r_hip_x", 10.0)
            for k, v in iaa.items():
                logging.info(f"  {k}: {v:.4f}")

    def step_sim(self) -> None:
        if not self.viewer.is_running():
            self.close()
            return

        with self.sim_lock:
            if self.is_playing:
                # 25ms timer / 5ms physics = 5 steps per visual frame
                for _ in range(5):
                    self.sim.step()

                # Update UI scrubber bounds sporadically
                self.ui_update_counter += 1
                if self.ui_update_counter >= 10:
                    self.ui_update_counter = 0
                    if (
                        not self.timeline_slider.underMouse()
                        and self.timeline_slider.isEnabled()
                    ):
                        # Just to keep max range correct occasionally
                        pass

            # Periodically sync to the visualizer
            self.viewer.sync()


def main() -> None:
    app = QApplication(sys.argv)

    xml = build_lower_body_xml()
    sim = LowerBodySimulator(xml)

    # Launch passive viewer in its own thread block
    mj_viewer = viewer.launch_passive(sim.model, sim.data)

    # Create PyQt interface
    window = ControlPanel(sim, mj_viewer)
    window.show()

    # Block on UI until closed
    app.exec()

    # Close viewer when UI is closed
    if mj_viewer.is_running():
        mj_viewer.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
