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
        self.setMinimumWidth(300)
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

        layout.addWidget(posture_group)

        # Re-apply Button
        apply_btn = QPushButton("Apply Initial Stance & Reset")
        apply_btn.clicked.connect(self.apply_stance)
        layout.addWidget(apply_btn)

        # IAA Demo Button
        iaa_btn = QPushButton("Run Induced Acceleration Trigger (Console)")
        iaa_btn.clicked.connect(self.run_iaa)
        layout.addWidget(iaa_btn)

        # Setup timer for simulator stepping
        self.timer = QTimer()
        self.timer.timeout.connect(self.step_sim)
        self.timer.start(int(sim.model.opt.timestep * 1000))

        # Lock for sim access
        self.sim_lock = threading.Lock()

        # Initial apply
        self.apply_stance()

    def apply_stance(self) -> None:
        tilt = self.tilt_slider.value()
        knee = self.knee_slider.value()
        foot = self.foot_slider.value()

        with self.sim_lock:
            # We reset physics time and state entirely
            mujoco.mj_resetData(self.sim.model, self.sim.data)

            # Re-apply requested stance constraints
            self.sim.setup_initial_pose(
                hip_anterior_tilt=tilt,
                knee_flexion=knee,
                foot_angle=foot,
            )

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
            # Step the simulation mathematically with our stability controls
            self.sim.step()

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
