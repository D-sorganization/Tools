# ruff: noqa: T201
"""
Lower Body Model - MuJoCo Simulation Viewer
"""

import argparse

from lower_body_model.builder import build_lower_body_xml
from lower_body_model.simulator import LowerBodySimulator


def main() -> None:
    parser = argparse.ArgumentParser(description="Lower Body Model Simulation")
    parser.add_argument(
        "--gui", action="store_true", help="Launch MuJoCo interactive viewer"
    )
    parser.add_argument(
        "--demo-iaa",
        action="store_true",
        help="Run a demo Induced Acceleration Analysis",
    )
    args = parser.parse_args()

    xml = build_lower_body_xml()
    sim = LowerBodySimulator(xml)

    # Initialize posture and set stability targets
    sim.setup_initial_pose(hip_anterior_tilt=30.0, knee_flexion=120.0, foot_angle=20.0)

    if args.demo_iaa:
        print(
            "Running Induced Acceleration Analysis on right hip (X axis) with 10.0 Nm torque..."
        )  # noqa: T201
        iaa = sim.analyze_induced_acceleration("act_r_hip_x", 10.0)
        print("Pelvis Induced Accelerations:")  # noqa: T201
        for k, v in iaa.items():
            print(f"  {k}: {v:.4f}")  # noqa: T201

    if args.gui:
        try:
            import mujoco.viewer

            print("Launching MuJoCo Viewer. Press ESC to exit.")  # noqa: T201
            mujoco.viewer.launch(sim.model, sim.data)
        except ImportError:
            print("mujoco.viewer not available. Make sure you install correctly.")  # noqa: T201

    if not args.gui and not args.demo_iaa:
        parser.print_help()


if __name__ == "__main__":
    main()
