# Lower Body Model Specification

## 1. Overview

The **Lower Body Model** is a 3D biomechanical simulation of a golfer's lower body. Currently deployed natively in Python with the MuJoCo physics engine, it aims to accurately model lower extremity kinematics and dynamics for swing optimization, Counterfactual analysis, and Induced Acceleration Analysis (IAA).

## 2. Architecture & Design Principles

- **DRY (Don't Repeat Yourself)**: The MuJoCo XML is constructed dynamically by `builder.py`, generating symmetrical left/right leg elements parametrically based on identical sub-routines mapped onto python strings.
- **LOD (Law of Demeter)**: Components interact precisely through restricted interfaces. `main.py` interfaces with `LowerBodySimulator`, and only `LowerBodySimulator` manipulates `mujoco.MjData` and `mujoco.MjModel`.
- **TDD (Test-Driven Development)**: Comprehensive coverage validates ZTCF calculation outputs, inverse kinematics stability, valid polynomial joint drivers, and mass-integrity matching inside XML generation.
- **DbC (Design by Contract)**: Methods like `setup_initial_pose` use pre-condition assertions (e.g., `-90 <= foot_angle <= 90`) to prevent unrealistic biomechanical configurations that break structural simulation limits.

## 3. Kinematic Implementation

- **Pelvis (Root)**: 6-DOF Free Joint. Modeled as a tailored ellipsoid.
- **Hips**: Gimbal Joints (3 independent hinge actuators: X, Y, Z).
- **Knees**: Revolute Joints (1 hinge actuator, 0 to 150 flexion range).
- **Ankles**: Universal Joints (2 hinge actuators, X and Y). Flat ellipsoids replace basic box feet for anatomical resemblance.
- **Golf Hip Rotation Target**: `InclinedPlaneHipRotationTarget` samples a deterministic two-phase motion from 0 degrees to 45 degrees clockwise, then through 90 degrees of counterclockwise travel to +45 degrees on an inclined plane. `LowerBodySimulator` applies the target to both hip sockets through shared side iteration, reports target diagnostics during playback, and snapshots the target state into history frames for scrub-based verification. The PyQt control panel exposes a full reset action that stops playback, clears history, returns MuJoCo time to zero, preserves any loaded target, and reapplies the target pose at `t=0`.

## 4. Stability and Control

A fundamental mathematical PID loop stabilizes local targets for the joints. The solver executes Damped Least Squares inverse kinematics across the free-floating root relative to the static ground boundary, preserving symmetrical ground contact constraint solving. By default, the stance is loaded at approximately `30` degrees anterior pelvic tilt, `120` degrees knee flexion, and a `20`-degree out-flared foot rotation.
