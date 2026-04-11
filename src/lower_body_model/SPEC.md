# Lower Body Model Specification

## 1. Overview
The **Lower Body Model** is a 3D biomechanical simulation of a golfer's lower body. Currently deployed natively in Python with the MuJoCo physics engine, it aims to accurately model lower extremity kinematics and dynamics for swing optimization, Counterfactual analysis, and Induced Acceleration Analysis (IAA).

## 2. Architecture & Design Principles
- **DRY (Don't Repeat Yourself)**: The MuJoCo XML is constructed dynamically by `builder.py`, generating symmetrical left/right leg elements parametrically based on identical sub-routines mapped onto python strings.
- **LOD (Law of Demeter)**: Components interact precisely through restricted interfaces. `main.py` interfaces with `LowerBodySimulator`, and only `LowerBodySimulator` manipulates `mujoco.MjData` and `mujoco.MjModel`.
- **TDD (Test-Driven Development)**: Comprehensive coverage validates ZTCF calculation outputs, inverse kinematics stability, valid polynomial joint drivers, and mass-integrity matching inside XML generation.
- **DbC (Design by Contract)**: Public entry points raise `TypeError` for wrong types and `ValueError` for out-of-range values, per the repository-wide CLAUDE.md rule. `setup_initial_pose`, `inverse_kinematics`, `build_lower_body_xml`, `InclinedPlaneHipRotationTarget.__post_init__`, and `set_pelvis_inclined_rotation` enforce all preconditions this way.

## 3. Kinematic Implementation

- **Pelvis (Root)**: 6-DOF Free Joint. The pelvis body is rendered as an anatomically-inspired composite: a single inertial ellipsoid (`pelvis_body`, carrying all `pelvis_mass`) hosts five mass=0 visual-only landmark geoms — `pelvis_sacrum`, bilateral `pelvis_r_ilium`/`pelvis_l_ilium`, bright-red `pelvis_r_asis`/`pelvis_l_asis` spheres, and `pelvis_pubis`. Markers make anterior, posterior, obliquity, and axial rotation visually unambiguous without affecting dynamics.
- **Hips**: Gimbal Joints (3 independent hinge actuators: X, Y, Z).
- **Knees**: Revolute Joints (1 hinge actuator, 0 to 150 flexion range).
- **Ankles**: Universal Joints (2 hinge actuators, X and Y). Flat ellipsoids replace basic box feet for anatomical resemblance.
- **Golf Hip Rotation Target**: `InclinedPlaneHipRotationTarget` samples a deterministic two-phase motion from 0 degrees to 45 degrees clockwise, then through 90 degrees of counterclockwise travel to +45 degrees on an inclined plane. `LowerBodySimulator` applies the target to both hip sockets through shared side iteration, reports target diagnostics during playback, and snapshots the target state into history frames for scrub-based verification. The PyQt control panel exposes a full reset action that stops playback, clears history, returns MuJoCo time to zero, preserves any loaded target, and reapplies the target pose at `t=0`.

## 4. Stability and Control
A fundamental mathematical PID loop stabilizes local targets for the joints. The solver executes Damped Least Squares inverse kinematics across the free-floating root relative to the static ground boundary, preserving symmetrical ground contact constraint solving. By default, the stance is loaded at approximately `30` degrees anterior pelvic tilt, `120` degrees knee flexion, and a `20`-degree out-flared foot rotation.
