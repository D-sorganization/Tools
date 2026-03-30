import mujoco
import numpy as np


class LowerBodySimulator:
    """
    Simulates the lower body model using MuJoCo.
    """

    def __init__(self, xml_string: str) -> None:
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)

        # Identify indices for sensors and joints
        self.joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
        ]
        self.actuator_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]

        self._cache_indices()
        self._polynomial_drivers: dict[int, np.ndarray] = {}
        mujoco.mj_forward(self.model, self.data)

    def _cache_indices(self) -> None:
        """Pre-compute indices for fast lookup."""
        self.jnt_qpos_idx = {}
        for name in self.joint_names:
            if name is None:
                continue
            idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.jnt_qpos_idx[name] = self.model.jnt_qposadr[idx]

        self.pelvis_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def set_joint_polynomial(self, joint_name: str, coeffs: list[float]) -> None:
        """
        Drive a joint using a polynomial function evaluated over time.
        coeffs: highest degree first (e.g. polyval format) or lowest degree first.
        We will assume standard np.polyval format: highest degree first.
        """
        if joint_name not in self.actuator_names:
            # Maybe the actuator is named act_{joint_name}
            act_name = f"act_{joint_name}"
        else:
            act_name = joint_name

        act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
        if act_id == -1:
            raise ValueError(f"No actuator found for {joint_name}")

        self._polynomial_drivers[act_id] = np.array(coeffs)

    def compute_zero_torque_counterfactual(self) -> dict[str, np.ndarray]:
        """
        Compute Zero Torque Counterfactual (ZTCF).
        Calculates the accelerations or forces that would occur if no torques were applied,
        but maintaining the current kinematic state.
        In MuJoCo, this is essentially inverse dynamics with zero target acceleration (to find gravity/Coriolis),
        or forward dynamics with zero control to find resulting acceleration.
        We return both for analytical purposes.
        """
        # Save state
        qfrc_applied_orig = self.data.qfrc_applied.copy()
        ctrl_orig = self.data.ctrl.copy()

        # Set controls to zero
        self.data.ctrl[:] = 0.0
        self.data.qfrc_applied[:] = 0.0

        # Forward dynamics to find induced acceleration without torques
        mujoco.mj_forward(self.model, self.data)
        qacc_zero_torque = self.data.qacc.copy()

        # Inverse dynamics to find forces required to maintain ZERO acceleration
        self.data.qacc[:] = 0.0
        mujoco.mj_inverse(self.model, self.data)
        qfrc_inverse = self.data.qfrc_inverse.copy()

        # Restore state
        self.data.ctrl[:] = ctrl_orig
        self.data.qfrc_applied[:] = qfrc_applied_orig
        mujoco.mj_forward(self.model, self.data)

        return {"qacc_ztcf": qacc_zero_torque, "qfrc_to_hold_state": qfrc_inverse}

    def compute_pelvis_kinematics(self) -> dict[str, float]:
        """
        Returns the lateral, vertical, and forward/back motion of the hips,
        as well as all the tilts. (Pelvis body position and orientation/euler)
        """
        # The pelvis is a free joint, its Cartesian position is directly in data.xpos
        pos = self.data.xpos[self.pelvis_body_id]
        quat = self.data.xquat[self.pelvis_body_id]

        # Convert quat (w, x, y, z) to Euler angles (tilt, obliquity, rotation)
        # We can extract it via rotmat
        mat_1d = np.zeros(9)
        mujoco.mju_quat2Mat(mat_1d, quat)
        mat = mat_1d.reshape(3, 3)

        # Z-Y-X Euler angles
        sy = np.sqrt(mat[0, 0] * mat[0, 0] + mat[1, 0] * mat[1, 0])
        singular = sy < 1e-6
        if not singular:
            roll = np.arctan2(mat[2, 1], mat[2, 2])
            pitch = np.arctan2(-mat[2, 0], sy)
            yaw = np.arctan2(mat[1, 0], mat[0, 0])
        else:
            roll = np.arctan2(-mat[1, 2], mat[1, 1])
            pitch = np.arctan2(-mat[2, 0], sy)
            yaw = 0

        return {
            "x_forward": pos[0],
            "y_lateral": pos[1],
            "z_vertical": pos[2],
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
        }

    def analyze_induced_acceleration(
        self, actuator_name: str, torque_value: float = 1.0
    ) -> dict[str, float]:
        """
        Perform Induced Acceleration Analysis (IAA) at the current state.
        This isolates the instantaneous acceleration effect of a specific torque
        applied to a specific actuator on the root pelvis body.

        Returns the induced linear and angular acceleration of the pelvis.
        """
        act_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
        )
        if act_id == -1:
            raise ValueError(f"No actuator found for {actuator_name}")

        # 1. Baseline acceleration (zero torque)
        ctrl_orig = self.data.ctrl.copy()
        qfrc_applied_orig = self.data.qfrc_applied.copy()

        self.data.ctrl[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Free joint accelerations are the first 6 elements of qacc (3 linear, 3 angular)
        qacc_base = self.data.qacc[:6].copy()

        # 2. Perturbed acceleration
        self.data.ctrl[act_id] = torque_value
        mujoco.mj_forward(self.model, self.data)
        qacc_pert = self.data.qacc[:6].copy()

        # 3. Induced acceleration
        induced_acc = qacc_pert - qacc_base

        # Restore
        self.data.ctrl[:] = ctrl_orig
        self.data.qfrc_applied[:] = qfrc_applied_orig
        mujoco.mj_forward(self.model, self.data)

        return {
            "forward_accel": induced_acc[0],
            "lateral_accel": induced_acc[1],
            "vertical_accel": induced_acc[2],
            "roll_accel": induced_acc[3],
            "pitch_accel": induced_acc[4],
            "yaw_accel": induced_acc[5],
        }

    def inverse_kinematics(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        max_iters: int = 100,
        tol: float = 1e-4,
    ) -> bool:
        """
        Solve Inverse Kinematics to move the pelvis to the target position and orientation (quaternion),
        finding the 'easiest path' (minimum joint motion) using Damped Least Squares.
        Assumes the feet are anchored to the ground. In this model, pelvis is the free joint,
        so we actually want to find the joint configurations that result in the feet being exactly at (or near)
        their rest positions when the pelvis is at `target_pos` & `target_quat`.
        However, since the feet are not strictly equality-constrained in the builder (using contact instead),
        true IK means we set the root position to target, and then optimize leg angles so feet reach the ground origin.
        We'll just set the root directly to target, then optimize the legs to place feet at [0.05, +-hip_offset, 0].
        """
        # Save old state
        self.data.qpos.copy()

        # Free joint QPOS is [x, y, z, qw, qx, qy, qz]
        self.data.qpos[0:3] = target_pos
        self.data.qpos[3:7] = target_quat

        # We need to map left foot and right foot sites to ground
        r_foot_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "r_foot_center"
        )
        l_foot_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "l_foot_center"
        )

        # Nominal target positions for feet (where they start)
        r_target = np.array([0.05, -0.15, -0.03])
        l_target = np.array([0.05, 0.15, -0.03])

        jac_r = np.zeros((3, self.model.nv))
        jac_l = np.zeros((3, self.model.nv))

        alpha = 0.1  # Damping

        for _ in range(max_iters):
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)

            r_err = r_target - self.data.site_xpos[r_foot_id]
            l_err = l_target - self.data.site_xpos[l_foot_id]

            err = np.concatenate((r_err, l_err))
            if np.linalg.norm(err) < tol:
                return True

            mujoco.mj_jacSite(self.model, self.data, jac_r, None, r_foot_id)
            mujoco.mj_jacSite(self.model, self.data, jac_l, None, l_foot_id)

            J = np.vstack((jac_r, jac_l))

            # Note: We only want to modify joint angles, NOT the free root (which we locked to target)
            # Root DOFs are the first 6 in qvel (indices 0-5). We zero them out in J so they aren't moved.
            J[:, 0:6] = 0.0

            # Damped least squares: dq = J.T @ inv(J @ J.T + alpha*I) @ err
            J_T = J.T
            dq = J_T @ np.linalg.inv(J @ J_T + alpha * np.eye(6)) @ err

            mujoco.mj_integratePos(self.model, self.data.qpos, dq, 1.0)

        return False

    def step(self) -> None:
        """Advance the simulation by one timestep, applying polynomial controls."""
        t = self.data.time
        for act_id, coeffs in self._polynomial_drivers.items():
            # Evaluate polynomial: p[0]*x**(N-1) + ... + p[N-1]
            ctrl_val = np.polyval(coeffs, t)
            self.data.ctrl[act_id] = ctrl_val

        mujoco.mj_step(self.model, self.data)
