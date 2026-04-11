"""MuJoCo MJCF XML generator for the lower body golf-swing model."""

from __future__ import annotations

import textwrap


def build_lower_body_xml(
    thigh_mass: float = 10.0,
    calf_mass: float = 4.5,
    foot_mass: float = 1.0,
    pelvis_mass: float = 20.0,
    thigh_length: float = 0.45,
    calf_length: float = 0.45,
    pelvis_width: float = 0.3,
) -> str:
    """Build an MJCF XML string describing the lower body model.

    The model has a 6-DOF free-joint pelvis, gimbal (3 axes) hips, revolute
    knees and universal ankles, plus contact-based foot-ground interaction.

    Args:
        thigh_mass: Mass of each thigh ellipsoid (kg).
        calf_mass: Mass of each calf ellipsoid (kg).
        foot_mass: Mass of each foot ellipsoid (kg).
        pelvis_mass: Mass of the pelvis body (kg).
        thigh_length: Length of each thigh (m).
        calf_length: Length of each calf (m).
        pelvis_width: Side-to-side width of the pelvis (m); hips are placed
            at ``pelvis_width / 2`` on either side of the body centreline.

    Returns:
        A dedented MJCF XML string ready for ``mujoco.MjModel.from_xml_string``.

    Raises:
        TypeError: If any argument is not a real number.
        ValueError: If any argument is not strictly positive.
    """
    _validate_positive_float("thigh_mass", thigh_mass)
    _validate_positive_float("calf_mass", calf_mass)
    _validate_positive_float("foot_mass", foot_mass)
    _validate_positive_float("pelvis_mass", pelvis_mass)
    _validate_positive_float("thigh_length", thigh_length)
    _validate_positive_float("calf_length", calf_length)
    _validate_positive_float("pelvis_width", pelvis_width)

    hip_offset = pelvis_width / 2.0
    pelvis_z = thigh_length + calf_length + 0.1

    xml = f"""
    <mujoco model="lower_body_model">
        <compiler angle="degree" coordinate="local"/>
        <option gravity="0 0 -9.81" timestep="0.005"/>
        <size njmax="500" nconmax="100"/>
        <asset>
            <material name="matplane" reflectance="0.3" specular="1" shininess="1" rgba="0.8 0.9 0.8 1"/>
            <material name="matgeom" rgba="0.8 0.6 0.4 1"/>
            <material name="matfoot" rgba="0.2 0.2 0.2 1"/>
        </asset>

        <worldbody>
            <light directional="true" diffuse=".8 .8 .8" specular=".2 .2 .2" pos="0 0 5" dir="0 0 -1"/>
            <geom name="floor" type="plane" pos="0 0 0" size="10 10 0.1" material="matplane" condim="3"/>

            <body name="pelvis" pos="0 0 {pelvis_z}">
                <freejoint name="root"/>
                <geom type="ellipsoid" size="0.15 {hip_offset + 0.05} 0.12" mass="{pelvis_mass}" material="matgeom"/>

                <!-- RIGHT LEG -->
                <body name="r_thigh" pos="0 -{hip_offset} -0.05">
                    <joint name="r_hip_x" type="hinge" axis="1 0 0" range="-90 90"/>
                    <joint name="r_hip_y" type="hinge" axis="0 1 0" range="-90 90"/>
                    <joint name="r_hip_z" type="hinge" axis="0 0 1" range="-90 90"/>
                    <geom type="ellipsoid" size="0.08 0.08 {thigh_length / 2}" pos="0 0 -{thigh_length / 2}" mass="{thigh_mass}" material="matgeom"/>
                    <body name="r_calf" pos="0 0 -{thigh_length}">
                        <joint name="r_knee" type="hinge" axis="0 1 0" range="0 150"/>
                        <geom type="ellipsoid" size="0.06 0.06 {calf_length / 2}" pos="0 0 -{calf_length / 2}" mass="{calf_mass}" material="matgeom"/>
                        <body name="r_foot" pos="0 0 -{calf_length}">
                            <joint name="r_ankle_x" type="hinge" axis="1 0 0" range="-30 30"/>
                            <joint name="r_ankle_y" type="hinge" axis="0 1 0" range="-30 30"/>
                            <geom name="r_foot_geom" type="ellipsoid" size="0.13 0.05 0.04" pos="0.06 0 -0.04" mass="{foot_mass}" material="matfoot" condim="3"/>
                            <site name="r_foot_center" type="sphere" size="0.01" pos="0.06 0 -0.04" rgba="1 0 0 1"/>
                        </body>
                    </body>
                </body>

                <!-- LEFT LEG -->
                <body name="l_thigh" pos="0 {hip_offset} -0.05">
                    <joint name="l_hip_x" type="hinge" axis="1 0 0" range="-90 90"/>
                    <joint name="l_hip_y" type="hinge" axis="0 1 0" range="-90 90"/>
                    <joint name="l_hip_z" type="hinge" axis="0 0 1" range="-90 90"/>
                    <geom type="ellipsoid" size="0.08 0.08 {thigh_length / 2}" pos="0 0 -{thigh_length / 2}" mass="{thigh_mass}" material="matgeom"/>
                    <body name="l_calf" pos="0 0 -{thigh_length}">
                        <joint name="l_knee" type="hinge" axis="0 1 0" range="0 150"/>
                        <geom type="ellipsoid" size="0.06 0.06 {calf_length / 2}" pos="0 0 -{calf_length / 2}" mass="{calf_mass}" material="matgeom"/>
                        <body name="l_foot" pos="0 0 -{calf_length}">
                            <joint name="l_ankle_x" type="hinge" axis="1 0 0" range="-30 30"/>
                            <joint name="l_ankle_y" type="hinge" axis="0 1 0" range="-30 30"/>
                            <geom name="l_foot_geom" type="ellipsoid" size="0.13 0.05 0.04" pos="0.06 0 -0.04" mass="{foot_mass}" material="matfoot" condim="3"/>
                            <site name="l_foot_center" type="sphere" size="0.01" pos="0.06 0 -0.04" rgba="1 0 0 1"/>
                        </body>
                    </body>
                </body>
            </body>
        </worldbody>

        <actuator>
            <motor name="act_r_hip_x" joint="r_hip_x" gear="1" ctrllimited="true" ctrlrange="-500 500"/>
            <motor name="act_r_hip_y" joint="r_hip_y" gear="1" ctrllimited="true" ctrlrange="-500 500"/>
            <motor name="act_r_hip_z" joint="r_hip_z" gear="1" ctrllimited="true" ctrlrange="-500 500"/>
            <motor name="act_r_knee" joint="r_knee" gear="1" ctrllimited="true" ctrlrange="-500 500"/>
            <motor name="act_r_ankle_x" joint="r_ankle_x" gear="1" ctrllimited="true" ctrlrange="-500 500"/>
            <motor name="act_r_ankle_y" joint="r_ankle_y" gear="1" ctrllimited="true" ctrlrange="-500 500"/>

            <motor name="act_l_hip_x" joint="l_hip_x" gear="1" ctrllimited="true" ctrlrange="-500 500"/>
            <motor name="act_l_hip_y" joint="l_hip_y" gear="1" ctrllimited="true" ctrlrange="-500 500"/>
            <motor name="act_l_hip_z" joint="l_hip_z" gear="1" ctrllimited="true" ctrlrange="-500 500"/>
            <motor name="act_l_knee" joint="l_knee" gear="1" ctrllimited="true" ctrlrange="-500 500"/>
            <motor name="act_l_ankle_x" joint="l_ankle_x" gear="1" ctrllimited="true" ctrlrange="-500 500"/>
            <motor name="act_l_ankle_y" joint="l_ankle_y" gear="1" ctrllimited="true" ctrlrange="-500 500"/>
        </actuator>

        <sensor>
            <touch name="r_foot_touch" site="r_foot_center"/>
            <touch name="l_foot_touch" site="l_foot_center"/>
        </sensor>
    </mujoco>
    """

    return textwrap.dedent(xml).strip()


def _validate_positive_float(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    if value <= 0.0:
        raise ValueError(f"{name} must be strictly positive, got {value}")
