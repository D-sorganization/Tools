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
    """
    Builds a MuJoCo MJCF XML string containing a lower body model with:
    - Gimbal hips (3 axes)
    - Revolute knees (1 axis)
    - Universal ankles (2 axes)

    The feet are constrained via an equality weld to the ground floor,
    while the pelvis is the floating root that we track for lateral/vertical/forward motion.
    But for better kinematics and dynamics in MuJoCo, we typically make the pelvis the free root,
    and let the feet make contact with the floor.
    We will use a free joint strictly at the pelvis.
    """
    hip_offset = pelvis_width / 2.0

    # We will define the legs. Right is -y, Left is +y (assuming z up, x forward)
    # The ellipsoids can be approximated by capsule or ellipsoid in MuJoCo.

    xml = f"""
    <mujoco model="lower_body_model">
        <compiler angle="degree" coordinate="local"/>
        <option gravity="0 0 -9.81" timestep="0.005"/>

        <asset>
            <material name="matplane" reflectance="0.3" specular="1" shininess="1" rgba="0.8 0.9 0.8 1"/>
            <material name="matgeom" rgba="0.8 0.6 0.4 1"/>
            <material name="matfoot" rgba="0.2 0.2 0.2 1"/>
        </asset>

        <worldbody>
            <light directional="true" diffuse=".8 .8 .8" specular=".2 .2 .2" pos="0 0 5" dir="0 0 -1"/>
            <geom name="floor" type="plane" pos="0 0 0" size="10 10 0.1" material="matplane" condim="3"/>

            <body name="pelvis" pos="0 0 {thigh_length + calf_length + 0.1}">
                <freejoint name="root"/>
                <!-- Pelvis geometry -->
                <geom type="box" size="0.1 {hip_offset} 0.1" mass="{pelvis_mass}" material="matgeom"/>

                <!-- RIGHT LEG -->
                <body name="r_thigh" pos="0 -{hip_offset} -0.05">
                    <!-- Hip Gimbal Joint -->
                    <joint name="r_hip_x" type="hinge" axis="1 0 0" range="-90 90"/>
                    <joint name="r_hip_y" type="hinge" axis="0 1 0" range="-90 90"/>
                    <joint name="r_hip_z" type="hinge" axis="0 0 1" range="-90 90"/>

                    <geom type="ellipsoid" size="0.08 0.08 {thigh_length / 2}" pos="0 0 -{thigh_length / 2}" mass="{thigh_mass}" material="matgeom"/>

                    <body name="r_calf" pos="0 0 -{thigh_length}">
                        <!-- Knee Revolute Joint -->
                        <joint name="r_knee" type="hinge" axis="0 1 0" range="0 150"/>
                        <geom type="ellipsoid" size="0.06 0.06 {calf_length / 2}" pos="0 0 -{calf_length / 2}" mass="{calf_mass}" material="matgeom"/>

                        <body name="r_foot" pos="0 0 -{calf_length}">
                            <!-- Ankle Universal Joint -->
                            <joint name="r_ankle_x" type="hinge" axis="1 0 0" range="-30 30"/>
                            <joint name="r_ankle_y" type="hinge" axis="0 1 0" range="-30 30"/>

                            <!-- Foot box so it contacts the floor flatly -->
                            <geom type="box" size="0.12 0.06 0.03" pos="0.05 0 -0.03" mass="{foot_mass}" material="matfoot" condim="3" friction="1 0.005 0.0001"/>
                        </body>
                    </body>
                </body>

                <!-- LEFT LEG -->
                <body name="l_thigh" pos="0 {hip_offset} -0.05">
                    <!-- Hip Gimbal Joint -->
                    <joint name="l_hip_x" type="hinge" axis="1 0 0" range="-90 90"/>
                    <joint name="l_hip_y" type="hinge" axis="0 1 0" range="-90 90"/>
                    <joint name="l_hip_z" type="hinge" axis="0 0 1" range="-90 90"/>

                    <geom type="ellipsoid" size="0.08 0.08 {thigh_length / 2}" pos="0 0 -{thigh_length / 2}" mass="{thigh_mass}" material="matgeom"/>

                    <body name="l_calf" pos="0 0 -{thigh_length}">
                        <!-- Knee Revolute Joint -->
                        <joint name="l_knee" type="hinge" axis="0 1 0" range="0 150"/>
                        <geom type="ellipsoid" size="0.06 0.06 {calf_length / 2}" pos="0 0 -{calf_length / 2}" mass="{calf_mass}" material="matgeom"/>

                        <body name="l_foot" pos="0 0 -{calf_length}">
                            <!-- Ankle Universal Joint -->
                            <joint name="l_ankle_x" type="hinge" axis="1 0 0" range="-30 30"/>
                            <joint name="l_ankle_y" type="hinge" axis="0 1 0" range="-30 30"/>

                            <!-- Foot box so it contacts the floor flatly -->
                            <geom type="box" size="0.12 0.06 0.03" pos="0.05 0 -0.03" mass="{foot_mass}" material="matfoot" condim="3" friction="1"/>
                        </body>
                    </body>
                </body>
            </body>
        </worldbody>

        <actuator>
            <motor joint="r_hip_x" name="act_r_hip_x" gear="1" ctrllimited="true" ctrlrange="-500 500"/>
            <motor joint="r_hip_y" name="act_r_hip_y" gear="1" ctrllimited="true" ctrlrange="-500 500"/>
            <motor joint="r_hip_z" name="act_r_hip_z" gear="1" ctrllimited="true" ctrlrange="-500 500"/>
            <motor joint="r_knee" name="act_r_knee" gear="1" ctrllimited="true" ctrlrange="-500 500"/>
            <motor joint="r_ankle_x" name="act_r_ankle_x" gear="1" ctrllimited="true" ctrlrange="-500 500"/>
            <motor joint="r_ankle_y" name="act_r_ankle_y" gear="1" ctrllimited="true" ctrlrange="-500 500"/>

            <motor joint="l_hip_x" name="act_l_hip_x" gear="1" ctrllimited="true" ctrlrange="-500 500"/>
            <motor joint="l_hip_y" name="act_l_hip_y" gear="1" ctrllimited="true" ctrlrange="-500 500"/>
            <motor joint="l_hip_z" name="act_l_hip_z" gear="1" ctrllimited="true" ctrlrange="-500 500"/>
            <motor joint="l_knee" name="act_l_knee" gear="1" ctrllimited="true" ctrlrange="-500 500"/>
            <motor joint="l_ankle_x" name="act_l_ankle_x" gear="1" ctrllimited="true" ctrlrange="-500 500"/>
            <motor joint="l_ankle_y" name="act_l_ankle_y" gear="1" ctrllimited="true" ctrlrange="-500 500"/>
        </actuator>

        <sensor>
            <force name="r_foot_force" site="r_foot_center" />
            <force name="l_foot_force" site="l_foot_center" />
        </sensor>
    </mujoco>
    """

    # We need to add the sites to sensors to measure GRF / constraint forces
    # I'll update the xml string using string replacements or just inject sites.
    # Let me just inline them properly!

    xml = xml.replace("<body>", "")

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

            <body name="pelvis" pos="0 0 {thigh_length + calf_length + 0.1}">
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
                            <geom type="ellipsoid" size="0.13 0.05 0.04" pos="0.06 0 -0.04" mass="{foot_mass}" material="matfoot" condim="3"/>
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
                            <geom type="ellipsoid" size="0.13 0.05 0.04" pos="0.06 0 -0.04" mass="{foot_mass}" material="matfoot" condim="3"/>
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
