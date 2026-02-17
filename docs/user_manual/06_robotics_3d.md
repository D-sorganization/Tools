# Chapter 6 — Robotics and 3D Tools

**Parent Document:** [Tools User Manual](./TOOLS_USER_MANUAL.md)

---

## 6.1 C3D Viewer

**Source:** `src/c3d_viewer/`
**GUI:** PyQt6
**Status:** ✅ Implemented

### 6.1.1 Purpose

Visualizer for C3D (Coordinate 3D) motion capture data files, commonly used in biomechanics and animation.

### 6.1.2 Capabilities

- Load and parse C3D binary format files
- 3D point cloud visualization
- Timeline scrubbing for animation playback
- Marker trajectory visualization
- Export capabilities

### 6.1.3 C3D File Format

The C3D format stores:

- 3D marker positions $(x, y, z)$ per frame
- Analog data channels
- Parameter sections (labels, descriptions, units)
- Frame rate and scaling information

---

## 6.2 Humanoid Builder GUI

**Source:** `src/humanoid_builder_gui/`
**Shared Library:** `src/shared/python/humanoid_character_builder/`
**GUI:** PyQt6
**Status:** ✅ Implemented

### 6.2.1 Purpose

Interactive tool for building humanoid character models with configurable body proportions, joint definitions, and mesh generation.

### 6.2.2 Capabilities

- Parametric humanoid model creation
- Configurable body segment proportions
- Joint definition and constraints
- Mesh generation from segment parameters
- URDF export for robotics simulation
- 3D visualization

### 6.2.3 Mathematical Model

**Forward Kinematics:**

$$T_n^0 = T_1^0 \cdot T_2^1 \cdot T_3^2 \cdots T_n^{n-1}$$

where each $T_i^{i-1}$ is a 4×4 homogeneous transformation matrix:

$$T = \begin{bmatrix} R & \vec{d} \\ 0 & 1 \end{bmatrix}$$

**Denavit-Hartenberg Parameters:**

$$T_i = \text{Rot}_z(\theta_i) \cdot \text{Trans}_z(d_i) \cdot \text{Trans}_x(a_i) \cdot \text{Rot}_x(\alpha_i)$$

### 6.2.4 Mesh Generation

The mesh generator creates primitive shapes (cylinders, spheres, boxes) for body segments using the `trimesh` library:

| Segment | Shape    | Parameters             |
| ------- | -------- | ---------------------- |
| Head    | Sphere   | radius                 |
| Torso   | Box      | width × height × depth |
| Limbs   | Cylinder | radius × length        |
| Joints  | Sphere   | radius                 |

---

## 6.3 URDF Builder GUI

**Source:** `src/urdf_builder_gui/`
**GUI:** PyQt6
**Status:** ✅ Implemented

### 6.3.1 Purpose

Visual editor for creating and editing URDF (Unified Robot Description Format) files used in ROS (Robot Operating System) and robotics simulators.

### 6.3.2 Capabilities

- Visual link and joint creation
- Joint type selection (revolute, prismatic, fixed, continuous)
- Inertia tensor configuration
- Collision and visual geometry definition
- URDF XML export/import
- 3D preview

### 6.3.3 URDF Structure

```xml
<robot name="robot_name">
  <link name="base_link">
    <visual>
      <geometry><box size="0.1 0.1 0.1"/></geometry>
    </visual>
    <collision>
      <geometry><box size="0.1 0.1 0.1"/></geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="1.0"/>
  </joint>
</robot>
```

### 6.3.4 Inertia Tensor

For a rigid body, the inertia tensor is:

$$I = \begin{bmatrix} I_{xx} & -I_{xy} & -I_{xz} \\ -I_{xy} & I_{yy} & -I_{yz} \\ -I_{xz} & -I_{yz} & I_{zz} \end{bmatrix}$$

**Standard shapes:**

| Shape             | $I_{xx}$                 | $I_{yy}$                 | $I_{zz}$                |
| ----------------- | ------------------------ | ------------------------ | ----------------------- |
| Box $(w, h, d)$   | $\frac{m}{12}(h^2+d^2)$  | $\frac{m}{12}(w^2+d^2)$  | $\frac{m}{12}(w^2+h^2)$ |
| Cylinder $(r, h)$ | $\frac{m}{12}(3r^2+h^2)$ | $\frac{m}{12}(3r^2+h^2)$ | $\frac{m}{2}r^2$        |
| Sphere $(r)$      | $\frac{2m}{5}r^2$        | $\frac{2m}{5}r^2$        | $\frac{2m}{5}r^2$       |

---

_[← Scientific Modeling](./05_scientific_modeling.md) | [Back to Manual](./TOOLS_USER_MANUAL.md) | [Next: Data & Document Processing →](./07_data_document_processing.md)_
