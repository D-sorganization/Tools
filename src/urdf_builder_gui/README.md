# Parametric URDF Builder

A PyQt6-based GUI application for generating parametric URDF (Unified Robot Description Format) models for robotics applications.

## Purpose

The Parametric URDF Builder allows users to create robot models by specifying high-level parameters rather than manually writing XML. It automatically calculates link dimensions, inertial properties, and joint configurations based on anthropometric scaling factors.

## Key Features

- **Parametric Model Generation**: Define robots using height, mass, and proportion factors
- **Template-Based Design**: Choose from full humanoid, upper body, lower body, or custom templates
- **Gender-Based Scaling**: Adjust body proportions using anthropometric gender factors
- **Geometry Options**: Support for capsule, cylinder, box, and sphere collision primitives
- **Joint Configuration**: Configurable damping, friction, and limit parameters
- **Live Preview**: View model structure before generating final URDF
- **Direct Export**: Save URDF files with proper XML formatting

## Installation / Prerequisites

### Dependencies

```bash
pip install PyQt6
```

### Optional (for advanced model generation)

```bash
pip install model_generation  # Internal package for parametric humanoid building
```

### Running the Application

```bash
python -m urdf_builder_gui.ui.pyqt6.main_window
# or
python launch_pyqt6.py
```

## Usage Instructions

### Body Parameters Tab

1. Enter a robot name (used as the URDF robot element name)
2. Set the total height in meters
3. Set the total mass in kilograms
4. Adjust the gender factor slider (affects shoulder/hip width ratios)
5. Select a model template

### Proportions Tab

1. Adjust individual body segment proportions using sliders
2. Values range from 50% to 150% of default proportions
3. Click "Reset to Defaults" to restore 100% scaling

### Options Tab

1. **Geometry Options**: Select default visual/collision geometry type
2. **Joint Options**: Set default damping and friction coefficients
3. **Inertia Calculation**: Choose primitive, mesh-based, or scaled inertia mode

### Generating Output

1. Click "Preview Structure" to see a summary of the model
2. Click "Generate URDF" to create the XML
3. Click "Export URDF File" to save to disk

## Input Parameters

### Basic Parameters

| Parameter | Unit | Range | Description |
|-----------|------|-------|-------------|
| Robot Name | - | text | Identifier for the robot model |
| Height | m | 0.5 - 3.0 | Total standing height |
| Mass | kg | 20 - 200 | Total body mass |
| Gender Factor | % | 0 - 100 | Female (0) to Male (100) scaling |

### Body Proportions

| Parameter | Range | Description |
|-----------|-------|-------------|
| Shoulder Width | 50% - 150% | Biacromial breadth scaling |
| Hip Width | 50% - 150% | Bi-iliac breadth scaling |
| Arm Length | 50% - 150% | Upper + lower arm scaling |
| Leg Length | 50% - 150% | Thigh + shin scaling |
| Torso Length | 50% - 150% | Lumbar + thorax scaling |
| Head Size | 50% - 150% | Head diameter scaling |

### Joint Configuration

| Parameter | Unit | Range | Description |
|-----------|------|-------|-------------|
| Default Damping | N*m*s/rad | 0 - 100 | Viscous damping coefficient |
| Default Friction | N*m | 0 - 100 | Coulomb friction coefficient |
| Density | kg/m^3 | 500 - 2000 | Default material density |

## Output Format

### URDF Structure

The generated URDF follows the standard format:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<robot name="humanoid">
  <link name="pelvis">
    <visual>
      <geometry><box size="0.2 0.3 0.15"/></geometry>
      <material name="skin"><color rgba="0.8 0.6 0.5 1.0"/></material>
    </visual>
    <collision>
      <geometry><box size="0.2 0.3 0.15"/></geometry>
    </collision>
    <inertial>
      <mass value="7.8400"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="pelvis_to_torso" type="fixed">
    <parent link="pelvis"/>
    <child link="torso"/>
    <origin xyz="0 0 0.25"/>
  </joint>
  <!-- Additional links and joints -->
</robot>
```

### Model Templates

| Template | Links Included |
|----------|----------------|
| Full Humanoid | Pelvis, torso, head, both arms, both legs |
| Upper Body Only | Pelvis, torso, head, both arms |
| Lower Body Only | Pelvis, both legs |
| Torso + Arms | Pelvis, torso, both arms |
| Torso + Legs | Pelvis, torso, both legs |

## Mathematical Models

### Segment Length Estimation

Segment lengths are calculated from total height using anthropometric ratios:

```
Pelvis Height  = 0.078 * H
Torso Height   = 0.278 * H
Head Diameter  = 0.139 * H
Thigh Length   = 0.245 * H
Shin Length    = 0.246 * H
Upper Arm      = 0.186 * H
Forearm        = 0.146 * H
```

### Mass Distribution

Segment masses are distributed according to de Leva (1996) ratios:

```
Pelvis Mass    = 0.112 * M
Torso Mass     = 0.350 * M
Head Mass      = 0.069 * M
Thigh Mass     = 0.142 * M (each)
Shin Mass      = 0.043 * M (each)
```

### Inertia Calculation

For primitive geometry mode, inertia is computed assuming uniform density:

```
I_box = (1/12) * m * (h^2 + d^2)  [for each principal axis]
I_cylinder = (1/12) * m * (3r^2 + h^2)  [transverse]
           = (1/2) * m * r^2  [axial]
```

## Example Usage

### Example 1: Standard Adult Male

Parameters:
- Height: 1.75 m
- Mass: 75 kg
- Gender Factor: 80%
- Template: Full Humanoid

Generated structure includes 17 links (pelvis, torso, head, 2x arms, 2x legs with segments) with properly scaled dimensions.

### Example 2: Child Robot Model

Parameters:
- Height: 1.20 m
- Mass: 25 kg
- Gender Factor: 50%
- Head Size: 120% (larger relative head for children)

### Example 3: Heavy-Duty Industrial Arm

Parameters:
- Template: Torso + Arms
- Mass: 150 kg
- Damping: 10.0
- Friction: 5.0

## Troubleshooting

### "Build failed: Unknown error"

Ensure all required parameters are set and within valid ranges. Check that height > 0.5m and mass > 20kg.

### URDF validation fails in ROS

- Verify all joints have valid parent/child links
- Check that link names don't contain special characters
- Ensure inertia values are positive definite

### Model appears distorted in visualization

- Reset proportions to 100% defaults
- Check that extreme proportion values aren't causing geometric issues
- Verify height and mass are reasonable for the template

### Export fails to save file

- Check write permissions in the target directory
- Ensure the filename doesn't contain invalid characters
- Verify sufficient disk space

## Related Tools

- **Inertia Calculator**: Compute precise inertias for custom shapes
- **Humanoid Builder GUI**: Advanced humanoid with anthropometric data
- **C3D Viewer**: Import motion capture data to animate models
- **MuJoCo Converter**: Convert URDF to MuJoCo XML format

## References

- URDF Specification: http://wiki.ros.org/urdf/XML
- de Leva, P. (1996). Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters.
- Winter, D.A. (2009). Biomechanics and Motor Control of Human Movement.
