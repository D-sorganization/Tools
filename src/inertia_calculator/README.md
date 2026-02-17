# Inertia Calculator

A PyQt6-based GUI application for calculating and validating inertia tensors for robotics and simulation applications.

## Purpose

The Inertia Calculator computes mass moments of inertia for primitive geometric shapes commonly used in robotic link modeling. It provides validated inertia tensors in formats compatible with URDF (Unified Robot Description Format) and other simulation frameworks.

## Key Features

- **Primitive Shape Calculations**: Computes inertia for solid boxes, cylinders, spheres, and hollow cylinders
- **Manual Input Mode**: Validate user-specified inertia tensors against physical constraints
- **URDF Export Format**: Generates ready-to-use `<inertia>` XML elements
- **Real-time Validation**: Checks positive definiteness and triangle inequality constraints
- **Matrix Visualization**: Displays the full 3x3 symmetric inertia tensor

## Installation / Prerequisites

### Dependencies

```bash
pip install PyQt6 numpy
```

### Running the Application

```bash
python -m inertia_calculator.ui.pyqt6.main_window
# or
python launch_pyqt6.py
```

## Usage Instructions

### Primitive Shapes Tab

1. Select a shape type from the dropdown menu
2. Enter the mass in kilograms
3. Enter the shape dimensions in meters
4. Click "Calculate Inertia" to compute the tensor

### Manual Input Tab

1. Enter the six independent inertia tensor components (Ixx, Iyy, Izz, Ixy, Ixz, Iyz)
2. Enter the associated mass
3. Click "Validate Inertia" to check physical validity

## Input Parameters

### Mass

| Parameter | Unit | Range          | Description                  |
| --------- | ---- | -------------- | ---------------------------- |
| Mass      | kg   | 0.001 - 10,000 | Total mass of the rigid body |

### Solid Box Dimensions

| Parameter | Unit | Range       | Description         |
| --------- | ---- | ----------- | ------------------- |
| Length X  | m    | 0.001 - 100 | Extent along X axis |
| Length Y  | m    | 0.001 - 100 | Extent along Y axis |
| Length Z  | m    | 0.001 - 100 | Extent along Z axis |

### Solid Cylinder Dimensions

| Parameter | Unit | Range       | Description                    |
| --------- | ---- | ----------- | ------------------------------ |
| Radius    | m    | 0.001 - 100 | Cylinder radius                |
| Height    | m    | 0.001 - 100 | Cylinder height (axis along Z) |

### Solid Sphere Dimensions

| Parameter | Unit | Range       | Description   |
| --------- | ---- | ----------- | ------------- |
| Radius    | m    | 0.001 - 100 | Sphere radius |

### Hollow Cylinder Dimensions

| Parameter    | Unit | Range       | Description                           |
| ------------ | ---- | ----------- | ------------------------------------- |
| Outer Radius | m    | 0.001 - 100 | Outer cylinder radius                 |
| Inner Radius | m    | 0.001 - 100 | Inner cavity radius (must be < outer) |
| Height       | m    | 0.001 - 100 | Cylinder height (axis along Z)        |

## Output Format

The calculator outputs:

1. **Principal Moments**: Ixx, Iyy, Izz in kg\*m^2
2. **Products of Inertia**: Ixy, Ixz, Iyz in kg\*m^2 (zero for primitive shapes)
3. **Full Inertia Tensor Matrix**: 3x3 symmetric matrix
4. **URDF Format**: Ready-to-paste XML element

```xml
<inertia ixx="0.001667" ixy="0.000000" ixz="0.000000"
         iyy="0.001667" iyz="0.000000" izz="0.001667"/>
```

## Mathematical Models

### Solid Box

For a rectangular box with dimensions (lx, ly, lz) and mass m:

```
Ixx = (1/12) * m * (ly^2 + lz^2)
Iyy = (1/12) * m * (lx^2 + lz^2)
Izz = (1/12) * m * (lx^2 + ly^2)
```

### Solid Cylinder (axis along Z)

For a cylinder with radius r, height h, and mass m:

```
Ixx = Iyy = (1/12) * m * (3*r^2 + h^2)
Izz = (1/2) * m * r^2
```

### Solid Sphere

For a sphere with radius r and mass m:

```
Ixx = Iyy = Izz = (2/5) * m * r^2
```

### Hollow Cylinder (axis along Z)

For a hollow cylinder with outer radius r_out, inner radius r_in, height h, and mass m:

```
Ixx = Iyy = (1/12) * m * (3*(r_out^2 + r_in^2) + h^2)
Izz = (1/2) * m * (r_out^2 + r_in^2)
```

### Parallel Axis Theorem

To translate inertia about the center of mass to a parallel axis at distance d:

```
I_new = I_cm + m * d^2
```

## Example Usage

### Example 1: Robot Link (Box)

A robotic forearm link modeled as a box:

- Dimensions: 0.05m x 0.04m x 0.20m
- Mass: 0.5 kg

Results:

```
Ixx = 0.001750 kg*m^2
Iyy = 0.001708 kg*m^2
Izz = 0.000171 kg*m^2
```

### Example 2: Cylindrical Motor

A DC motor housing:

- Radius: 0.025m
- Height: 0.06m
- Mass: 0.3 kg

Results:

```
Ixx = Iyy = 0.000141 kg*m^2
Izz = 0.000094 kg*m^2
```

## Troubleshooting

### "Triangle inequality violated"

The principal moments of inertia must satisfy the triangle inequality:

- |Ixx - Iyy| <= Izz <= Ixx + Iyy (and cyclic permutations)

This occurs when manually entered values don't correspond to a physically realizable mass distribution.

### "Inertia tensor is not positive definite"

The inertia tensor must be positive definite, meaning all eigenvalues must be positive. This validation uses Cholesky decomposition. Check that:

- All diagonal elements (Ixx, Iyy, Izz) are positive
- Products of inertia are not too large relative to diagonal elements

### "Inner radius must be less than outer radius"

For hollow cylinders, ensure the inner cavity radius is strictly smaller than the outer radius.

## Related Tools

- **URDF Builder GUI**: Use calculated inertias when building robot models
- **Humanoid Builder GUI**: Automatic inertia calculation for body segments
- **MuJoCo XML Generator**: Alternative simulation format support

## References

- Goldstein, H. (1980). Classical Mechanics. Addison-Wesley.
- Craig, J.J. (2005). Introduction to Robotics. Pearson.
- URDF Specification: http://wiki.ros.org/urdf/XML/link
