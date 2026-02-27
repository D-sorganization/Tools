# Humanoid Character Builder

A PyQt6-based GUI application for building parametric humanoid characters using anthropometric data, with URDF export for robotics and biomechanics applications.

## Purpose

The Humanoid Character Builder generates anatomically-proportioned humanoid models based on the de Leva (1996) anthropometric dataset. It calculates segment masses, lengths, and inertias from total body height and mass, enabling rapid creation of biomechanically accurate models for simulation, motion analysis, and robotics research.

## Key Features

- **Anthropometric Database**: Uses de Leva (1996) body segment parameters
- **Build Type Selection**: Ectomorph, mesomorph, endomorph, or average body types
- **Gender-Specific Models**: Male, female, or neutral anthropometric scaling
- **BMI Calculation**: Real-time body mass index with category classification
- **Segment Detail View**: Displays computed mass, length, width, and depth for all 17 segments
- **Proportion Customization**: Fine-tune shoulder width, limb lengths, and extremity sizes
- **Multi-Format Export**: URDF package, standalone URDF, or JSON configuration

## Installation / Prerequisites

### Dependencies

```bash
pip install PyQt6
```

### Running the Application

```bash
python -m humanoid_builder_gui.ui.pyqt6.main_window
# or
python launch_pyqt6.py
```

## Usage Instructions

### Body Parameters Tab

1. **Set Height and Mass**: Enter subject's height (m) and body mass (kg)
2. **Select Build Type**: Choose body composition (affects segment proportions)
3. **Select Gender Model**: Affects mass distribution and segment geometry
4. **Adjust Build Factors**: Fine-tune muscularity and body fat percentage
5. **Monitor BMI**: Real-time BMI calculation with health category

### Proportions Tab

1. Adjust sliders for individual body segment proportions
2. Values range from 0.50 to 1.50 (50% to 150% of default)
3. Click "Reset to Defaults" to restore 1.00 scaling

### Results Tab

After clicking "Build Character":

- View summary statistics (total height, mass, segment count)
- Examine detailed segment table with all computed values
- Verify computed total mass matches input mass

### Export Tab

1. Select export format (URDF Package, URDF Only, JSON Config)
2. Choose mesh format if applicable (STL, OBJ, DAE)
3. Set character name
4. Preview configuration and export

## Input Parameters

### Primary Parameters

| Parameter    | Unit | Range                  | Description                |
| ------------ | ---- | ---------------------- | -------------------------- |
| Height       | m    | 0.5 - 3.0              | Standing height            |
| Mass         | kg   | 10 - 300               | Total body mass            |
| Build Type   | -    | Ecto/Meso/Endo/Average | Body composition category  |
| Gender Model | -    | Male/Female/Neutral    | Anthropometric data source |

### Build Factors

| Parameter   | Range     | Description               |
| ----------- | --------- | ------------------------- |
| Muscularity | 0.0 - 1.0 | Muscle mass proportion    |
| Body Fat    | 0.0 - 1.0 | Adipose tissue proportion |

### Body Proportions

| Parameter      | Range       | Description               |
| -------------- | ----------- | ------------------------- |
| Shoulder Width | 0.50 - 1.50 | Biacromial breadth factor |
| Hip Width      | 0.50 - 1.50 | Bi-iliac breadth factor   |
| Arm Length     | 0.50 - 1.50 | Upper + lower arm factor  |
| Leg Length     | 0.50 - 1.50 | Thigh + shin factor       |
| Torso Length   | 0.50 - 1.50 | Trunk length factor       |
| Head Scale     | 0.50 - 1.50 | Head size factor          |
| Neck Length    | 0.50 - 1.50 | Cervical length factor    |
| Hand Scale     | 0.50 - 1.50 | Hand size factor          |
| Foot Scale     | 0.50 - 1.50 | Foot size factor          |

## Output Format

### Segment Data Table

| Column  | Unit | Description               |
| ------- | ---- | ------------------------- |
| Segment | -    | Anatomical name           |
| Mass    | kg   | Computed segment mass     |
| Length  | m    | Proximal-distal length    |
| Width   | m    | Medial-lateral extent     |
| Depth   | m    | Anterior-posterior extent |

### Export Formats

**URDF Package**:

```
character_name/
  humanoid.urdf
  meshes/
    pelvis.stl
    torso.stl
    ...
  config/
    body_params.yaml
```

**JSON Configuration**:

```json
{
  "name": "humanoid",
  "height_m": 1.75,
  "mass_kg": 75.0,
  "segments": [
    {"name": "Head", "mass_kg": 5.205, "length_m": 0.244, ...}
  ]
}
```

## Mathematical Models

### de Leva Segment Mass Ratios

Mass is distributed according to validated anthropometric data:

| Segment   | Mass Ratio (% body mass) |
| --------- | ------------------------ |
| Head      | 6.94%                    |
| Neck      | 2.40%                    |
| Thorax    | 21.60%                   |
| Lumbar    | 13.90%                   |
| Pelvis    | 11.17%                   |
| Upper Arm | 2.71% (each)             |
| Forearm   | 1.62% (each)             |
| Hand      | 0.61% (each)             |
| Thigh     | 14.16% (each)            |
| Shin      | 4.33% (each)             |
| Foot      | 1.37% (each)             |

### Segment Length Ratios

Lengths are proportional to total body height:

| Segment   | Length Ratio (% height) |
| --------- | ----------------------- |
| Head      | 13.95%                  |
| Neck      | 5.2%                    |
| Thorax    | 17.0%                   |
| Lumbar    | 10.8%                   |
| Pelvis    | 7.8%                    |
| Upper Arm | 18.6%                   |
| Forearm   | 14.6%                   |
| Hand      | 10.8%                   |
| Thigh     | 24.5%                   |
| Shin      | 24.6%                   |
| Foot      | 15.2%                   |

### Body Mass Index (BMI)

```
BMI = mass / height^2
```

| BMI Range   | Category    |
| ----------- | ----------- |
| < 18.5      | Underweight |
| 18.5 - 24.9 | Normal      |
| 25.0 - 29.9 | Overweight  |
| >= 30.0     | Obese       |

### Segment Geometry Estimation

Width and depth are estimated from length:

```
width = 0.30 * length
depth = 0.25 * length
```

## Example Usage

### Example 1: Average Adult Male

Input:

- Height: 1.78 m
- Mass: 80 kg
- Build Type: Average
- Gender: Male

Key Outputs:

- Head: 5.55 kg, 0.248 m length
- Thigh (each): 11.33 kg, 0.436 m length
- Total segments: 17

### Example 2: Female Athlete

Input:

- Height: 1.65 m
- Mass: 58 kg
- Build Type: Mesomorph
- Gender: Female
- Muscularity: 0.7

Key Outputs:

- Shoulder Width: standard
- Hip Width: scaled for female proportions
- BMI: 21.3 (Normal)

### Example 3: Custom Character for Animation

Input:

- Height: 1.80 m
- Mass: 70 kg
- Head Scale: 1.20 (stylized large head)
- Arm Length: 1.10 (slightly longer arms)
- Export: URDF Package with OBJ meshes

## Troubleshooting

### Computed mass doesn't match input mass

The de Leva ratios should sum to approximately 100%. Small discrepancies may occur due to:

- Rounding in proportion adjustments
- Bilateral segments counted once vs. twice

Verify that arm/leg ratios account for both left and right limbs.

### BMI shows unexpected category

BMI is calculated from raw height and mass inputs. It does not account for:

- Muscle vs. fat composition
- Frame size variations
- Athletic body types with high muscle mass

Use BMI as a general reference, not a definitive health indicator.

### Export fails with missing module error

Some export features require optional dependencies:

```bash
pip install pyyaml  # For YAML configuration export
pip install numpy-stl  # For STL mesh generation
```

### Segments appear too wide or narrow

The default width/depth estimation (30%/25% of length) may not suit all segments. For more accurate models:

- Manually adjust individual segment geometries after export
- Use the URDF Builder GUI for fine-tuned geometry control

## Related Tools

- **Inertia Calculator**: Compute precise inertia tensors for segments
- **URDF Builder GUI**: General-purpose robot model builder
- **C3D Viewer**: Import motion capture data for model animation
- **Biomechanics Analysis Suite**: Inverse dynamics calculations

## References

- de Leva, P. (1996). Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters. Journal of Biomechanics, 29(9), 1223-1230.
- Winter, D.A. (2009). Biomechanics and Motor Control of Human Movement. 4th ed. Wiley.
- Zatsiorsky, V.M. (2002). Kinetics of Human Motion. Human Kinetics.
- Drillis, R., & Contini, R. (1966). Body Segment Parameters. NTIS Report.

## Current Features

- Purpose: Build parametric humanoid characters with anthropometric calculations
- Category: Robotics
- Python files in tool path: 9
- Surface support: PyQt6=implemented, Web manifest=no, Web implementation=missing
- Test visibility: 0 name-matched test files under tests/

## Implementation State

- PyQt6 launcher: Implemented
- Web surface declared in manifest: No
- Web surface implementation: Gap / Not present
- README last reviewed: 2026-02-27

## Implementation Gaps

- No name-matched tests detected in repository-level tests/.
