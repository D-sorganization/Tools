# C3D Motion Capture Viewer

A PyQt6-based GUI application for viewing and analyzing C3D motion capture files, with marker visualization and data export capabilities.

## Purpose

The C3D Viewer provides a comprehensive interface for inspecting C3D (Coordinate 3D) motion capture files commonly used in biomechanics, gait analysis, and animation. It displays metadata, marker labels, analog channels, and force plate data, with export functionality for further analysis.

## Key Features

- **File Metadata Display**: Frame count, sample rate, duration, and units
- **Marker List View**: Browse all marker labels with selection capability
- **Trajectory Analysis**: Statistical analysis of selected marker trajectories
- **Analog Channel Support**: View EMG, force, and other analog data channels
- **Force Plate Analysis**: Detect and analyze force plate configurations
- **Multi-Format Export**: Export to CSV, JSON, or NPZ formats
- **Unit Conversion**: Convert marker data between mm, cm, and m
- **Frame Range Selection**: Export specific frame ranges

## Installation / Prerequisites

### Dependencies

```bash
pip install PyQt6
pip install ezc3d  # Required for actual C3D file reading
```

### Running the Application

```bash
python -m c3d_viewer.ui.pyqt6.main_window
# or
python launch_pyqt6.py
```

**Note**: Without the ezc3d library, the viewer operates in demo mode with sample data.

## Usage Instructions

### Loading a File

1. Click "Load C3D File" button
2. Select a .c3d file from the file browser
3. The file metadata will populate automatically

### Metadata Tab

Displays recording information:

- Number of markers
- Total frame count
- Frame rate (Hz)
- Recording duration (seconds)
- Coordinate units
- Analog channel count and sample rate
- Event markers (heel strikes, toe-offs, etc.)

### Markers Tab

1. View all marker labels in the list
2. Select one or more markers (Ctrl+click for multiple)
3. Click "Analyze Selected Markers" for trajectory statistics

### Analog Channels Tab

1. View analog channel labels, units, and sample counts
2. Click "Analyze Force Plates" to detect force plate configuration
3. Displays force (Fx, Fy, Fz) and moment (Mx, My, Mz) channels

### Export Tab

1. Select export format (CSV, JSON, NPZ)
2. Choose target units (Original, m, mm, cm)
3. Set start and end frame range
4. Click export button for desired data type:
   - **Export Marker Data**: 3D point trajectories
   - **Export Analog Data**: All analog channels
   - **Export Force Plate Data**: Computed ground reaction forces

## Input Parameters

### C3D File Format

The application reads standard C3D files containing:

| Data Type   | Description                               |
| ----------- | ----------------------------------------- |
| Point Data  | 3D marker coordinates (X, Y, Z) per frame |
| Analog Data | Time-series data from analog devices      |
| Events      | Labeled time points (e.g., gait events)   |
| Parameters  | Recording metadata and calibration        |

### Export Options

| Parameter    | Values              | Description                |
| ------------ | ------------------- | -------------------------- |
| Format       | CSV, JSON, NPZ      | Output file format         |
| Target Units | Original, m, mm, cm | Coordinate unit conversion |
| Start Frame  | 0 - max             | First frame to export      |
| End Frame    | 0 - max             | Last frame to export       |

## Output Format

### CSV Export

Marker data is exported with columns for each marker:

```csv
Frame,Time,LASI_X,LASI_Y,LASI_Z,RASI_X,RASI_Y,RASI_Z,...
0,0.000,234.5,156.2,987.3,245.1,152.8,985.9,...
1,0.010,234.6,156.3,987.2,245.2,152.9,985.8,...
```

### JSON Export

Structured data with metadata:

```json
{
  "metadata": {
    "frame_rate": 100.0,
    "units": "mm",
    "marker_count": 39
  },
  "markers": {
    "LASI": [[234.5, 156.2, 987.3], [234.6, 156.3, 987.2], ...],
    "RASI": [[245.1, 152.8, 985.9], ...]
  }
}
```

### NPZ Export (NumPy)

Binary format for efficient Python loading:

```python
import numpy as np
data = np.load('export.npz')
markers = data['markers']  # Shape: (n_markers, n_frames, 3)
labels = data['labels']    # Marker names
time = data['time']        # Time vector
```

## Mathematical Models

### Coordinate Systems

C3D files typically use a right-handed coordinate system:

- **X**: Anterior-posterior (forward positive)
- **Y**: Medial-lateral (left positive)
- **Z**: Vertical (up positive)

### Unit Conversion

```
1 m = 1000 mm = 100 cm
```

Conversion is applied to all XYZ coordinates during export.

### Frame-to-Time Conversion

```
time = frame_number / frame_rate
```

### Force Plate Data

Ground reaction force from force plate channels:

```
GRF = [Fx, Fy, Fz]  (Newtons)
COP = [COPx, COPy]  (Center of Pressure, in mm)
Moments = [Mx, My, Mz]  (Newton-meters)
```

## Example Usage

### Example 1: Gait Analysis File

Loading a walking trial:

- File: walk_01.c3d
- Markers: 39 (full-body Plug-in Gait)
- Frames: 500
- Frame Rate: 100 Hz
- Duration: 5.0 seconds

Metadata shows:

- 4 events (LHS, RTO, RHS, LTO)
- 2 force plates
- 6 analog channels per plate

### Example 2: Export for Python Analysis

1. Load file: run_trial.c3d
2. Export Tab: Select NPZ format
3. Units: meters
4. Frame range: 100 - 400 (steady-state running)
5. Export Marker Data

```python
import numpy as np
data = np.load('run_trial_points.npz')
heel_z = data['markers'][data['labels'] == 'RHEE', :, 2]
```

### Example 3: Force Plate Ground Reaction

1. Load file with force plate data
2. Navigate to Analog Channels tab
3. Click "Analyze Force Plates"
4. Export force plate data to CSV

## Troubleshooting

### "ezc3d library not available"

The application shows demo data without ezc3d:

```bash
pip install ezc3d
```

On some systems, you may need to install from conda:

```bash
conda install -c conda-forge ezc3d
```

### "No force plates detected in file"

Force plate data requires:

- Properly configured FORCE_PLATFORM parameter group
- Analog channels mapped to force plate outputs
- Calibration matrices in the C3D parameters

### Markers appear at incorrect positions

Check the coordinate system convention:

- Some systems use Y-up instead of Z-up
- Lab coordinate system may differ from file coordinates
- Verify units match expected scale (mm vs m)

### Large files load slowly

For files with many frames (>10,000):

- Consider exporting a subset of frames
- Use NPZ format for faster subsequent loading
- Ensure sufficient system RAM

### Export produces empty file

Verify that:

- A C3D file is loaded (green filename indicator)
- Selected markers or channels exist in the file
- Start frame < End frame

## Related Tools

- **Inertia Calculator**: Compute segment inertias from marker positions
- **URDF Builder GUI**: Create robot models for simulation
- **Humanoid Builder GUI**: Generate biomechanical models
- **Biomechanics Toolkit (BTK)**: Alternative C3D library

## References

- C3D.org File Format Documentation: https://www.c3d.org/
- ezc3d Library: https://github.com/pyomeca/ezc3d
- Motion Analysis Corporation C3D Specification
- Vicon Documentation: C3D File Format Reference
- ISB Recommendations for Coordinate Systems

## Current Features

- Purpose: View and analyze C3D motion capture files
- Category: Biomechanics
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
