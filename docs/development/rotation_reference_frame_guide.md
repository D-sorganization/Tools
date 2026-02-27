# Rotation Tool: Reference-Frame and Lie-Group Guide

## Purpose
The rotation tool now includes an educational reference-frame module for:
- Twist conversion between frames using adjoint transforms.
- Homogeneous transform construction/decomposition in SE(3).
- so(3) to SO(3) mappings using hat/vee, matrix exponential, and matrix logarithm.

This guide explains both how to operate the features and the core mathematics.

## UI Workflow
1. Open the rotation tool web UI.
2. Select the `Reference Frames & Lie Groups` tab.
3. Choose one operation:
- `Twist Frame Conversion (Adjoint)`
- `Homogeneous Transform Builder`
- `so(3) ↔ SO(3) Exponential / Log`
4. Enter numeric inputs.
5. Click `Compute`.
6. Read:
- `Results (JSON)` for machine-usable values.
- `Explanation (Markdown)` for conceptual guidance.
- `Formulas (LaTeX)` for precise mathematical notation.

## API Endpoint
`POST /api/calc/rotation-converter/reference-frame`

### Operation: Twist Frame Conversion
Request example:
```json
{
  "operation": "twist_frame_conversion",
  "transform": [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ],
  "twist": [0.1, 0.2, 0.3, 1.0, 2.0, 3.0]
}
```

Mathematics:
- A homogeneous transform has structure:
  - \( T = \begin{bmatrix} R & p \\ 0 & 1 \end{bmatrix} \)
- Twists transform by:
  - \( V_b = \mathrm{Ad}_T V_a \)
- Adjoint:
  - \( \mathrm{Ad}_T = \begin{bmatrix} R & 0 \\ [p]_\times R & R \end{bmatrix} \)

Here, \([p]_\times\) is the skew matrix of \(p\).

### Operation: Homogeneous Transform Builder
Request example:
```json
{
  "operation": "homogeneous_transform",
  "rotation_matrix": [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
  ],
  "translation": [1, 2, 3]
}
```

Mathematics:
- Build:
  - \( T = \begin{bmatrix} R & p \\ 0 & 1 \end{bmatrix} \)
- Inverse:
  - \( T^{-1} = \begin{bmatrix} R^\top & -R^\top p \\ 0 & 1 \end{bmatrix} \)

Usage:
- Position transformation (homogeneous coordinates):
  - \( x_b = T x_a \)
- Frame inversion:
  - Use \(T^{-1}\) to map from `b` back to `a`.

### Operation: so(3) ↔ SO(3)
Request example:
```json
{
  "operation": "so3_so3_maps",
  "so3_vector": [0, 0, 0.5]
}
```

Mathematics:
- Hat map:
  - \( \omega \in \mathbb{R}^3 \mapsto \widehat{\omega} \in \mathfrak{so}(3) \)
  - \( \widehat{\omega} = \begin{bmatrix}
0 & -\omega_3 & \omega_2 \\
\omega_3 & 0 & -\omega_1 \\
-\omega_2 & \omega_1 & 0
\end{bmatrix} \)
- Exponential map:
  - \( R = \exp(\widehat{\omega}) \in SO(3) \)
- Log map:
  - \( \omega = \mathrm{vee}(\log R) \)

Interpretation:
- so(3) stores infinitesimal/axis-angle rotational information.
- SO(3) stores finite rotation matrices.
- exp/log move between algebra and group representations.

## Practical Notes
- Input validation enforces matrix/vector dimensions.
- Rotation matrices are validated for SO(3)-compatibility.
- Educational text is returned with every operation for instructional use.

## Testing
Coverage for this feature lives in:
- `tests/calc_backend/test_rotation_converter_api.py`

These tests validate:
- Twist conversion behavior for identity transforms.
- Homogeneous matrix assembly shape/structure.
- so(3)/SO(3) output structure for map operations.
