# Assessment N: Visualization & Export

## Executive Summary
**Score: 6/10**
**Severity: MINOR**

Visualization capabilities are adequate but disjointed. The move to WebGL (Three.js) in `urdf_viewer` is a major upgrade over Matplotlib/PyQt graphs.

## Key Findings

### 1. 3D Visualization
- **Strengths**: `urdf_viewer` provides high-quality, interactive 3D rendering.
- **Weaknesses**: Collision geometry debugging is still basic.

### 2. 2D Plotting
- **Strengths**: `src/python/src/utils/plotting.py` provides shared plotting logic.
- **Weaknesses**: Legacy tools often embed plotting logic directly (`God functions`), ignoring the shared library.

### 3. Export
- **Issue**: Export formats are limited. Screenshots? Vector graphics (SVG/PDF)?
- **Status**: Unclear if users can generate publication-quality figures directly.

## Recommendations
1. **Unify Plotting**: Refactor all 2D plotting to use `utils/plotting.py`.
2. **Export Options**: Add "Save as SVG" and "Save as High-Res PNG" buttons to all visualization widgets.
3. **Theming**: Implement a consistent color theme (e.g., "Scientific Dark Mode") across all tools.
