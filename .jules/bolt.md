
## 2026-08-31 - Canvas Rendering Hot Path Optimization
**Learning:** In heavily populated canvas plots (like PlotCanvasCard), using `.forEach` for iterating over thousands of data points adds significant closure allocation and function call overhead per point, leading to increased CPU cycles and garbage collection pressure.
**Action:** Use standard `for` loops in canvas and SVG hot paths where large arrays of points are iterated.
