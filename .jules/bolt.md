## 2024-07-26 - Eliminate intermediate array allocations and closure overhead in high-frequency React UI rendering paths
**Learning:** To reduce garbage collection pressure in high-frequency React UI rendering paths (e.g., pointer move events for SVG crosshairs), chained `.map()` and `.reduce()` operations create unnecessary intermediate arrays and involve closure allocations.
**Action:** Replace chained `.map()` and `.reduce()` operations with single-pass `for` loops to eliminate closure allocations and intermediate arrays in high-frequency rendering paths.
