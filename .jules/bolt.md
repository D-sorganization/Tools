## 2024-05-14 - Optimize Array Methods in PCA Calculation
**Learning:** Chained array methods like `.reduce()` and `.map()` on arrays (e.g. eigenvalues in PCA calculations) cause noticeable garbage collection and allocation overhead in performance-critical sections.
**Action:** Replace chained `.reduce()` and `.map()` calls with single-pass `for` loops that pre-allocate target arrays (`new Array(size)`) to avoid intermediate array creation.

## 2024-05-15 - Improve Cache Locality in Matrix Operations
**Learning:** Reversing loop order for matrix operations (like calculating PCA scores from column-major `Z_cols` arrays) drastically improves CPU cache locality, significantly reducing execution time by keeping the innermost loop within a contiguous block of memory (a single Typed Array column) instead of fetching disparate rows.
**Action:** When iterating over column-major arrays (e.g., `Float64Array[]`), ensure the innermost loop traverses elements sequentially within a single column rather than jumping across columns.
