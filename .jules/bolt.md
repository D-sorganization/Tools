## 2024-05-14 - Optimize Array Methods in PCA Calculation
**Learning:** Chained array methods like `.reduce()` and `.map()` on arrays (e.g. eigenvalues in PCA calculations) cause noticeable garbage collection and allocation overhead in performance-critical sections.
**Action:** Replace chained `.reduce()` and `.map()` calls with single-pass `for` loops that pre-allocate target arrays (`new Array(size)`) to avoid intermediate array creation.
## 2026-04-25 - Optimize 2D matrix transpositions
**Learning:** Chained array methods like `.map()` to transpose a 2D matrix cause severe O(N^2) memory allocation overhead and garbage collection pauses during calculations.
**Action:** Use pre-allocated nested `for` loops for transposing 2D arrays in performance-critical areas instead of chained iterators.
## 2024-05-24 - Maximize CPU Cache Locality in Column-Major Matrices
**Learning:** When computing PCA scores over column-major data implemented as `Float64Array[]`, a traditional row-major inner loop severely thrashes the CPU cache by hopping between disjoint memory buffers.
**Action:** Reordered the nested loops to ensure the innermost loop iterates sequentially over rows for a given column (`cols[j][i]`), resulting in a sequential memory access pattern that drastically reduces execution time.
