## 2024-05-14 - Optimize Array Methods in PCA Calculation
**Learning:** Chained array methods like `.reduce()` and `.map()` on arrays (e.g. eigenvalues in PCA calculations) cause noticeable garbage collection and allocation overhead in performance-critical sections.
**Action:** Replace chained `.reduce()` and `.map()` calls with single-pass `for` loops that pre-allocate target arrays (`new Array(size)`) to avoid intermediate array creation.
