## 2024-05-14 - Optimize Array Methods in PCA Calculation
**Learning:** Chained array methods like `.reduce()` and `.map()` on arrays (e.g. eigenvalues in PCA calculations) cause noticeable garbage collection and allocation overhead in performance-critical sections.
**Action:** Replace chained `.reduce()` and `.map()` calls with single-pass `for` loops that pre-allocate target arrays (`new Array(size)`) to avoid intermediate array creation.
## 2026-04-25 - Optimize 2D matrix transpositions
**Learning:** Chained array methods like `.map()` to transpose a 2D matrix cause severe O(N^2) memory allocation overhead and garbage collection pauses during calculations.
**Action:** Use pre-allocated nested `for` loops for transposing 2D arrays in performance-critical areas instead of chained iterators.
## 2024-05-18 - Split Array Windowing to Avoid Bounds Checking in Hot Paths
**Learning:** For array sliding-window algorithms (like median or Gaussian filtering), running `Math.min` and `Math.max` bounds checks on every iteration of the main loop adds significant overhead.
**Action:** Split the loop into three parts: left edge, middle section, and right edge. The middle section (the hot path) can then run without bounds checking, which noticeably speeds up execution for large arrays.
