## 2024-05-14 - Optimize Array Methods in PCA Calculation
**Learning:** Chained array methods like `.reduce()` and `.map()` on arrays (e.g. eigenvalues in PCA calculations) cause noticeable garbage collection and allocation overhead in performance-critical sections.
**Action:** Replace chained `.reduce()` and `.map()` calls with single-pass `for` loops that pre-allocate target arrays (`new Array(size)`) to avoid intermediate array creation.
## 2026-04-25 - Optimize 2D matrix transpositions
**Learning:** Chained array methods like `.map()` to transpose a 2D matrix cause severe O(N^2) memory allocation overhead and garbage collection pauses during calculations.
**Action:** Use pre-allocated nested `for` loops for transposing 2D arrays in performance-critical areas instead of chained iterators.
## 2024-05-18 - Split Array Windowing to Avoid Bounds Checking in Hot Paths
**Learning:** For array sliding-window algorithms (like median or Gaussian filtering), running `Math.min` and `Math.max` bounds checks on every iteration of the main loop adds significant overhead.
**Action:** Split the loop into three parts: left edge, middle section, and right edge. The middle section (the hot path) can then run without bounds checking, which noticeably speeds up execution for large arrays.
## 2024-05-18 - Optimize array statistics and chained iteration on large datasets
**Learning:** Using the spread operator (`...vals`) inside `Math.min()` or `Math.max()` on large signal data arrays causes a "Maximum call stack size exceeded" error. Similarly, using chained `.map().filter()` or `.reduce()` calls on such datasets incurs significant overhead and triggers excessive garbage collection pauses.
**Action:** Replace `Math.min(...vals)` and chained iterators with single-pass `for` loops to manually track statistics and build subset arrays without intermediate array allocation.
## 2024-05-24 - Maximize CPU Cache Locality in Column-Major Matrices
**Learning:** When computing PCA scores over column-major data implemented as `Float64Array[]`, a traditional row-major inner loop severely thrashes the CPU cache by hopping between disjoint memory buffers.
**Action:** Reordered the nested loops to ensure the innermost loop iterates sequentially over rows for a given column (`cols[j][i]`), resulting in a sequential memory access pattern that drastically reduces execution time.
## 2024-05-25 - Avoid Dynamic Array Resizing and .reduce() Overhead in FFT Windowing
**Learning:** In fast Fourier transform calculations (`computeFFT`), inline array construction and functional array methods (like `.map()` and `.reduce()`) introduce dynamic array resizing overhead and intermediary array allocations, stalling signal updates.
**Action:** Replaced dynamic `.push()` calls and array method chains with pre-allocated arrays (`new Array(n)`) and a single-pass `for` loop to inline Hanning windowing, which significantly boosts real-time processing throughput.
## 2026-04-30 - Optimize Array Downsampling in Chart Data Generation
**Learning:** When downsampling large arrays in JS/TS (e.g., extracting data points for charts using a calculated `step`), avoid chained iterators like `.filter((_, i) => i % step === 0).map(...)`. These execute in O(N) iterations and allocate intermediate arrays.
**Action:** Instead, use a single-pass `for` loop that increments by `step` (`for (let i = 0; i < len; i += step)`) to reduce iterations to O(N/step) and eliminate garbage collection overhead.
## 2024-05-18 - Avoid array methods chained with mapped row building for large datasets
**Learning:** Generating downsampled data table views using chained `.filter().map()` causes massive O(N) intermediate array generation and garbage collection pauses when running on large `results` arrays.
**Action:** Replaced chained iterators with an inline IIFE containing a single-pass `for` loop to increment by the calculated `step`, allowing O(N/step) performance without intermediate mapping allocations.
