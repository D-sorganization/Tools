## 2024-05-01 - Optimization of ODESolverCalculator Downsampling
**Learning:** Chaining `.filter()` and `.map()` with an index check (`i % step === 0`) on large result arrays is inefficient, causing O(N) intermediate array allocations and slowing down rendering loops.
**Action:** Replace `.filter().map()` chains with single-pass manual loops pre-allocated to the correct size or standard downsampling iterators to minimize garbage collection overhead.
## 2024-05-01 - Optimization of Results Iteration in ODESolverCalculator
**Learning:** Using `results.map` followed by `Math.min(...values)` and `Math.max(...values)` on large dataset arrays in JS causes excessive memory allocation (the intermediate mapped array) and throws "Maximum call stack size exceeded" errors if the array size exceeds the stack limit.
**Action:** Replace `map` and spread operations for min/max calculations with a single-pass `for` loop tracking min and max manually.
## 2024-05-15 - Optimization of Array Allocations in FunctionGenerator
**Learning:** Using `time.map(() => 0)` or `time.map((_, i) => ...)` to generate large high-frequency signal arrays inside React components triggers significant garbage collection overhead and intermediate closures for every data point.
**Action:** Replace `.map()` on large sample rate arrays with pre-allocated arrays (`new Array(n)`) combined with `.fill(0)` for zeroing out, or single-pass `for` loops when computing mathematical sums over signal layers.
