## 2024-05-01 - Optimization of ODESolverCalculator Downsampling
**Learning:** Chaining `.filter()` and `.map()` with an index check (`i % step === 0`) on large result arrays is inefficient, causing O(N) intermediate array allocations and slowing down rendering loops.
**Action:** Replace `.filter().map()` chains with single-pass manual loops pre-allocated to the correct size or standard downsampling iterators to minimize garbage collection overhead.
## 2024-05-01 - Optimization of Results Iteration in ODESolverCalculator
**Learning:** Using `results.map` followed by `Math.min(...values)` and `Math.max(...values)` on large dataset arrays in JS causes excessive memory allocation (the intermediate mapped array) and throws "Maximum call stack size exceeded" errors if the array size exceeds the stack limit.
**Action:** Replace `map` and spread operations for min/max calculations with a single-pass `for` loop tracking min and max manually.
## 2024-05-15 - Optimization of Array Allocations in FunctionGenerator
**Learning:** Using `time.map(() => 0)` or `time.map((_, i) => ...)` to generate large high-frequency signal arrays inside React components triggers significant garbage collection overhead and intermediate closures for every data point.
**Action:** Replace `.map()` on large sample rate arrays with pre-allocated arrays (`new Array(n)`) combined with `.fill(0)` for zeroing out, or single-pass `for` loops when computing mathematical sums over signal layers.
## 2024-05-18 - Optimization of Polynomial Evaluation in High-Frequency Loops
**Learning:** Evaluating polynomials in tight high-frequency loops (like RK4 physics integration or signal generation) using array `.reduce()` combined with exponentiation (`t ** i` or `Math.pow()`) or chained with `.map()` causes severe parsing overhead, object allocation, and garbage collection pauses.
**Action:** Implement Horner's method using a standard reverse `for` loop (`acc = acc * t + coeffs[i]`) and a pre-allocated array to eliminate callback allocation overhead and replace expensive power operations with simple multiplication, yielding up to 20x performance gains.

## 2024-05-18 - Optimization of Numerical Integration Loops
**Learning:** Creating thousands of short-lived objects (like intermediate states or argument arrays) per numerical integration step (e.g., RK4) causes severe garbage collection pauses and degrades simulation performance.
**Action:** Avoid in-loop allocations. Pre-allocate state objects and argument arrays outside the integration loop, and use an out-parameter pattern (e.g., passing `outDerivs` to the derivative function) to mutate existing objects rather than returning new ones.
## 2024-05-18 - Optimization of Array Allocations and Reductions in PSA Calculator
**Learning:** Using multiple chained `.map()` allocations and array `.reduce()` inside tight calculation steps generates numerous intermediate arrays, leading to O(N) memory allocation and subsequent garbage collection overhead.
**Action:** Replace multiple `.map()` and `.reduce()` invocations with a single-pass `for` loop that uses pre-allocated arrays (e.g., `new Array(size)`) and accumulates totals manually in high-frequency data calculation modules.
