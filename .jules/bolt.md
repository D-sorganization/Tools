## 2024-05-01 - Optimization of ODESolverCalculator Downsampling
**Learning:** Chaining `.filter()` and `.map()` with an index check (`i % step === 0`) on large result arrays is inefficient, causing O(N) intermediate array allocations and slowing down rendering loops.
**Action:** Replace `.filter().map()` chains with single-pass manual loops pre-allocated to the correct size or standard downsampling iterators to minimize garbage collection overhead.
## 2024-05-01 - Optimization of Results Iteration in ODESolverCalculator
**Learning:** Using `results.map` followed by `Math.min(...values)` and `Math.max(...values)` on large dataset arrays in JS causes excessive memory allocation (the intermediate mapped array) and throws "Maximum call stack size exceeded" errors if the array size exceeds the stack limit.
**Action:** Replace `map` and spread operations for min/max calculations with a single-pass `for` loop tracking min and max manually.
## 2024-05-19 - Optimization of Sliding Window Algorithms
**Learning:** Using `.slice()` and `.reduce()` inside a sliding-window loop (like `smoothAngles`) creates massive O(N) array allocations and garbage collection overhead. Furthermore, running `Math.min()` and `Math.max()` bounds checks on every iteration of the main loop adds significant execution time.
**Action:** Replace slice/reduce chains in sliding-window algorithms with manual inner loops tracking the sum. To eliminate bounds checking in the hot path, split the main loop into three parts: left edge, middle section (running without bounds checking), and right edge.
## 2024-05-19 - Optimization of Object Property Copying in Tight Loops
**Learning:** Using a `for...in` loop combined with `Object.prototype.hasOwnProperty.call()` to copy object properties is significantly slower than using `Object.keys()` combined with a standard `for` loop, because `for...in` crawls the entire prototype chain.
**Action:** When copying own properties of objects inside tight data processing loops, use `Object.keys()` with a standard `for` loop. It natively filters own properties and avoids the overhead of prototype chain traversal and repeated function calls.
## 2024-05-19 - Optimization of High-Frequency Event Handlers
**Learning:** Using array iterator methods like `.reduce()` inside event handlers that run per-frame (e.g., 30-60 times a second, like MediaPipe pose detection callbacks) introduces continuous callback allocation overhead and can hurt overall framerate.
**Action:** Replace iterator methods like `.reduce()`, `.map()`, and `.filter()` with standard `for` loops inside high-frequency real-time event handlers to eliminate callback overhead and minimize garbage collection.
## 2024-05-19 - Optimization of Equation Evaluation in Tight Loops
**Learning:** Instantiating `new Function(...)` inside a tight numerical loop (e.g. within an RK4 solver integrating thousands of time steps) bypasses JavaScript engine optimizations, results in continuous parsing overhead, and causes extreme garbage collection pauses, dragging down overall loop performance.
**Action:** When evaluating dynamic math equations multiple times, pre-compile the string expression into a JavaScript function via `new Function` *once* outside the loop, passing variable names as arguments. Inside the loop, simply call the pre-compiled function with the current numeric values.
