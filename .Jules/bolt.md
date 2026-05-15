## 2024-05-01 - Optimization of ODESolverCalculator Downsampling
**Learning:** Chaining `.filter()` and `.map()` with an index check (`i % step === 0`) on large result arrays is inefficient, causing O(N) intermediate array allocations and slowing down rendering loops.
**Action:** Replace `.filter().map()` chains with single-pass manual loops pre-allocated to the correct size or standard downsampling iterators to minimize garbage collection overhead.
## 2024-05-01 - Optimization of Results Iteration in ODESolverCalculator
**Learning:** Using `results.map` followed by `Math.min(...values)` and `Math.max(...values)` on large dataset arrays in JS causes excessive memory allocation (the intermediate mapped array) and throws "Maximum call stack size exceeded" errors if the array size exceeds the stack limit.
**Action:** Replace `map` and spread operations for min/max calculations with a single-pass `for` loop tracking min and max manually.
## 2024-05-14 - Optimization of Gas Composition Object Iteration in PressureDropCalculator
**Learning:** Using `Object.values().reduce()` and `Object.entries().map()` inside hot rendering loops causes O(N) intermediate array allocations and excessive garbage collection overhead.
**Action:** Replace these operations with a single-pass standard `for` loop over `Object.keys()` to prevent intermediate object allocations and minimize garbage collection.
## 2024-05-19 - Object vs Array access in numerical hot paths
**Learning:** Tight numerical integration loops (like RK4 solvers in JS/TS) that use Object lookup for variables (`currentState[varName]`) are significantly slower due to property hashing overhead in JavaScript engines.
**Action:** Replace Object states with contiguous Arrays (`new Array<number>(len)`) and use positional index mapping in tight loops. This avoids the hashing overhead and massively improves execution speed.

## 2026-05-15 - Optimize RK4 Expression Compilation Array Destructuring
**Learning:** When compiling mathematical expressions dynamically using `new Function(...)` to evaluate inside tight integration loops like RK4, spreading parameters (e.g. `(...args)` and calling with `f(...args)`) introduces significant and continuous garbage collection overhead and prevents JavaScript engines from optimizing function calls due to dynamic arity.
**Action:** When dynamically generating expressions, define the function signature to accept a single arguments array (`args: number[]`) and generate static declarations from the array based on positional indices within the function body itself (e.g. `const x = args[0];`). This avoids spread allocation entirely while keeping the invocation simple.
