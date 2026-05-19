## 2024-05-24 - Array Pre-allocation over map
**Learning:** When optimizing high-frequency event handlers in JavaScript/TypeScript (e.g., pose detection over multiple video frames), replacing array iterators like `.map()` with standard `for` loops and pre-allocating arrays eliminates continuous callback allocation and minimizes garbage collection overhead. (Note: Only applies to large arrays or high-frequency loops; tiny arrays provide zero measurable performance benefit, and shouldn't be touched per project guidelines).
**Action:** Always prefer standard `for` loops over iterators for large arrays inside high-frequency execution pathways to eliminate callback allocation and GC pauses.

## 2024-05-27 - Optimizing Map and Slice in Hot Loops
**Learning:** In optimization loops like Nelder-Mead (e.g. `src/pendulum_simulator/pendulum-web/src/optimizer.ts`), the repeated use of array prototype methods like `.map()` and `.slice()` inside algorithmic iterations causes severe garbage collection pauses due to intermediate array allocations and closure creation. Standard `for` loops combined with pre-allocated arrays (e.g. `new Array(size)`) avoid these overheads and can execute over 2-3x faster.
**Action:** When working on numerical optimizers or simulation inner loops in JS/TS, manually rewrite `.map()`, `.reduce()`, and `.slice()` into explicit `for` loops with pre-allocated arrays.
## 2026-05-19 - Replace .reduce() with for loops in high-frequency confidence calculation
**Learning:** When calculating aggregates over large arrays (like video pose frames) in high-frequency paths, using `.reduce()` creates unnecessary callback allocation and GC overhead. A standard `for` loop is faster and avoids memory pressure.
**Action:** Always replace `.reduce()` with standard `for` loops when computing sums or averages over large datasets in hot paths.
