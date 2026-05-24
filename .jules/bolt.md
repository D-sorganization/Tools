## 2024-05-24 - Array Pre-allocation over map
**Learning:** When optimizing high-frequency event handlers in JavaScript/TypeScript (e.g., pose detection over multiple video frames), replacing array iterators like `.map()` with standard `for` loops and pre-allocating arrays eliminates continuous callback allocation and minimizes garbage collection overhead. (Note: Only applies to large arrays or high-frequency loops; tiny arrays provide zero measurable performance benefit, and shouldn't be touched per project guidelines).
**Action:** Always prefer standard `for` loops over iterators for large arrays inside high-frequency execution pathways to eliminate callback allocation and GC pauses.

## 2024-05-27 - Optimizing Map and Slice in Hot Loops
**Learning:** In optimization loops like Nelder-Mead (e.g. `src/pendulum_simulator/pendulum-web/src/optimizer.ts`), the repeated use of array prototype methods like `.map()` and `.slice()` inside algorithmic iterations causes severe garbage collection pauses due to intermediate array allocations and closure creation. Standard `for` loops combined with pre-allocated arrays (e.g. `new Array(size)`) avoid these overheads and can execute over 2-3x faster.
**Action:** When working on numerical optimizers or simulation inner loops in JS/TS, manually rewrite `.map()`, `.reduce()`, and `.slice()` into explicit `for` loops with pre-allocated arrays.

## 2024-05-30 - Downsampling Overheads in React Memos
**Learning:** In high-frequency rendering components (like React chart wrappers), downsampling massive streams of state data via chained array iterators (e.g., `indices.map()`) inside `useMemo` hooks causes severe framerate drops. The intermediate array creations for every trace during chart updates trigger massive O(N) memory allocations and garbage collection pauses.
**Action:** When extracting data subsets for visualizations, use explicit `for` loops combined with pre-allocated arrays (e.g. `new Array(len)`) to dramatically reduce memory allocation and eliminate iterator callback overhead.

## 2026-05-19 - Replace .reduce() with for loops in high-frequency confidence calculation
**Learning:** When calculating aggregates over large arrays (like video pose frames) in high-frequency paths, using `.reduce()` creates unnecessary callback allocation and GC overhead. A standard `for` loop is faster and avoids memory pressure.
**Action:** Always replace `.reduce()` with standard `for` loops when computing sums or averages over large datasets in hot paths.
## 2024-05-30 - Memoizing Render-Blocking O(N) Array Operations in React
**Learning:** In React components that manage high-frequency inputs (e.g. text areas for calculator expressions) alongside large arrays of data (e.g. an array of 10k generated data points), rendering unmemoized array loops like `.map()` combined with local `min`/`max` loops blocks the main thread. This leads to severe lag when typing in the input fields, as React re-evaluates the large data arrays on every keystroke.
**Action:** Always wrap heavy O(N) loops that aggregate state data (such as finding `min`/`max` limits across large solution arrays for summary cards) in a `useMemo` block, with dependency arrays scoped strictly to the generated result data.
## 2024-05-23 - In-place Mutation for Integration Loops
**Learning:** In tight numerical integration loops (like RK4), instantiating and returning new state array objects per step causes severe garbage collection pauses and frame drops, even if the loops themselves are manually unrolled.
**Action:** Use an out-parameter pattern where pre-allocated state objects and argument arrays are instantiated once outside the loop and mutated continuously.
## 2024-05-23 - SVG Chart Data Overload
**Learning:** Passing raw, high-resolution arrays (>1000 points) directly to React charting libraries like Recharts creates massive DOM/SVG nodes, causing severe main thread blocking and unresponsive UI.
**Action:** Always downsample large result arrays via `useMemo` with single-pass loops and pre-allocated arrays (e.g. `new Array(len)`) to a visual maximum (~500 points) before passing them to charting components.## 2025-02-14 - Unmemoized Charts block UI
**Learning:** Unmemoized Recharts components wrapped alongside frequent input controls (like textareas) will cause severe input lag because the entire DOM/SVG tree re-renders on every keystroke.
**Action:** Always extract heavy data-visualization JSX into `useMemo` or `React.memo` when they sit adjacent to fast-updating inputs in the same component tree.
