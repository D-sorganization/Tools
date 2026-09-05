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

## 2024-06-10 - Eliminate Map/FromEntries overhead for object creation
**Learning:** Initializing objects with `Object.fromEntries(Object.entries(obj).map(...))` allocates intermediate arrays for entries, map results, and internal fromEntries representations.
**Action:** Always replace chained `Object.entries(obj).map()` object initializations with a pre-allocated empty object and a single-pass `for (const key of Object.keys(obj))` loop.
## 2024-06-25 - Avoid array iteration chaining for simple counts
**Learning:** When calculating aggregates (like counts or finding a maximum) over an array, using chained `.filter().length` and `.reduce()` operations creates unnecessary intermediate array allocations and executes multiple passes over the dataset, leading to increased garbage collection overhead in React components.
**Action:** Replace chained `.filter()` and `.reduce()` with a single-pass `for` loop to compute all needed aggregates simultaneously, especially in frequently re-rendered UI components.

## 2025-02-14 - Optimize repetitive array filters
**Learning:** Chained `.filter()` and `.reduce()` operations in high frequency code paths, like iterating through frames/phases during swing analysis, create unnecessary overhead due to memory allocation and callback invocation.
**Action:** Always replace chained `.filter()`/`.reduce()` iterations in tight algorithmic paths with a single-pass `for` loop, eliminating array allocations and garbage collection pressure.
## 2026-05-20 - Replace chained .filter() passes with single-pass for-loops
**Learning:** Multiple `.filter()` passes on every render create unnecessary intermediate array allocations, closure overhead, and force the JS engine to iterate the same arrays repeatedly. In React components, this triggers excessive garbage collection and blocks the main thread unnecessarily.
**Action:** Replace chained `.filter()` array passes with single-pass `for` loops in components to reduce array traversals from O(xN) to O(N), eliminate intermediate array allocations per render, and prevent GC pauses during high-frequency UI updates.

## 2024-05-31 - Fast NaN checks and Pairwise Precomputation
**Learning:** In hot loops computing pairwise relationships across large arrays (like Correlation Matrices), `Number.isNaN()` calls are extremely slow. Using the self-inequality check `x !== x` to identify `NaN` is significantly faster. Furthermore, if a single pass verifies there are zero `NaN` values in the dataset (the fast path), the `O(N^2)` combinatorial work can be drastically reduced by pre-computing sums (`sumX`, `sumX2`) per column, leaving only `sumXY` to be computed pair-wise.
**Action:** Replace `Number.isNaN()` with `x !== x` (or `x === x` for validity) inside dense numeric algorithmic loops. For pairwise O(N^2) calculations, scan for missing data once upfront to enable a "fast path" that pre-computes properties per column.
## 2024-07-28 - Replace Math.min/max spread with loops for dynamic scales
**Learning:** Using `Math.min(...activeValues)` and `Math.max(...activeValues)` on large streams of extracted subset data (e.g., when determining Y-axis scales in React charts) frequently leads to "Maximum call stack size exceeded" errors. Additionally, chaining operations like `flatMap` and `.map` to prepare this data dynamically creates massive memory pressure across high-frequency re-renders.
**Action:** When calculating min/max bounds across historically tracked subsets, avoid `.flatMap`, `.map`, and the `Math.min(...spread)` syntax entirely. Use a single-pass `for` loop that computes `realMin` and `realMax` dynamically, avoiding massive call stack allocations and garbage collection pressure.
## 2024-06-09 - Memoize and pull out string ops in React component filters
**Learning:** Chaining `.map().filter()` combined with `toLowerCase()` string operations inside an unmemoized React component render block causes huge performance drops, especially when typed inputs continuously trigger renders.
**Action:** Always wrap heavy list filtering/derivations in `useMemo`, pull static string operations (`.toLowerCase()`) outside of the filter callbacks, and replace array function chains with single-pass `for` loops using `Set` for distinct value extraction.

## 2026-06-12 - Pre-computing static mapped index arrays
**Learning:** In React components like the `RoutingMatrix` in `p1am_control_system`, using inline array initialization like `Array.from({ length: X }).map(...)` directly inside JSX rows allocates intermediate arrays on every render.
**Action:** Pre-calculate static length iterators as module-level constants and map over the constants instead.

## 2026-06-12 - Pre-allocate Arrays in Nelder-Mead Loops
**Learning:** In optimization loops like Nelder-Mead, allocating new arrays inside the hot iteration loop creates unnecessary garbage collection pressure.
**Action:** Pre-allocate working arrays such as `centroid`, `reflected`, `expanded`, and `contracted` outside the main algorithm loop and mutate them in place.
## 2026-06-12 - Avoid Array.from({ length }) in Math Hot Paths
**Learning:** In JavaScript/V8 numerical computing hot paths (like PCA or matrix calculations), using `Array.from({ length: N }, () => ...)` incurs significant overhead from iterability checks, iterator creation, and closure execution per element.
**Action:** Instead, pre-allocate arrays using `new Array(N)` and populate them with standard `for` loops to prevent O(N) intermediate garbage collection pressure.

## 2024-08-01 - Avoid multiple .filter() passes for bucketing
**Learning:** Calling `.filter()` multiple times on the same array to separate elements into different buckets (e.g. major vs moderate issues) creates unnecessary iterations and intermediate array allocations, adding up to GC pressure during recurring tasks.
**Action:** Replace multiple `.filter()` calls over the same source array with a single-pass `for` loop that pushes into pre-allocated or localized arrays.
## 2024-05-24 - Array.prototype.sort Overhead in Hot Loops
**Learning:** Discovered that for sorting tiny, statically-sized arrays (<= 20 elements) repeatedly inside high-frequency algorithmic hot loops (like Nelder-Mead optimization), `Array.prototype.sort()` incurs severe execution overhead due to callback invocation and closure allocation.
**Action:** Replace `Array.prototype.sort()` with a manual in-place insertion sort for tiny arrays inside hot paths to eliminate function call overhead and improve execution speed.
## 2024-07-16 - TabBar O(N^2) Optimization
**Learning:** Optimizing `Array.filter` chained with `includes` on a tiny array (10 items) to a `Set` offers zero measurable improvement.
**Action:** Avoid micro-optimizing operations that run on tiny static arrays executed during initialization or renders.

## 2024-05-31 - Fast NaN checks using Number.isNaN vs x !== x
**Learning:** In modern JavaScript engines like V8 (used in Chrome and Node.js), `Number.isNaN(v)` is an intrinsic function that is heavily optimized and compiled down to the exact same machine code instructions as the manual check `v !== v`. Replacing `Number.isNaN()` with `v !== v` does not provide any measurable performance improvement and only serves to degrade code readability.
**Action:** Do not micro-optimize `Number.isNaN()` checks into `v !== v` or `v === v`. Rely on the built-in semantics as modern engines handle them with zero overhead.
## 2026-06-12 - Eliminate map/reduce overhead for parsing assignments
**Learning:** Parsing simple string formats using `.split().reduce().map()` chains creates unnecessary array allocations, function calls, and closures on every pass, which adds noticeable garbage collection pressure when executing hot paths or frequent input changes.
**Action:** Replace string processing array chains with single-pass `for` loops and standard `indexOf`/`substring` operations to eliminate closure allocations and minimize object creations.
## 2024-07-26 - Single-pass loops for high-frequency React UI rendering
**Learning:** In high-frequency React UI rendering paths (e.g., pointer move events for SVG crosshairs), using chained `.map()` and `.reduce()` operations creates unnecessary garbage collection pressure due to intermediate array allocations and closure overhead.
**Action:** Replace chained `.map()` and `.reduce()` operations with a single-pass `for` loop to eliminate closure allocations and intermediate arrays, leading to smoother UI interactions.
## 2024-05-24 - Avoid chained map and every array iterations for parsing
**Learning:** Multiple array methods (`.map()`, `.every()`, `.filter()`) chained together for iterating over datasets cause unnecessary intermediate array allocations, adding up to increased garbage collection pressure.
**Action:** Replace multiple chained array passes with a single-pass `for` loop that pre-allocates arrays or calculates results inline.

## 2024-08-01 - Avoid allocating string arrays for SVG paths
**Learning:** In high-frequency chart updates, building SVG `d` paths using `.map(p => '...').join(' ')` allocates a new array of strings on every frame, causing unnecessary garbage collection pressure and main thread stalls.
**Action:** Build SVG `d` paths using a single-pass `for` loop and string concatenation to eliminate intermediate array allocations.
## 2025-05-18 - Avoid array methods for small static arrays in frequently called initializers
**Learning:** Using `.reduce()` or `.map()` on static arrays like tabs definitions inside frequently called functions (e.g. state initializers or local storage hydration) incurs unnecessary closure and function call overhead.
**Action:** Replace `.reduce()` and `.map()` with single-pass `for` loops in simple data transformation functions (like `defaultTabVisibility`) to eliminate closure allocations.
## 2026-08-13 - Replace chained .map().join() in CSV generation
**Learning:** Using chained array methods like `.map().join()` for large data serialization (like CSV exports) allocates intermediate arrays for every row, putting immense pressure on the garbage collector and stalling the main thread.
**Action:** Replace chained array map/join operations in data serialization hot paths with single-pass `for` loops and string concatenation to eliminate intermediate allocations.
## 2026-10-27 - Remove Array.from allocations in hot paths like Histogram renders
**Learning:** Found instances of `Array.from({ length: N }, ...)` directly inside React component render functions (e.g. `Histogram.tsx`). This allocates a new array, creates iterators, and calls a mapping function on every render, stalling the UI thread during drag/zoom events.
**Action:** Replace inline `Array.from` renders with IIFEs that pre-allocate using `new Array(N)` and iterate with a standard `for` loop, eliminating closure and iterator overhead per element.
## 2026-05-20 - Eliminate .forEach closure overhead in SVG hot paths
**Learning:** In high-frequency React UI rendering paths (e.g., highly dynamic animation frames building SVG paths), using `.forEach` inside hot render loops causes unnecessary closure allocation overhead and function call overhead for every point.
**Action:** Replace `.forEach` iterations with a standard `for` loop to eliminate closure allocation overhead and avoid function invocation penalties per data point in hot paths.
## 2025-02-12 - CSV Export Overhead
**Learning:** Chained array methods (.map().join()) in data-intensive hot paths (like exporting thousands of LaunchMonitor rows) create massive numbers of intermediate arrays, increasing GC pressure and memory consumption.
**Action:** Always replace chained declarative array operations with single-pass `for`-loops and string concatenation when generating large text payloads to bypass unnecessary memory allocations.

## 2024-05-18 - CSV Export Overhead
**Learning:** Chained array methods (.map().join()) in data-intensive hot paths (like exporting thousands of LaunchMonitor rows) create massive numbers of intermediate arrays, increasing GC pressure and memory consumption.
**Action:** Always replace chained declarative array operations with single-pass `for`-loops and string concatenation when generating large text payloads to bypass unnecessary memory allocations.
## 2026-12-07 - Avoid Array.from combined with map in mathematical matrices
**Learning:** Found instances of `Array.from({ length: cols })` inside `.map` during `designMatrix` construction for polynomial curve fitting in `torqueProfileEditor.ts`. This dynamically creates arrays and iterators in a tight numerical loop, adding closure and function call overhead which increases GC pressure.
**Action:** Always pre-allocate matrix/array dimensions using `new Array(size)` and populate them with standard `for` loops in mathematical or performance-sensitive hot paths to eliminate iteration overhead.
## 2024-08-30 - Replace Math.min/max spread with loops for dynamic scales
**Learning:** Using `Math.min(...spread)` and `Math.max(...spread)` on large streams of extracted subset data frequently leads to "Maximum call stack size exceeded" errors. It allocates a new array and spreads it out onto the call stack.
**Action:** When calculating min/max bounds across historically tracked subsets, avoid the `Math.min(...spread)` syntax entirely. Use a single-pass `for` loop that computes `realMin` and `realMax` dynamically, avoiding massive call stack allocations and garbage collection pressure.

## 2026-08-31 - Canvas Rendering Hot Path Optimization
**Learning:** In heavily populated canvas plots (like PlotCanvasCard), using `.forEach` for iterating over thousands of data points adds significant closure allocation and function call overhead per point, leading to increased CPU cycles and garbage collection pressure.
**Action:** Use standard `for` loops in canvas and SVG hot paths where large arrays of points are iterated.

## 2026-09-01 - Charting Scale Array Spreads O(N^2)
**Learning:** Computing chart bounds with `Math.max(...values)` and `Math.min(...values)` inline within a `.map()` operation over data arrays degrades rendering from O(N) to O(N^2) and generates extreme garbage collection pressure, leading to "Maximum call stack size exceeded" and UI thread blocking on large datasets.
**Action:** Always pre-calculate charting domain bounds explicitly outside of rendering loops using a single-pass O(N) standard `for` loop.

## 2024-09-01 - Avoid Array Spread in Grouping Loops
**Learning:** Using array spread syntax (`[...arr, item]`) inside tight loops to accumulate datasets by key degrades performance to O(N^2) and generates severe garbage collection pressure during React renders.
**Action:** Always initialize an array for the key and use direct mutation (`group.push(item)`) inside loops to achieve O(N) performance when grouping datasets.
## 2024-05-18 - Prevent stack overflow in dataset bounds calculation
**Learning:** Using `Math.max(...spread)` chained with `.map()` on large chart plotting paths creates massive call stack expansions and heavy garbage collection pressure, which could even throw "Maximum call stack size exceeded" on larger dynamic arrays.
**Action:** Use a single-pass `for` loop for minimum/maximum bound extraction in data-intensive charting paths.

## 2024-05-19 - Array map Math.abs spread overhead
**Learning:** Using `Math.max(...array.map(Math.abs))` creates an intermediate array and passes all items to the call stack via the spread operator, causing noticeable GC pressure and risking call stack size exceeded errors on large datasets.
**Action:** Replace `Math.max(...array.map(Math.abs))` with a standard O(N) single-pass `for` loop in numerical hotspots.
## 2024-11-20 - Chart Bounds Calculation O(N^2) Optimization
**Learning:** Using `Math.max(...spread)` and `Math.min(...spread)` mapped over large numerical arrays (e.g., inside `VariationScatter`'s `plotBounds`) causes extreme GC pressure and can trigger "Maximum call stack size exceeded" errors due to V8's argument limits.
**Action:** Always use a single-pass `for` loop to compute array extents when dealing with large visualization or numeric datasets.

## 2026-09-03 - Prevent stack overflows and GC pressure in tight loops
**Learning:** Using `Math.min(...arr)` or `Math.max(...arr)` on dynamically mapped arrays within render loops is an anti-pattern. It generates significant garbage collection overhead and risks stack overflows for large arrays.
**Action:** Replace `Math.min(...spread)` / `Math.max(...spread)` with a single-pass `for` loop to compute bounds. This eliminates intermediate allocations and runs substantially faster while scaling to any data size.
