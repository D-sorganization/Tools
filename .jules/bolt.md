## 2026-04-06 - [Optimize God component rendering]
**Learning:** In the Data Processor web app (`src/data_processing/data_processor/web`), the root `App.tsx` component acts as a God component managing UI state (tabs) and large data structures. Unnecessary re-render cascades happen when switching UI tabs because heavy presentational child components (e.g., `SignalList`, `StatisticsPanel`) were not wrapped in `React.memo()`.
**Action:** When a God component manages state and large data, explicitly wrap heavy presentational child components in `React.memo()` to prevent UI stuttering and massive re-render cascades when unrelated parent state (like UI tabs) changes.

## 2026-04-06 - [Avoid chained array operations for large data arrays]
**Learning:** Chained array methods (like `.map().filter().map()`) or inline object creations (e.g. `({x, y: yData[i]})`) inside evaluation or mapping logic create numerous intermediate arrays and massive garbage collection overhead. In JS/TS regressions for 100k points, this causes 3s delays.
**Action:** When preparing data for regression or tight data loops in JS/TS, use a single-pass `for` loop, allocate standard arrays with maximum possible size initially (`new Array(len)`), and truncate them (`array.length = validCount`) at the end.
