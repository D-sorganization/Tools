## 2026-04-13 - [App Component Global State Causes Re-renders]
**Learning:** In the Data Processor UI, `App.tsx` acts as a God component managing UI state (tabs) and processing state. Changes to local UI state (like `leftPanelTab`) trigger re-renders of heavy list/table components (`SignalList`, `StatisticsPanel`) unless they are explicitly wrapped in `React.memo()`.
**Action:** When adding or modifying heavy components that render array data (like signals or stats) within `App.tsx`, always wrap them in `React.memo` to prevent global re-render cascades.
