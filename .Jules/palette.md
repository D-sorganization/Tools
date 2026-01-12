## 2024-05-22 - Missing Focus Indicators in Custom UIs
**Learning:** Even visually polished custom UIs (like calculator keypads) often completely miss keyboard focus indicators because standard browser outlines are suppressed or hidden by custom backgrounds.
**Action:** Always add explicit `*:focus-visible` styles using theme variables (like `--accent`) when auditing any custom UI component to ensure keyboard navigability is visible.
