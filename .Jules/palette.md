## 2026-01-12 - [Calculator Focus Indicators]
**Learning:** The calculator web app was missing `*:focus-visible` styles, making keyboard navigation difficult. The dark theme background with light interactive elements requires a specific focus color (`var(--accent)`) to ensure visibility on both dark shell and light display areas.
**Action:** Always verify `*:focus-visible` is present in global CSS for web apps, especially those with custom themes. Test with keyboard navigation (Tab) to ensure the focus ring contrasts well with all backgrounds.

## 2024-05-22 - Missing Focus Indicators in Custom UIs
**Learning:** Even visually polished custom UIs (like calculator keypads) often completely miss keyboard focus indicators because standard browser outlines are suppressed or hidden by custom backgrounds.
**Action:** Always add explicit `*:focus-visible` styles using theme variables (like `--accent`) when auditing any custom UI component to ensure keyboard navigability is visible.
