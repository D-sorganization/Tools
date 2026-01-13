## 2026-01-13 - [Unit Converter Modal Validation]
**Learning:** Using `alert()` for form validation in modals disrupts flow and feels archaic.
**Action:** Replace `alert()` with inline error messages (reusing existing `.error-message` styles) and visual input validation states (red borders) for a smoother experience.

## 2026-01-13 - [Calculator Clear Button Safety]
**Learning:** Destructive actions like "Clear" on touch interfaces are prone to accidental activation.
**Action:** Implemented a "soft confirm" pattern where the first click changes the button text to "CONFIRM?" and a second click is required to execute the action. This prevents data loss without the intrusion of a modal dialog. The state is managed locally in the component.

## 2026-01-12 - [Calculator Focus Indicators]
**Learning:** The calculator web app was missing `*:focus-visible` styles, making keyboard navigation difficult. The dark theme background with light interactive elements requires a specific focus color (`var(--accent)`) to ensure visibility on both dark shell and light display areas.
**Action:** Always verify `*:focus-visible` is present in global CSS for web apps, especially those with custom themes. Test with keyboard navigation (Tab) to ensure the focus ring contrasts well with all backgrounds.

## 2024-05-23 - [Calculator Loading State]
**Learning:** For vanilla JS apps without reactive frameworks, loading states must be manually managed in a `finally` block to ensure the UI is reset even on error.
**Action:** When adding async interactions, always wrap the state toggle in a try/finally block and verify error paths restore the UI.

## 2024-05-23 - [Form Validation Feedback]
**Learning:** Using `:invalid` pseudo-class is powerful but can be too aggressive if shown immediately.
**Action:** Consider combining `:invalid` with `:not(:placeholder-shown)` or a dirty class to show errors only after user interaction.

## 2024-05-22 - Missing Focus Indicators in Custom UIs
**Learning:** Even visually polished custom UIs (like calculator keypads) often completely miss keyboard focus indicators because standard browser outlines are suppressed or hidden by custom backgrounds.
**Action:** Always add explicit `*:focus-visible` styles using theme variables (like `--accent`) when auditing any custom UI component to ensure keyboard navigability is visible.

## 2026-02-05 - [Modal Feedback Patterns]
**Learning:** Native `alert()` dialogs in web apps are disruptive and cannot be styled to match the theme.
**Action:** Replace `alert()` with inline message containers (e.g., `<div role="alert">`) within modals. Use utility classes (like `.success`, `.error`) to style them according to the app's theme.
