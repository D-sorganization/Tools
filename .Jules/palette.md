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

## 2026-02-18 - [Interactive Lists Accessibility]
**Learning:** List items that function as buttons (like "Recent Conversions") implemented as `<div>`s with click handlers are invisible to screen readers and keyboard users.
**Action:** Always use `<button>` elements for interactive list items, ensuring `width: 100%` and `text-align: left` in CSS to maintain the list appearance while gaining native accessibility benefits (keyboard focus, role activation).

## 2026-03-30 - [Hover Overlays Keyboard Accessibility]
**Learning:** Container elements that rely on `opacity-0 hover:opacity-100` to reveal interactive children (like a play button over a video) make those children completely invisible to keyboard users who tab to them, as `focus` does not trigger `hover`.
**Action:** Always add `focus-within:opacity-100` to such container elements to ensure the interactive elements become visible when they receive keyboard focus.

## 2026-04-05 - Expandable Accordion Accessibility
**Learning:** React state-driven toggle components that lack `aria-expanded` and `aria-controls` fail to announce their state to screen readers, making accordions effectively invisible to non-visual users. Furthermore, relying entirely on hover states without `focus-visible` styles makes them unusable via keyboard navigation.
**Action:** When implementing expandable content sections, always pair the trigger button with `aria-expanded={isExpanded}` and `aria-controls={contentId}`, while ensuring the content block has the corresponding `id={contentId}`. Provide explicit visual focus indicators (`focus-visible:ring-2`) for all interactive elements.
