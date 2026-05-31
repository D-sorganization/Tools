## 2024-05-23 - Interactive Element Hover States
**Learning:** Interactive elements like copy buttons, mode buttons, action buttons, and regular keys in this calculator app are missing hover state styling. Only `.soft-key` currently has a defined hover style.
**Action:** Always verify consistent hover state styling on all interactive elements across the application to ensure visual feedback when hovered.
## 2024-05-23 - Interactive Element Hover States
**Learning:** Interactive elements like copy buttons, mode buttons, action buttons, and regular keys in this calculator app are missing hover state styling. Only `.soft-key` currently has a defined hover style.
**Action:** Always verify consistent hover state styling on all interactive elements across the application to ensure visual feedback when hovered.
## 2024-05-28 - Add accessible roles to mapped tab components
**Learning:** When generating tabs via a map loop in React, dynamic components often lack essential accessibility structure, leaving screen reader users without proper context (e.g. knowing it's a tab, which is selected).
**Action:** Always ensure any tab list mapped dynamically explicitly receives `role="tablist"` on the wrapper, and `role="tab"`, `aria-selected`, `aria-controls`, and an `id` on each button. Wrap corresponding sections with `role="tabpanel"`.
## 2026-05-31 - Link form labels to inputs dynamically
**Learning:** Using React's `useId()` to dynamically link `<label htmlFor>` with `<input id>` ensures proper semantic accessibility mapping without the risk of duplicate hard-coded IDs if a component is used multiple times.
**Action:** When adding labels to complex form structures in React applications, utilize `useId()` to automatically coordinate standard accessible connections.
## 2026-05-31 - aria-pressed on Custom Toggles
**Learning:** When using custom toggle buttons in React that rely on dynamic styling (`className` or inline `style`) rather than standard HTML checkbox/radio inputs, screen readers have no way of knowing the button's active state.
**Action:** Always pair custom state styling (like `.active` classes or dynamic background colors) with a dynamic `aria-pressed={isActive}` attribute on the `<button>` element to ensure the semantic state is announced to assistive technologies.
