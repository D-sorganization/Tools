## 2024-05-23 - Interactive Element Hover States
**Learning:** Interactive elements like copy buttons, mode buttons, action buttons, and regular keys in this calculator app are missing hover state styling. Only `.soft-key` currently has a defined hover style.
**Action:** Always verify consistent hover state styling on all interactive elements across the application to ensure visual feedback when hovered.
## 2024-05-23 - Interactive Element Hover States
**Learning:** Interactive elements like copy buttons, mode buttons, action buttons, and regular keys in this calculator app are missing hover state styling. Only `.soft-key` currently has a defined hover style.
**Action:** Always verify consistent hover state styling on all interactive elements across the application to ensure visual feedback when hovered.
## 2024-05-28 - Add accessible roles to mapped tab components
**Learning:** When generating tabs via a map loop in React, dynamic components often lack essential accessibility structure, leaving screen reader users without proper context (e.g. knowing it's a tab, which is selected).
**Action:** Always ensure any tab list mapped dynamically explicitly receives `role="tablist"` on the wrapper, and `role="tab"`, `aria-selected`, `aria-controls`, and an `id` on each button. Wrap corresponding sections with `role="tabpanel"`.
