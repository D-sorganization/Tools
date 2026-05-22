## 2026-05-18 - Missing label associations for range inputs
**Learning:** Found multiple range inputs with visual labels that lacked explicit connection via htmlFor and id. This creates a confusing experience for screen reader users as they encounter standalone inputs without proper context.
**Action:** Always ensure <label> tags use htmlFor attributes mapped to the corresponding input's id, especially for complex range sliders where context is critical.

## 2026-05-20 - Toggle Button Accessibility
**Learning:** Custom toggle buttons implemented via class changes (e.g., `.active`) lack semantic state for screen readers, hiding the active mode from assistive technologies.
**Action:** Always pair visual active class toggles with `aria-pressed="true|false"` dynamic attribute updates.
## 2024-05-30 - Accessible Reusable Form Controls
**Learning:** When creating reusable form controls (like unit selectors or dropdowns) that require linking `<label>` elements to inputs using `id` and `htmlFor`, hardcoded IDs can cause collisions if the component is rendered multiple times. Using `React.useId()` generates unique IDs per component instance, ensuring screen readers can correctly associate the label with the input without collisions.
**Action:** Always use `React.useId()` for `id` and `htmlFor` attributes when creating reusable React components that involve form labels.
