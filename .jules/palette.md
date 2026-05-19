## 2026-05-18 - Missing label associations for range inputs
**Learning:** Found multiple range inputs with visual labels that lacked explicit connection via htmlFor and id. This creates a confusing experience for screen reader users as they encounter standalone inputs without proper context.
**Action:** Always ensure <label> tags use htmlFor attributes mapped to the corresponding input's id, especially for complex range sliders where context is critical.
