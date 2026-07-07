## 2024-05-24 - Semantic State for Custom Toggle Buttons
**Learning:** In `SignalList.tsx`, custom toggle buttons used dynamic CSS classes to indicate their active state, but lacked ARIA attributes, making their state invisible to screen readers.
**Action:** When implementing or modifying custom toggle buttons that rely on dynamic styling, always include `aria-pressed={isActive}` on the `<button>` element to ensure semantic state visibility for assistive technologies.
