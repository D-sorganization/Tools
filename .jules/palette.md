## 2024-05-23 - Interactive Element Hover States
**Learning:** Interactive elements like copy buttons, mode buttons, action buttons, and regular keys in this calculator app are missing hover state styling. Only `.soft-key` currently has a defined hover style.
**Action:** Always verify consistent hover state styling on all interactive elements across the application to ensure visual feedback when hovered.
## 2024-05-24 - Math Form Inputs
**Learning:** Text inputs expecting mathematical expressions or variables can trigger native mobile OS autocorrect, auto-capitalization, and spellchecking features, which creates a frustrating user experience when typing variable names like `x` or functions like `sin`.
**Action:** Always add `spellcheck="false"`, `autocorrect="off"`, and `autocapitalize="none"` to any text input that expects math, formulas, or raw code.
