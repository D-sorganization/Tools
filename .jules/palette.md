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
## 2024-05-31 - Tab Roles vs aria-pressed
**Learning:** When adding `role="tab"` to buttons to form an accessible tab list, existing `aria-pressed` attributes must be removed, as `aria-pressed` is intended for toggle buttons and creates conflicting semantics on tab elements. Furthermore, dynamically updating `aria-controls` with missing IDs creates broken reference validations.
**Action:** When implementing tab semantics, ensure `role="tab"` is paired strictly with `aria-selected` (not `aria-pressed`), and verify `aria-controls` points to a statically identifiable and valid `role="tabpanel"` container whose `aria-labelledby` points back to the selected tab.
## 2024-06-05 - Preserving existing aria-describedby for dynamic validation
**Learning:** When dynamically associating a validation error message with an input field using `aria-describedby`, setting the attribute blindly will overwrite any existing helpful associations (like an input hint ID).
**Action:** Always concatenate the error message ID with any existing hint IDs (e.g., `aria-describedby="errorId hintId"`) to ensure screen readers announce both the validation error and the original helper text. When clearing the error, restore the `aria-describedby` to just the hint ID.
## 2024-06-11 - Dynamic Input ID Generation for Screen Readers
**Learning:** Hardcoded IDs in reusable React components (like `InputField` in `FinancialCalculator.tsx`) lead to ID collisions when rendered multiple times. This breaks `<label htmlFor="...">` associations, severely impacting screen reader accessibility as the label will only associate with the first instance of the ID on the page.
**Action:** Always use React's `useId()` hook within reusable components to dynamically generate unique, accessible, and SSR-safe IDs for `htmlFor` and `id` bindings.
## 2024-06-25 - Custom Toggle Controls
**Learning:** When using `<button>` elements to create custom ON/OFF toggles or mode switches (like the "Add" / "Subtract" operations) that rely entirely on background color changes for state, screen readers cannot determine their current state.
**Action:** Always add `aria-pressed={isActive}` to custom toggle buttons, along with keyboard-accessible hover (`hover:bg-...`) and focus states (`focus-visible:ring-2`) to ensure both semantic state and visual focus are clear.

## 2024-05-24 - Form Submissions for Keyboard Users
**Learning:** Wrapping a collection of data entry fields in a native `<form>` element instead of a `<div>` instantly enables native "Enter" key form submission, which is critical for heavy data-entry applications. Users don't need to manually tab to the calculate button.
**Action:** For all data-entry calculators or settings panels, always use a `<form onSubmit={...}>` structure instead of binding `onClick` directly to the submit button.
## 2024-05-31 - Tab Roles vs aria-pressed
**Learning:** When implementing custom tab components in React, use `role="tab"` paired strictly with `aria-selected` (not `aria-pressed`, which is intended for toggle buttons) and ensure `aria-controls` points to a valid `role="tabpanel"` container whose `aria-labelledby` points back to the tab.
**Action:** When adding accessible properties to custom tabs, replace `aria-pressed` with `aria-selected`, ensure a `role="tablist"` wrapper is present, and correctly cross-reference `aria-controls` with the tab panel IDs.
## 2024-05-24 - Semantic State for Custom Toggle Buttons
**Learning:** In `SignalList.tsx`, custom toggle buttons used dynamic CSS classes to indicate their active state, but lacked ARIA attributes, making their state invisible to screen readers.
**Action:** When implementing or modifying custom toggle buttons that rely on dynamic styling, always include `aria-pressed={isActive}` on the `<button>` element to ensure semantic state visibility for assistive technologies.

## 2024-07-23 - Focus Rings on Core Buttons
**Learning:** Discovered a pattern where standard buttons utilizing the `.btn` utility class lacked focus ring accessibility, despite inputs and tabs being styled properly.
**Action:** Ensure global utility classes for interactive elements inherently contain standard `focus-visible` styling to enforce a baseline accessibility standard across the application.

## 2024-07-23 - Focus Rings on Core Buttons
**Learning:** Discovered a pattern where standard buttons utilizing the `.btn` utility class lacked focus ring accessibility, despite inputs and tabs being styled properly.
**Action:** Ensure global utility classes for interactive elements inherently contain standard `focus-visible` styling to enforce a baseline accessibility standard across the application.
## 2024-07-30 - Added focus-visible states to Rotation Converter
**Learning:** Found an accessibility issue pattern where inputs and buttons in Rotation Converter have `outline-none` but lack focus indicators, making keyboard navigation difficult.
**Action:** Replace `outline-none` with `focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500` to preserve keyboard accessibility while styling interactive elements.
## 2024-05-31 - Tab Roles vs aria-pressed
**Learning:** When implementing custom tab components in React, use `role="tab"` paired strictly with `aria-selected` (not `aria-pressed`, which is intended for toggle buttons) and ensure `aria-controls` points to a valid `role="tabpanel"` container whose `aria-labelledby` points back to the tab.
**Action:** When adding accessible properties to custom tabs, replace `aria-pressed` with `aria-selected`, ensure a `role="tablist"` wrapper is present, and correctly cross-reference `aria-controls` with the tab panel IDs.
