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

## 2026-04-06 - [Visually Hidden File Inputs and Focus]
**Learning:** Components that use a click-forwarding container around a visually hidden (`sr-only`) `<input type="file">` become inaccessible to keyboard navigation because the hidden input receives focus but has no visible area to display a focus ring. This creates a "focus trap" where users tab, the focus seemingly disappears, and they don't know what is active.
**Action:** Always add `focus-within` styles (e.g., `focus-within:ring-2`) to the outer container of the hidden file input. This ensures that when the input natively receives keyboard focus, the parent container visually highlights, providing a clear indication to the user.

## 2026-04-13 - Contextless Raw Timers
**Learning:** Raw dynamic text like "03:45" or "105.4" provides no contextual meaning to screen readers if the visual context (like a pulsing red record dot or surrounding icon) is purely decorative or uses aria-hidden.
**Action:** When displaying timers, numeric readings, or dynamic status values, prepend a visually `<span className="sr-only">` label (e.g., `<span className="sr-only">Recording duration:</span>`) right before the raw value so assistive tech announces "Recording duration: 03:45" instead of just "zero three colon four five". Ensure purely decorative visual context like pulsing dots have `aria-hidden="true"`.

## 2026-04-14 - [Inline Error Feedback & Focus Improvements]
**Learning:** Native `alert()` dialogs used for errors (e.g., failed exports or imports) disrupt user flow, block interaction, and break from the app's visual theme. Furthermore, some interactive UI elements may lack keyboard focus rings.
**Action:** Replace `alert()` with inline UI elements (e.g., `<div role="alert">`) to provide non-blocking error feedback. Always ensure custom or unstyled buttons include `focus-visible` utility classes to support keyboard navigation.
## 2026-04-15 - Contextless Dynamic Text Readouts
**Learning:** Raw dynamic text like "03:45 / 10:00" in a video player or "3 detections" provides no contextual meaning to screen readers if the visual context is purely decorative or uses aria-hidden.
**Action:** When displaying timers, numeric readings, or dynamic status values, prepend a visually hidden `<span className="sr-only">` label (e.g., `<span className="sr-only">Current time:</span>`) right before the raw value so assistive tech announces "Current time: 03:45" instead of just the numbers. Ensure purely decorative visual context like pulsing dots have `aria-hidden="true"`.

## 2026-04-20 - [Dynamic ARIA Labels for Confirmation Buttons]
**Learning:** When implementing multi-step or confirming buttons (like a 'Clear' button that temporarily changes to 'Confirm?'), the visual text change is not always announced by screen readers if focus remains on the button, leaving users unaware of the new required action.
**Action:** Always dynamically update the element's `aria-label` to reflect the new state (e.g., "Confirm clear history"), and ensure a relevant parent container uses `aria-live='polite'` so the state change is announced smoothly.

## 2026-04-28 - [Accessible Soft Confirm Pattern]
**Learning:** Native `confirm()` dialogs block the main thread and provide poor user experience, especially for accessibility where focus management can be lost upon return.
**Action:** When implementing destructive actions, use an inline "soft confirm" pattern (requiring a second click on the same button with updated text/aria-label) combined with `aria-live` on the container to ensure smooth flow and screen reader awareness.

## 2026-04-25 - [Dynamic Focus for Inline Validation]
**Learning:** When inline form validation fails, relying solely on visual error states or `aria-invalid` attributes is insufficient. If keyboard focus remains unchanged, screen reader and keyboard users are not directed to the error, creating a confusing experience where the form submission seemingly does nothing.
**Action:** When implementing inline form validation, always dynamically shift focus to the first invalid field using `.focus()` in addition to setting `aria-invalid="true"`. This prevents keyboard focus traps and ensures assistive technologies immediately announce the context of the error.

## 2026-04-26 - [Dynamic Accessible Name for Soft Confirm Pattern]
**Learning:** When using the soft confirm pattern on destructive actions (e.g., clear buttons changing to "CONFIRM?"), merely changing the text content (`textContent`) and providing `aria-live="polite"` on the parent does not update the accessible name of the button itself, leaving screen reader users without explicit context for the required secondary action.
**Action:** Always dynamically manipulate the `aria-label` attribute (e.g., changing from "Clear all fields" to "Confirm clear all fields") concurrently with the visual text change, restoring it appropriately when the timeout occurs or the action completes.

## 2026-04-28 - [Combobox Label Retargeting]
**Learning:** When replacing native `<select>` elements with custom comboboxes (search inputs with dropdowns), leaving the `<label for="...">` targeted at the hidden select element breaks click-to-focus behavior and fails to provide the newly visible search input with its required accessible name.
**Action:** Always dynamically update the `for` attribute of the corresponding `<label>` to target the newly visible interactive input field's `id` (and restore it if swapping back) to maintain accessibility and user flow.
