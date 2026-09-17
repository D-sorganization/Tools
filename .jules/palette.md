## $(date +%Y-%m-%d) - Consistent Keyboard Focus Indicators
**Learning:** Found inconsistent and missing focus states for keyboard users across form elements and custom buttons in `FunctionGenerator.tsx`.
**Action:** Used `focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[color]` utilities across inputs, selects, and buttons to provide clear, consistent focus rings for keyboard navigation while avoiding them for mouse clicks.
