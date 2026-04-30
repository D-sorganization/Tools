# Phase 2.2 - Mobile Testing Checklist

## Overview
This document provides comprehensive testing procedures for Phase 2.2 Frontend Polish & Integration across three web applications:
1. **Aurora CAS Calculator** - `/calculator`
2. **Unit Converter** - `/unit_converter/unit-converter-app`
3. **URDF Viewer** - `/urdf_viewer`

All tests must pass on the specified viewport sizes using actual devices or browser DevTools mobile simulation.

---

## Test Environments

### Viewport Sizes
- **375px** (iPhone SE, 5, 5S) - Mobile Small
- **768px** (iPad Mini) - Tablet Portrait
- **1024px** (iPad Pro) - Tablet Landscape
- **1920px** (Desktop) - Full screen reference

### Devices for Real Hardware Testing
- iPhone or Android phone (375-414px width)
- iPad or Android tablet (768px+)
- Desktop browser (1920px+)

### Browser DevTools Setup
```
Chrome/Edge: Toggle device toolbar (Ctrl+Shift+M / Cmd+Shift+M)
Firefox: Responsive Design Mode (Ctrl+Shift+M / Cmd+Shift+M)
Safari: Develop → Enter Responsive Design Mode
```

---

## 1. RESPONSIVE LAYOUT TESTS (375px - Mobile)

### 1.1 Aurora CAS Calculator

#### Layout Integrity
- [ ] Status bar: Brand, mode, battery all visible and readable
- [ ] Main screen (history + I/O grid) renders without horizontal scroll
- [ ] Input fields stack vertically (1 column layout)
- [ ] "Bounds row" inputs stack vertically (3 single-column inputs)
- [ ] Result display section doesn't overflow
- [ ] Copy buttons are side-by-side and tappable (≥44px height)
- [ ] Touch edit panel visible and functional

#### Keypad & Buttons
- [ ] Keypad buttons are ≥44px × 44px (minimum touch target)
- [ ] Button text is readable (no truncation)
- [ ] CLEAR and ENTER buttons are full-width and easily tappable
- [ ] Mode buttons (CAS, Algebra, etc.) don't wrap or overflow
- [ ] Function strip buttons don't wrap or overlap

#### Spacing & Padding
- [ ] No elements touching screen edges (≥8px padding)
- [ ] Adequate spacing between interactive elements (≥8px)
- [ ] Vertical rhythm maintained (consistent spacing)
- [ ] No cramped or overlapping text

#### Visual Clarity
- [ ] Text is readable (≥12px font size for inputs)
- [ ] Contrast ratios meet WCAG AA (4.5:1 for text)
- [ ] Screen area is green, keypad is dark blue
- [ ] No visual clipping or hidden content

#### Scrolling Behavior
- [ ] Vertical scrolling works smoothly
- [ ] No horizontal scrolling needed
- [ ] Content remains accessible while scrolling
- [ ] Scroll doesn't trigger unwanted page-level scrolling

---

### 1.2 Unit Converter

#### Layout Integrity
- [ ] Header with title and action buttons visible
- [ ] Category selector dropdown spans full width and opens without overflow
- [ ] From/To unit selectors stack vertically
- [ ] Input fields are full-width or nearly full-width
- [ ] Result display is readable and doesn't overflow
- [ ] No horizontal scroll

#### Input Fields
- [ ] Input fields are ≥44px tall
- [ ] Keyboard-friendly (numbers trigger numeric keyboard on mobile)
- [ ] Placeholder text is visible
- [ ] Focus state is clear and visible

#### Buttons & Controls
- [ ] Custom units button is tappable (≥44px)
- [ ] Theme toggle button is tappable (≥44px)
- [ ] All buttons have adequate spacing (≥8px)

#### Visual Clarity
- [ ] Text is readable at small sizes
- [ ] Color contrast is sufficient
- [ ] Icons are clear and appropriately sized
- [ ] Category labels are properly displayed

---

### 1.3 URDF Viewer

#### Layout Integrity
- [ ] 3D viewport takes appropriate space (not too small)
- [ ] File upload area is accessible and readable
- [ ] Controls are accessible and tappable
- [ ] Sidebar (if present) doesn't overlap content
- [ ] No horizontal scroll

#### Viewport & 3D Content
- [ ] 3D canvas resizes with screen
- [ ] Models render without distortion
- [ ] Interaction controls (zoom, pan, rotate) work on touch

---

## 2. TAB ORDER & KEYBOARD NAVIGATION (All Viewports)

### 2.1 Aurora CAS Calculator

#### Tab Order Check (375px)
Open DevTools console and run:
```javascript
// Log all focusable elements in order
document.querySelectorAll(
  'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
).forEach((el, i) => console.log(i, el.getAttribute('aria-label') || el.textContent?.slice(0,30) || el.type));
```

- [ ] Tab order is logical and intuitive
- [ ] Start: Expression input (focused first)
- [ ] Navigate through all inputs sequentially
- [ ] Then through buttons (mode buttons, function strip)
- [ ] Then through keypad (left-to-right, top-to-bottom)
- [ ] End: CLEAR and ENTER buttons

#### Keyboard Shortcuts
- [ ] `Enter` key submits calculation
- [ ] `Escape` clears form (if implemented)
- [ ] Arrow keys work in touch panel (if applicable)
- [ ] Number keys can trigger keypad buttons (if implemented)

#### Focus Indicators
- [ ] Every interactive element has visible focus indicator
- [ ] Focus outline is ≥2px wide
- [ ] Focus outline color contrasts with background
- [ ] Focus indicator doesn't hide content

---

### 2.2 Unit Converter

#### Tab Order Check
- [ ] Category dropdown is first focusable element
- [ ] From unit dropdown is next
- [ ] To unit dropdown follows
- [ ] Input fields are in logical order
- [ ] Action buttons (custom units, theme) are reachable via tab

#### Keyboard Functionality
- [ ] `Tab` and `Shift+Tab` navigate all controls
- [ ] `Enter` triggers conversions/actions
- [ ] Arrow keys work in dropdowns
- [ ] No keyboard traps

---

### 2.3 URDF Viewer

#### Tab Order Check
- [ ] File upload element is focusable
- [ ] All control buttons are reachable
- [ ] Model tree (if present) is keyboard navigable

---

## 3. TOUCH TARGET SIZE TESTS (All Viewports)

Use browser DevTools to inspect button dimensions:

### Minimum Requirements
- [ ] All buttons: ≥44px × 44px (Apple) or ≥48px × 48px (Material)
- [ ] Input fields: ≥44px tall
- [ ] Spacing between targets: ≥8px minimum

### Test Procedure
1. Open DevTools Inspector
2. Right-click → Inspect on each button/input
3. Check computed dimensions in Styles pane
4. Verify against checklist

### Buttons to Check
**Calculator:**
- [ ] Keypad number buttons (0-9)
- [ ] Operation buttons (+, -, ×, ÷)
- [ ] Function buttons (sin, cos, tan, etc.)
- [ ] Mode buttons (CAS, Algebra, Systems, etc.)
- [ ] CLEAR and ENTER buttons
- [ ] Touch edit controls

**Unit Converter:**
- [ ] Category select (≥44px height)
- [ ] Unit selects (≥44px height)
- [ ] Custom units button
- [ ] Theme toggle

**URDF Viewer:**
- [ ] Upload button/area
- [ ] Control buttons

---

## 4. ACCESSIBILITY TESTS

### 4.1 Screen Reader Testing (ARIA & Semantic HTML)

#### Aurora Calculator
```
Expected Screen Reader Output:
- "Aurora CAS Calculator, application"
- "Status bar, battery level"
- "Main landmark"
- "Expression, required, edit text"
- "Variable, edit text"
- "Order, spinner, 1 to infinity"
- "Bounds row, group"
  - "Lower bound, edit text"
  - "Upper bound, edit text"
  - "Limit value, edit text"
- "Result display, region, aria-live polite"
  - "Result, heading"
  - "Ready. (or calculation output)"
  - "Approximate value (if visible)"
  - "Copy result, button"
  - "Copy input, button"
- "Touch edit, region"
- "Function strip, region"
  - "Evaluate, button"
  - "Simplify, button"
  - [etc.]
- "Mode strip, region"
- "Keypad, region"
  - All buttons with proper aria-labels
- "Action row, region"
```

#### Unit Converter
```
Expected Output:
- "Unit Converter, application"
- "Header"
  - "Unit Converter, heading"
  - "Custom units, button"
  - "Toggle dark mode, button"
- "Main content"
  - "Category, combobox"
  - "From unit, combobox"
  - "To unit, combobox"
  - "Input, edit text, number"
  - "Result (display), output"
```

#### URDF Viewer
```
Expected Output:
- "URDF Viewer, application"
- [File upload controls]
- [3D viewport with description]
```

### 4.2 Aria-Labels Verification

**Tools:** Use axe DevTools Chrome extension

Run automated scan:
1. Open page in browser
2. Open DevTools → axe DevTools
3. Scan page
4. Check results:
   - [ ] No violations (critical)
   - [ ] No missing aria-labels on icon buttons
   - [ ] No empty button text
   - [ ] Proper heading hierarchy (h1 > h2 > h3, no gaps)

### 4.3 Manual Keyboard Navigation Test

#### Procedure (using keyboard only)
1. [ ] Load page
2. [ ] `Tab` through all controls
3. [ ] `Shift+Tab` back through all controls
4. [ ] Verify every interactive element is reachable
5. [ ] Check no keyboard traps
6. [ ] Verify focus is always visible

#### Expected Behavior
- [ ] Start with first input/button focused
- [ ] Tab moves forward logically
- [ ] All controls can be activated via keyboard
- [ ] No elements stuck as unreachable
- [ ] Focus indicator always visible

---

## 5. ERROR HANDLING TESTS

### 5.1 Toast Component Integration

#### Setup
Verify Toast component exists at:
- Path: `/src/web_applications/[app]/static/toast.js` or similar
- HTML: Toast container in index.html with ID `toast-container` or similar

#### Test Cases

##### 5.1.1 Invalid Input Errors
**Calculator:**
- [ ] Enter invalid expression: `((1+2)` (unmatched parentheses)
- [ ] Expected: Error toast appears saying "Unmatched parentheses"
- [ ] Toast auto-dismisses after 5 seconds OR has close button
- [ ] Toast doesn't block other UI

- [ ] Enter invalid variable: `123invalid` (starts with number)
- [ ] Expected: Error toast saying "Invalid variable name"

- [ ] Leave expression empty, click ENTER
- [ ] Expected: Error toast saying "Expression required"

**Unit Converter:**
- [ ] Enter non-numeric value in input field
- [ ] Expected: Error toast: "Please enter a valid number"
- [ ] Invalid category selection (if possible)
- [ ] Expected: Appropriate error message

**URDF Viewer:**
- [ ] Upload non-URDF file (e.g., .txt)
- [ ] Expected: Error toast: "Invalid URDF file format"

##### 5.1.2 Validation Errors
- [ ] Out-of-range numeric input
- [ ] Expected: Error toast with specific bounds or constraint message

- [ ] Filter with invalid parameters
- [ ] Expected: Error toast explaining valid parameter format

##### 5.1.3 Network Errors (if applicable)
- [ ] Disable network (DevTools → Network throttling → Offline)
- [ ] Try to perform action requiring backend
- [ ] Expected: Error toast: "Network error. Please check your connection."
- [ ] Retry UI appears (button or retry option)
- [ ] Re-enable network and retry succeeds

### 5.2 Toast Styling & UX
- [ ] Toast is visible against background (good contrast)
- [ ] Toast text is readable (≥14px, sufficient color contrast)
- [ ] Toast position doesn't obscure critical content
- [ ] Multiple toasts stack appropriately (no overlap)
- [ ] Toast animation is smooth (≤300ms)
- [ ] Close button (if present) is accessible (≥44px)

### 5.3 Error Recovery
- [ ] After error toast, input field(s) retain user data
- [ ] User can correct input and retry without re-entering everything
- [ ] Form doesn't auto-reset on error

---

## 6. FOCUS-VISIBLE STYLES TEST

### Procedure
1. Open DevTools on desktop
2. Use keyboard to tab through page
3. No mouse movement

### Test Cases
- [ ] Every button has a visible focus indicator
- [ ] Every input field has a visible focus indicator
- [ ] Outline color: cyan/blue (#8bd3f7 for calculator)
- [ ] Outline width: ≥2px
- [ ] Outline offset: ≥2px (doesn't overlap content)
- [ ] Focus state is clear and easy to see

### Missing Focus Styles (if found)
Log any elements without focus indicator:
- [ ] Mode buttons
- [ ] Function strip buttons
- [ ] Keypad buttons
- [ ] Any custom interactive elements

---

## 7. CHART SCALING & RESIZE OBSERVER TESTS

*Only applicable if charts/graphs are present*

### Procedure
1. Add a chart to the page (if not present)
2. Open page with chart at:
   - [ ] 375px width
   - [ ] 768px width
   - [ ] 1024px width

### Expected Behavior
- [ ] Chart scales proportionally to viewport
- [ ] Chart title is readable
- [ ] Chart axes are labeled and readable
- [ ] Legend doesn't overlap chart
- [ ] No chart overflow or clipping

### ResizeObserver Implementation Check
In DevTools Console:
```javascript
// Check if ResizeObserver is initialized
window.resizeObserver ? console.log('ResizeObserver active') : console.log('No ResizeObserver');
```

---

## 8. TEST RESULTS DOCUMENTATION

### Template for Each Test Session

```
TEST SESSION: [App Name] - [Date] - [Viewport]

Device: [Real Device / Browser Simulator]
Browser: [Chrome/Firefox/Safari/Edge]
Viewport: [375px / 768px / 1024px / 1920px]

### Layout Tests
- Status: PASS / FAIL
- Notes: [Any visual issues, layout problems]

### Tab Order Tests
- Status: PASS / FAIL
- Notes: [Tab sequence issues, if any]

### Touch Target Tests
- Status: PASS / FAIL
- Failed targets: [List any buttons/inputs <44px]

### Accessibility Tests
- Axe DevTools violations: [Number]
- Critical issues: [List any]
- Screen reader test: PASS / FAIL

### Keyboard Navigation
- Status: PASS / FAIL
- Traps found: [Any]

### Error Handling
- Toast component: [MISSING / WORKING / PARTIAL]
- Test results: [Details]

### Screenshots
- [Attach screenshot at each viewport]

Overall Status: PASS / FAIL
Blockers: [If any]
```

---

## 9. VISUAL REGRESSION TESTING

### Screenshot Checklist

Capture screenshots at each viewport for:
1. **Calculator** @ 375px, 768px, 1024px
   - [ ] Default state with "Ready."
   - [ ] After calculation (with history)
   - [ ] With error toast
   - [ ] With focus on first input

2. **Unit Converter** @ 375px, 768px, 1024px
   - [ ] Default state
   - [ ] With category expanded
   - [ ] Dark mode toggled
   - [ ] Custom units modal (if present)

3. **URDF Viewer** @ 375px, 768px, 1024px
   - [ ] Upload area
   - [ ] Model loaded and rendered
   - [ ] Mobile rotation/zoom in action

### Storage
Save screenshots to: `/test_results/screenshots/[date]/`

### Comparison
- [ ] No layout shifts from previous session
- [ ] No color/contrast changes
- [ ] Typography unchanged
- [ ] Interactive elements styling consistent

---

## 10. AUTOMATED TESTING SETUP

### Unit Tests (JavaScript)
Location: `/tests/*.test.js`

Run tests:
```bash
npm test 2>/dev/null || yarn test 2>/dev/null || echo "No automated tests configured"
```

Tests should cover:
- [ ] Touch target sizes
- [ ] Aria attributes
- [ ] Focus management
- [ ] Error handling

### Accessibility Audit (axe-core)
```bash
# If axe-core is installed
npx axe-core [url] --standard WCAG2AA
```

Expected: 0 violations at WCAG2AA level

---

## 11. MOBILE DEVICE CHECKLIST

### Setup on Real Device

#### iPhone / iOS
1. [ ] Connect to WiFi
2. [ ] Open Safari
3. [ ] Navigate to dev server or deployed URL
4. [ ] Home → Swipe up → Verify responsive layout
5. [ ] Rotate device → Check responsive layout update
6. [ ] Test all inputs with mobile keyboard
7. [ ] Test all buttons with thumb taps
8. [ ] Test zoom behavior (should be limited or disabled)

#### Android Device
1. [ ] Connect to WiFi
2. [ ] Open Chrome or Firefox
3. [ ] Navigate to dev server or deployed URL
4. [ ] Verify responsive layout in portrait
5. [ ] Rotate to landscape → Check layout
6. [ ] Test inputs with mobile keyboard
7. [ ] Test buttons with finger taps
8. [ ] Back button doesn't interfere with app

### Touch Gesture Tests
- [ ] Tap selects/activates element (no double-tap needed)
- [ ] Double-tap doesn't zoom unexpectedly
- [ ] Swipe doesn't trigger back navigation
- [ ] Long-press doesn't show browser menu
- [ ] Scroll is smooth (no jank or stuttering)

---

## 12. SIGN-OFF

### For QA / Developer

```
Tested by: [Name]
Date: [YYYY-MM-DD]
Applications: [All / Calculator / Unit Converter / URDF Viewer]
Viewports tested: [375px / 768px / 1024px / 1920px]
Devices used: [Device list]

Overall Result: [PASS / FAIL]
Blockers: [None / List]
Notes: [Any additional findings]

Sign-off: ___________________
```

---

## 13. COMMON ISSUES & FIXES

### Issue: Buttons too small on mobile
**Fix:** Increase button padding and min-height to 44px, adjust font size if needed

### Issue: Text overflows input fields
**Fix:** Add `max-width: 100%` to input containers, use `overflow-wrap: break-word`

### Issue: Tab order jumps around
**Fix:** Remove unnecessary `tabindex` attributes, rely on DOM order

### Issue: Focus indicator invisible
**Fix:** Add `:focus-visible { outline: 2px solid #8bd3f7; outline-offset: 2px; }`

### Issue: Modal/Toast blocks input
**Fix:** Ensure z-index is high enough, don't trap focus

### Issue: 3D viewport doesn't resize
**Fix:** Implement ResizeObserver on canvas element, call render() on resize

---

## References

- [WCAG 2.1 Level AA](https://www.w3.org/WAI/WCAG21/quickref/)
- [ARIA Practices Guide](https://www.w3.org/WAI/ARIA/apg/)
- [Touch Target Size Guidelines](https://www.smashingmagazine.com/2022/09/inline-links-touch-targets-web-design-ux/)
- [Focus Visible Spec](https://drafts.csswg.org/selectors-4/#the-focus-visible-pseudo-class)
- [Responsive Web Design Best Practices](https://www.nngroup.com/articles/mobile-usability/)

---

End of Mobile Testing Checklist. See TESTING_GUIDE.md for integration testing procedures.
