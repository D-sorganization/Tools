# Phase 2.1 Accessibility Testing Plan

## Pre-Testing Setup

### Environment

- **Browser:** Chrome 125+ (primary), Firefox, Safari
- **OS:** Windows 10+, macOS 12+
- **Screen reader:** NVDA (Windows) or VoiceOver (Mac) - optional for Phase 2.1
- **Testing mode:** Keyboard-only (no mouse)

### Start the App

```bash
cd /home/user/Tools/src/data_processing/data_processor/web
npm install  # If needed
npm run dev
# App runs at http://localhost:5173 (Vite default)
```

---

## Test 1: Keyboard Navigation (Full App)

### Objective

Verify all interactive elements are reachable via Tab key and in logical order.

### Test Steps

1. **Load the app**

   - Press Tab repeatedly from page load
   - Should focus header first
   - Count tab stops before reaching footer

2. **Tab through header**

   - [ ] Menu/hamburger button gets focus (visible ring)
   - [ ] App title is not focusable (correct)
   - [ ] Settings icon text is not focusable (correct)

3. **Tab through main content (Desktop)**

   - [ ] Tab enters left sidebar
   - [ ] File upload area is focusable (role="button")
   - [ ] All tab buttons are focusable:
     - [ ] Signals tab
     - [ ] Advanced tab
     - [ ] Resample tab
     - [ ] Time tab
   - [ ] Focus indicator visible on all tabs
   - [ ] FilterPanel inputs are reachable
   - [ ] Filter Type select is focusable
   - [ ] Window Size input is focusable
   - [ ] Apply button is focusable
   - [ ] Reset button is focusable

4. **Tab through main content area**

   - [ ] Main Chart/Table tabs are focusable
   - [ ] Focus indicator visible on tabs
   - [ ] Chart area itself not focusable (correct)
   - [ ] Table is scrollable but not tab-focusable (correct)

5. **Tab through right sidebar (Desktop only)**

   - [ ] Stats tab is focusable
   - [ ] Analytics tab is focusable
   - [ ] Trendline tab is focusable
   - [ ] Export tab is focusable
   - [ ] Help tab is focusable
   - [ ] All tabs show focus indicator

6. **Tab through footer**
   - [ ] Footer text is not focusable (correct)
   - [ ] Can Tab back to header (cyclic)

### Pass Criteria

- [ ] All interactive elements reachable
- [ ] Tab order is logical (left-to-right, top-to-bottom)
- [ ] No focus traps (except modal when implemented)
- [ ] All focused elements show visible indicator

### Issues Found

(Record any issues here)

---

## Test 2: Keyboard Navigation (Mobile - if applicable)

### Objective

Test Tab navigation and mobile sidebar keyboard accessibility.

### Setup

- Open DevTools
- Press Ctrl+Shift+M (Chrome) or Cmd+Shift+M (Mac) to open mobile view
- Set to iPhone 12 or similar (375px width)

### Test Steps

1. **Open sidebar**

   - Press Tab until hamburger button focused
   - Press Enter to open sidebar
   - [ ] Sidebar slides in from left
   - [ ] Focus remains on hamburger (or moves to sidebar - either OK)

2. **Tab through sidebar**

   - [ ] File upload focusable
   - [ ] Tab buttons focusable
   - [ ] Filter inputs focusable
   - [ ] All have focus indicators

3. **Close sidebar via Escape** (NEW - Testing after fix)

   - [ ] Sidebar is open
   - [ ] Press Escape key
   - [ ] Sidebar closes
   - [ ] Focus returns to hamburger button (or somewhere reasonable)

4. **Close sidebar via overlay**
   - [ ] Open sidebar
   - [ ] Press Tab until focus reaches content behind modal
   - [ ] Focus should NOT reach content (focus trap)
   - [ ] OR press Escape and sidebar closes

### Pass Criteria

- [ ] Can open/close sidebar with keyboard
- [ ] Escape key closes sidebar
- [ ] All sidebar content focusable
- [ ] Focus doesn't escape behind modal (if trap implemented)

### Issues Found

(Record any issues here)

---

## Test 3: Focus Indicators Visibility

### Objective

Verify focus indicators are clearly visible on all interactive elements.

### Desktop Setup

- Chrome, at 100% zoom
- Default dark theme
- No special contrast settings

### Test Steps

1. **Tab through entire app**

   - Press Tab once
   - **Look for:** Blue ring around focused element
   - [ ] Ring is clearly visible?
   - [ ] Ring is at least 2px wide?
   - [ ] Ring color is blue-500 (#3b82f6)?

2. **Check each element type**

   **Tab buttons (left panel):**

   - Focus on "Signals" tab
   - [ ] Blue ring visible around button?
   - [ ] Ring doesn't get cut off by button border?
   - [ ] Ring has sufficient contrast with background?

   **Form inputs:**

   - Focus on "Filter Type" select
   - [ ] Blue ring visible inside or around input?
   - [ ] Ring is visible when typing?

   **Regular buttons:**

   - Focus on "Apply" button
   - [ ] Blue ring visible?
   - [ ] Ring different from button background color?

   **Icon buttons (SignalList):**

   - Focus on Upload icon button (top right of SignalList)
   - [ ] Blue ring visible around icon?
   - [ ] Ring has good contrast with button background?

3. **Check focus ring on different backgrounds**

   - Focus element on dark-800 background
     - [ ] Ring visible? (should be 4.8:1 contrast)
   - Focus element on dark-700 background
     - [ ] Ring visible? (should be 5.1:1 contrast)

4. **Check dark mode focus (already in dark mode)**
   - [ ] No issues with dark theme visibility?
   - [ ] Ring color (blue-500) works on all dark backgrounds?

### Pass Criteria

- [ ] All interactive elements show focus indicator
- [ ] Indicator is consistently styled (blue ring)
- [ ] Indicator is visible on all dark backgrounds
- [ ] Indicator doesn't obscure element content

### Issues Found

(Record any issues here)

---

## Test 4: Color Contrast (Manual)

### Objective

Verify critical text/background combinations meet WCAG AA 4.5:1.

### Tools Needed

- Chrome DevTools (built-in)

### Test Steps

1. **Check label text**

   - Right-click on any label (e.g., "Filter Type" in FilterPanel)
   - Inspect → Select element in DOM
   - In Styles tab, find color property
   - Look for contrast ratio indicator at bottom of color picker
   - [ ] Shows 4.5:1 or higher?
   - [ ] Indicates "AA PASS" or similar?

2. **Check placeholder text**

   - Right-click on search input in SignalList
   - Inspect placeholder (might need to focus input first)
   - Check contrast of placeholder color on input background
   - [ ] Shows 4.5:1 or higher?

3. **Check icon colors**

   - Right-click on upload icon (top of FileUpload)
   - Inspect parent element with text-dark-500 color
   - [ ] Contrast is 4.5:1 or higher?
   - OR visual check: Can you clearly see the icon without squinting?

4. **Check button text**

   - Right-click "Apply" button
   - Inspect text color and background
   - [ ] Shows 4.5:1 or higher?

5. **Check tab colors**
   - Right-click inactive "Advanced" tab
   - Inspect text color
   - [ ] Shows 4.5:1 or higher when compared to background?

### Using WebAIM Contrast Checker (Online)

1. Go to https://webaim.org/resources/contrastchecker/
2. Foreground color: Copy hex from DevTools
3. Background color: Copy hex of background element
4. [ ] Ratio shows as 4.5:1 or higher?
5. [ ] Large text (14pt+) shows 3:1 or higher?

### Pass Criteria

- [ ] All normal text: 4.5:1 or higher
- [ ] Large text (14pt+): 3:1 or higher
- [ ] UI components: 3:1 or higher
- [ ] Color alone not used to convey meaning

### Issues Found

(Record any issues here)

---

## Test 5: Form Input Accessibility

### Objective

Verify form labels, inputs, and validation are properly structured.

### Test Steps

1. **Label association**

   - Right-click on label in FilterPanel (e.g., "Window Size")
   - Inspect
   - [ ] Label has `htmlFor` attribute?
   - [ ] `htmlFor` matches an input `id`?
   - [ ] Clicking label focuses corresponding input?

2. **Input keyboard navigation**

   - Tab to "Filter Type" select
   - [ ] Select is focused (blue ring visible)?
   - Press ArrowDown
   - [ ] Next option selected?
   - Press ArrowUp
   - [ ] Previous option selected?
   - Press Enter/Space
   - [ ] Option confirmed?

3. **Number inputs**

   - Tab to "Window Size" input
   - [ ] Input focused?
   - Type "10"
   - [ ] Value appears in input?
   - Press Tab
   - [ ] Moves to next element?
   - Press ArrowUp/Down
   - [ ] Value increments/decrements? (native number input behavior)

4. **Error state** (if applicable)
   - Try to enter invalid value (< min or > max)
   - [ ] Browser shows validation error?
   - [ ] Error text is accessible (for future implementation)?

### Pass Criteria

- [ ] All labels properly associated
- [ ] All inputs keyboard accessible
- [ ] Form navigation logical
- [ ] Validation feedback clear

### Issues Found

(Record any issues here)

---

## Test 6: Signal List Interaction

### Objective

Test signal selection and search functionality with keyboard.

### Test Steps

1. **Load a CSV file first**

   - Tab to FileUpload area
   - Press Enter to activate
   - Select any CSV file
   - [ ] File loaded (toast notification appears)?

2. **Signal selection with keyboard**

   - Tab to first signal button
   - [ ] Button has focus ring?
   - [ ] Visual change showing focus?
   - Press Enter
   - [ ] Signal toggles selection?
   - [ ] Checkbox/indicator updates?
   - Press Tab to next signal
   - [ ] Navigation smooth?

3. **Search functionality**

   - Tab to search input
   - [ ] Input has focus ring?
   - Type "temp" (if "Temperature" signal exists)
   - [ ] List filters to matching signals?
   - [ ] Number of visible signals decreases?
   - Press ArrowDown
   - [ ] Can navigate filtered results? (If implemented)
   - Press Escape
   - [ ] Clears search and shows all signals?

4. **Select All / Deselect All buttons**
   - Tab to "All" button
   - [ ] Button focused?
   - Press Enter
   - [ ] All signals selected?
   - Tab to "None" button
   - [ ] Button focused?
   - Press Enter
   - [ ] All signals deselected?

### Pass Criteria

- [ ] All signal operations work from keyboard
- [ ] No mouse required for any selection
- [ ] Feedback clear (visual + color change)
- [ ] Roving tabindex implemented (if 100+ signals)

### Issues Found

(Record any issues here)

---

## Test 7: Tab Panel Navigation

### Objective

Verify tab panels are keyboard accessible and properly structured.

### Test Steps

1. **Tab panel ARIA roles**

   - Right-click on "Signals" tab button
   - Inspect → Check Attributes
   - [ ] Has `role="tab"`?
   - [ ] Has `aria-selected` attribute?
   - [ ] Has `aria-controls` pointing to panel ID?

2. **Tab panel content**

   - Right-click on SignalList content area
   - Inspect → Check Attributes
   - [ ] Has `role="tabpanel"`?
   - [ ] Has `aria-labelledby` pointing to tab ID?

3. **Tab switching**

   - Focus on "Signals" tab
   - [ ] Blue ring visible?
   - Press ArrowRight
   - [ ] Focus moves to "Advanced" tab?
   - [ ] Content changes to Advanced panel?
   - [ ] aria-selected updates?
   - Press ArrowRight again
   - [ ] Cycles through all tabs?
   - Press ArrowLeft
   - [ ] Cycles backward?

4. **Right panel tabs**
   - Same as above for right panel tabs
   - [ ] ARIA roles present?
   - [ ] Arrow key navigation works?

### Pass Criteria

- [ ] All tab panels have proper ARIA roles
- [ ] Keyboard navigation works (Tab to reach, Arrow to switch)
- [ ] Visual feedback clear (highlighted tab)
- [ ] Panel content updates on switch

### Issues Found

(Record any issues here)

---

## Test 8: Modal/Sidebar Focus (After Implementation)

### Objective

Test focus trap and modal keyboard behavior.

### Test Steps

1. **Open mobile sidebar (Mobile view)**

   - Reduce window to < 768px width
   - Tab to hamburger button
   - Press Enter
   - [ ] Sidebar opens?
   - [ ] Focus visible?

2. **Focus trap test**

   - Continue pressing Tab
   - [ ] Focus stays within sidebar?
   - [ ] Focus doesn't jump to content behind modal?
   - After cycling through all sidebar elements, Tab again
   - [ ] Focus loops back to first sidebar element?

3. **Escape to close**

   - Sidebar open
   - Press Escape
   - [ ] Sidebar closes immediately?
   - [ ] Focus returns to hamburger button?
   - [ ] Can open again with Enter?

4. **Click overlay to close**
   - Sidebar open
   - Tab to ensure focus in sidebar
   - With keyboard only, is there a way to close via overlay?
   - (Overlay is click-only, so this might fail - that's OK for now)

### Pass Criteria

- [ ] Focus trapped in sidebar (doesn't escape to background)
- [ ] Escape key closes sidebar
- [ ] Focus returns to trigger element
- [ ] No keyboard trap (can always close and move on)

### Issues Found

(Record any issues here)

---

## Test 9: Screen Reader Testing (Optional - Phase 2.2+)

### Tools

- NVDA (Windows) - Free, download from https://www.nvaccess.org/
- VoiceOver (Mac) - Built-in, activate with Cmd+F5

### Quick NVDA Test

1. Download NVDA
2. Run and start scanning (Insert key by default)
3. Tab through app, listen to announcements
4. [ ] Elements announced properly?
5. [ ] Tab panels announced as "tab" role?
6. [ ] Buttons announced as "button"?
7. [ ] Checkboxes announced as "checkbox"?
8. [ ] Error messages announced?

### Expected Announcements

- "Signals tab, selected" (or current tab name)
- "Advanced tab, not selected"
- "Signals checkbox, checked" (for selected signal)
- "Temperature checkbox, not checked" (for unselected)
- "Load signal set button"
- "Filter Type combobox" (or similar)

### Status for Phase 2.1

- [ ] ARIA roles present and correct
- [ ] ARIA labels present where needed
- [ ] ARIA live regions functional

---

## Test 10: Responsive Design (Keyboard + Focus)

### Test Steps

1. **Desktop (1024px+)**

   - [ ] Tab navigation works?
   - [ ] Focus indicators visible?
   - [ ] All panels visible?
   - [ ] No overflow issues?

2. **Tablet (768-1023px)**

   - [ ] Right panel hidden (correct)?
   - [ ] Tab navigation still complete?
   - [ ] Focus indicators visible?
   - [ ] Sidebar functions?

3. **Mobile (< 768px)**
   - [ ] Left sidebar hidden by default?
   - [ ] Hamburger button toggles sidebar?
   - [ ] Focus trap in sidebar?
   - [ ] Can access all features with keyboard?

### Pass Criteria

- [ ] Functionality maintained across all breakpoints
- [ ] Keyboard navigation always available
- [ ] Focus always visible
- [ ] No unexpected scrolling or jumps

---

## Test 11: Browser Compatibility

### Browsers to Test

- [ ] Chrome 125+
- [ ] Firefox 125+
- [ ] Safari 17+
- [ ] Edge 125+

### Test Steps for Each Browser

1. Load app
2. Tab through app
3. [ ] Focus visible in this browser?
4. [ ] Colors appear correct?
5. [ ] No console errors?
6. [ ] Focus-visible works? (may require polyfill in older browsers)

### Known Issues

- Safari: May need `-webkit-focus-visible` fallback
- Firefox: Standard focus-visible support good
- Chrome/Edge: Full support

---

## Test Report Template

### Issue Summary

| ID  | Element | Issue              | Severity | Browser | Screenshot |
| --- | ------- | ------------------ | -------- | ------- | ---------- |
| 1   | Labels  | Text not visible   | P0       | Chrome  | [link]     |
| 2   | Tabs    | Focus ring missing | P0       | Chrome  | [link]     |

### Focus Indicators Test

- [ ] Header buttons: ✓ / ✗
- [ ] Tab buttons: ✓ / ✗
- [ ] Form inputs: ✓ / ✗
- [ ] Regular buttons: ✓ / ✗
- [ ] Icon buttons: ✓ / ✗

### Keyboard Navigation Test

- [ ] All elements reachable: ✓ / ✗
- [ ] Tab order logical: ✓ / ✗
- [ ] No focus traps: ✓ / ✗
- [ ] Escape closes modal: ✓ / ✗
- [ ] Arrow keys work in selects: ✓ / ✗

### Color Contrast Test

- [ ] Labels WCAG AA: ✓ / ✗
- [ ] Placeholders WCAG AA: ✓ / ✗
- [ ] Icon colors visible: ✓ / ✗
- [ ] Button text WCAG AA: ✓ / ✗
- [ ] Tabs WCAG AA: ✓ / ✗

### ARIA Test

- [ ] Tab roles present: ✓ / ✗
- [ ] Icon labels present: ✓ / ✗
- [ ] Signal items semantic: ✓ / ✗
- [ ] Labels associated: ✓ / ✗
- [ ] Live regions work: ✓ / ✗

---

## Sign-Off

- [ ] All P0 tests passing
- [ ] All P1 tests passing
- [ ] No critical issues remain
- [ ] At least 2 browsers tested
- [ ] Screen reader basic check done (optional)

**Tester Name:** **\*\***\_\_\_**\*\***  
**Date:** **\*\***\_\_\_**\*\***  
**Overall Status:** ✓ PASS / ✗ FAIL / ~ PARTIAL

---

## Next Steps After Testing

1. **Document any failures** with screenshots and exact steps to reproduce
2. **Create GitHub issues** for each failure
3. **Link to this test plan** in issue descriptions
4. **Prioritize fixes** as P0 (blocking), P1 (high), P2 (medium)
5. **Re-test after fixes** with this same checklist

---

**Test Plan Created:** April 30, 2026  
**Version:** 1.0  
**Status:** Ready for Testing
