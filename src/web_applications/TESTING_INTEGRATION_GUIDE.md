# Phase 2.2 - Testing Integration Guide

## Overview

This guide covers the integration testing procedures for Phase 2.2 Frontend Polish & Integration across Aurora CAS Calculator, Unit Converter, and URDF Viewer. It builds on the Mobile Testing Checklist and provides step-by-step procedures for manual and automated testing.

---

## Test Structure

### Test Phases

1. **Responsive Design Validation** (375px, 768px, 1024px, 1920px)
2. **Accessibility Compliance** (WCAG AA, aria-labels, keyboard navigation)
3. **Error Handling Integration** (Toast component, input validation, network errors)
4. **Focus Management & Keyboard Navigation** (Tab order, focus-visible styles)
5. **Touch Target Sizing** (Minimum 44px × 44px, 8px spacing)
6. **Browser Compatibility** (Chrome, Firefox, Safari, Edge)

---

## Part 1: Responsive Design Testing

### 1.1 Mobile Testing (375px Viewport)

#### Aurora CAS Calculator

**Test Environment:**
- DevTools: Toggle device toolbar (Ctrl+Shift+M)
- Select "iPhone SE" preset (375px width)
- Start dev server: `cd calculator && flask --app webapp run`
- Navigate to: `http://localhost:5000`

**Layout Validation:**
1. Open DevTools Console and run:
```javascript
// Verify layout dimensions
const shell = document.querySelector('.calculator-shell');
const screen = document.querySelector('.screen');
const keypad = document.querySelector('.keypad');

console.log('Shell width:', shell.offsetWidth);
console.log('Screen height:', screen.offsetHeight);
console.log('Keypad visible:', keypad.offsetHeight);

// Check for horizontal overflow
const scrollWidth = document.documentElement.scrollWidth;
const clientWidth = document.documentElement.clientWidth;
console.log('Horizontal overflow:', scrollWidth > clientWidth ? 'YES' : 'NO');
```

**Checklist:**
- [ ] Shell width ≤ 375px (no horizontal scroll)
- [ ] All inputs stack vertically (1 column)
- [ ] Bounds row inputs stack in single column
- [ ] Buttons ≥44px height
- [ ] Text readable (≥12px)
- [ ] No content clipping

**Screenshot:**
```bash
# Take screenshot at 375px
# File: test_results/calculator-375px-layout.png
```

#### Unit Converter

**Test Environment:**
- Open: `http://localhost:8000` (or file://)
- DevTools: iPhone SE preset (375px)

**Layout Validation:**
1. Category dropdown spans full width
2. Input fields stack vertically
3. All controls ≥44px tall
4. No horizontal scroll

**Screenshot:**
```bash
# File: test_results/unit-converter-375px-layout.png
```

#### URDF Viewer

**Test Environment:**
- Start server: `cd urdf_viewer && uvicorn app:app --reload`
- Navigate to: `http://localhost:8000`
- DevTools: 375px preset

**Validation:**
- [ ] 3D canvas responsive
- [ ] File upload area accessible
- [ ] Controls tappable

---

### 1.2 Tablet Testing (768px Viewport)

**Test Environment:**
- DevTools: iPad Mini preset (768px)

**All Apps:**
1. [ ] Layout adapts for wider screen
2. [ ] Multi-column layouts work
3. [ ] Touch targets still ≥44px
4. [ ] Text readable without zoom

**Screenshots:**
- calculator-768px-layout.png
- unit-converter-768px-layout.png
- urdf-viewer-768px-layout.png

---

### 1.3 Large Tablet / Landscape (1024px Viewport)

**Test Environment:**
- DevTools: iPad Pro preset (1024px)

**Validation:**
- [ ] Utilizes horizontal space efficiently
- [ ] Multi-column layouts functional
- [ ] All elements accessible

---

## Part 2: Accessibility Testing

### 2.1 Aria-Labels & Semantic HTML

#### Aurora CAS Calculator

**Critical Elements Requiring aria-labels:**
- Keypad buttons (numbers, operators)
- Function strip buttons
- Mode buttons
- Touch control buttons
- Copy buttons

**Verification Script:**
```javascript
// Run in DevTools Console
const unlabeledElements = [];
document.querySelectorAll('button').forEach(btn => {
  const hasLabel = btn.getAttribute('aria-label') || btn.textContent?.trim();
  if (!hasLabel) {
    unlabeledElements.push(btn);
    console.log('Missing label:', btn.outerHTML.slice(0, 100));
  }
});
console.log('Total unlabeled buttons:', unlabeledElements.length);
```

**Expected Results:**
- [ ] 0 unlabeled buttons
- [ ] All inputs have associated labels
- [ ] Result display has aria-live="polite"
- [ ] Proper heading hierarchy

#### Unit Converter

**Critical Elements:**
- Category selector
- From/To unit dropdowns
- Custom units button
- Theme toggle button

**Verification:**
```javascript
// Check for semantic form elements
const form = document.querySelector('form');
const inputs = form ? form.querySelectorAll('input, select') : [];
console.log('Form inputs:', inputs.length);

inputs.forEach(input => {
  const label = document.querySelector(`label[for="${input.id}"]`);
  console.log(input.id, label ? 'HAS label' : 'MISSING label');
});
```

---

### 2.2 Screen Reader Testing

#### Test Procedure (Mac with VoiceOver / Windows with NVDA)

**Calculator - Expected Output Order:**
1. "Aurora CAS Calculator, application"
2. "Battery level indicator"
3. "Main landmark"
4. "Expression, required, edit text"
5. "Variable / Unknown, edit text"
6. "Order, spinner"
7. "Bounds row, group"
8. "Result display, region, polite"
9. "Touch edit, region"
10. "Function strip, region"
11. "Mode buttons, region"
12. "Keypad, region"

**Test Steps:**
1. Open calculator in browser
2. Enable screen reader (VoiceOver/NVDA)
3. Navigate with arrow keys and Tab
4. Verify output matches expected sequence
5. Perform calculation and verify result announcement

---

### 2.3 Keyboard Navigation & Tab Order

#### Aurora Calculator Tab Order Test

**Procedure:**
1. Load page
2. Press `Tab` repeatedly to navigate
3. Record the order of focused elements

**Expected Order:**
```
1. Expression input
2. Variable input
3. Order input
4. Lower bound input
5. Upper bound input
6. Value input
7. Substitutions input
8. Copy result button
9. Copy expression button
10. Touch control buttons (in order)
11. Function strip buttons
12. Mode buttons
13. Keypad buttons (left-to-right, top-to-bottom)
14. CLEAR and ENTER
```

**Verification Script:**
```javascript
// Log tab order
let tabIndex = 0;
document.querySelectorAll(
  'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
).forEach(el => {
  console.log(
    tabIndex++,
    el.tagName,
    el.id || el.class || el.getAttribute('aria-label')?.slice(0, 30) || 'unlabeled'
  );
});
```

#### Test Keyboard Shortcuts

**Calculator:**
- [ ] `Enter` submits calculation
- [ ] `Escape` clears form (if implemented)
- [ ] Arrow keys navigate history (if applicable)

**Unit Converter:**
- [ ] `Tab` navigates dropdowns
- [ ] `Enter` confirms selections
- [ ] Arrow keys work in dropdowns

---

### 2.4 Focus Indicators & Focus-Visible

#### Procedure

1. Open page
2. Use keyboard only (NO MOUSE)
3. Tab through all elements
4. Verify focus indicator visible on each element

#### Focus Indicator Requirements

```css
*:focus-visible {
  outline: 2px solid #8bd3f7;     /* cyan/blue */
  outline-offset: 2px;             /* doesn't overlap */
}
```

#### Elements Requiring Focus Styles

**Calculator:**
- Input fields
- All buttons (keypad, function, mode, touch, copy)
- Mode buttons

**Unit Converter:**
- Category dropdown
- Unit dropdowns
- Input fields
- All action buttons

**Verification:**
```javascript
// Test focus visibility
const focusableElements = document.querySelectorAll(
  'button, input, select, textarea, [tabindex]:not([tabindex="-1"])'
);

focusableElements.forEach(el => {
  el.focus();
  const styles = window.getComputedStyle(el, ':focus-visible');
  const outline = styles.outline;
  const visible = outline && outline !== 'none';
  if (!visible) {
    console.log('MISSING focus style:', el.getAttribute('aria-label') || el.type || el.textContent);
  }
});
```

---

## Part 3: Error Handling Integration

### 3.1 Toast Component Implementation

#### Requirements

**Toast Container (index.html):**
```html
<div id="toast-container" class="toast-container" role="region" aria-label="notifications" aria-live="polite" aria-atomic="false"></div>
```

**Toast Component (toast.js):**
```javascript
class Toast {
  constructor() {
    this.container = document.getElementById('toast-container');
  }

  show(message, type = 'info', duration = 5000) {
    const id = `toast-${Date.now()}`;
    const toast = document.createElement('div');
    toast.id = id;
    toast.className = `toast toast-${type}`;
    toast.setAttribute('role', 'alert');
    toast.textContent = message;

    this.container.appendChild(toast);

    if (duration > 0) {
      setTimeout(() => this.dismiss(id), duration);
    }

    return id;
  }

  dismiss(id) {
    const toast = document.getElementById(id);
    if (toast) toast.remove();
  }
}

const toast = new Toast();
```

---

### 3.2 Error Test Cases

#### Calculator - Invalid Input

**Test Case 1: Unmatched Parentheses**
1. Enter expression: `((1+2)`
2. Click or press Enter
3. Expected: Toast shows "Error: Unmatched parentheses"
4. Toast auto-dismisses after 5s
5. Input field retains value

**Test Case 2: Invalid Variable Name**
1. Enter expression: `x^2`
2. Enter variable: `123invalid`
3. Submit
4. Expected: Toast shows "Error: Invalid variable name"

**Test Case 3: Missing Expression**
1. Leave expression empty
2. Click Enter
3. Expected: Toast shows "Error: Expression required"

#### Unit Converter - Invalid Input

**Test Case 1: Non-numeric Input**
1. Enter non-number: `abc`
2. Expected: Toast shows "Error: Please enter a valid number"

**Test Case 2: Out of Range**
1. Enter very large number: `9e308` (near float limit)
2. Expected: Toast shows validation warning if applicable

#### URDF Viewer - File Upload

**Test Case 1: Invalid File Format**
1. Upload .txt file instead of .urdf
2. Expected: Toast shows "Error: Invalid URDF file format"

**Test Case 2: Network Error**
1. Disable network in DevTools
2. Try to load model
3. Expected: Toast shows "Error: Network error. Please check connection."
4. Retry button appears

---

### 3.3 Toast Styling Requirements

**CSS:**
```css
.toast-container {
  position: fixed;
  top: 16px;
  right: 16px;
  z-index: 9999;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.toast {
  padding: 12px 16px;
  border-radius: 8px;
  font-size: 14px;
  line-height: 1.4;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  animation: slideIn 300ms ease-out;
  max-width: 90vw;
}

.toast-success {
  background-color: #10b981;
  color: white;
}

.toast-error {
  background-color: #ef4444;
  color: white;
}

.toast-info {
  background-color: #3b82f6;
  color: white;
}

.toast-warning {
  background-color: #f59e0b;
  color: white;
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateX(100%);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

@media (max-width: 640px) {
  .toast-container {
    top: 8px;
    right: 8px;
    left: 8px;
  }

  .toast {
    max-width: 100%;
  }
}
```

---

## Part 4: Touch Target Sizing

### 4.1 Measurement Procedure

**DevTools Inspector:**
1. Right-click on button
2. Select "Inspect"
3. In Elements panel, find computed dimensions
4. Verify width ≥ 44px and height ≥ 44px

**Automated Check:**
```javascript
// Check all button sizes
document.querySelectorAll('button').forEach(btn => {
  const width = btn.offsetWidth;
  const height = btn.offsetHeight;
  const label = btn.getAttribute('aria-label') || btn.textContent?.slice(0, 20);

  if (width < 44 || height < 44) {
    console.log(`SMALL: ${label} [${width}x${height}]`);
  }
});

// Check spacing between buttons
const buttons = Array.from(document.querySelectorAll('button'));
buttons.forEach((btn, i) => {
  const next = buttons[i + 1];
  if (!next) return;

  const gap = next.getBoundingClientRect().left - btn.getBoundingClientRect().right;
  if (gap < 8) {
    console.log(`TIGHT spacing: ${gap}px between buttons`);
  }
});
```

### 4.2 Elements to Check

**Calculator:**
- [ ] Keypad buttons (0-9): 44×44px minimum
- [ ] Operation buttons (+, -, ×, ÷): 44×44px
- [ ] Function buttons (sin, cos, etc.): 44×44px
- [ ] Mode buttons: 44×44px
- [ ] CLEAR and ENTER: 44×44px
- [ ] Touch controls: 44×44px
- [ ] Copy buttons: 44×44px

**Unit Converter:**
- [ ] Category dropdown: 44px height
- [ ] Unit selectors: 44px height
- [ ] Custom units button: 44×44px
- [ ] Theme toggle: 44×44px

---

## Part 5: Cross-Browser Testing

### Test Matrix

| Browser | Mobile | Tablet | Desktop |
|---------|--------|--------|---------|
| Chrome  | ✓      | ✓      | ✓       |
| Firefox | ✓      | ✓      | ✓       |
| Safari  | ✓      | ✓      | ✓       |
| Edge    | -      | -      | ✓       |

### Browser-Specific Tests

**Safari on iOS:**
- [ ] Viewport scaling correct
- [ ] Touch interactions smooth
- [ ] Zoom behavior controlled (`maximum-scale=1.0`)
- [ ] Safe area insets respected (notch/home indicator)

**Firefox on Android:**
- [ ] Hardware back button doesn't break app
- [ ] Touch gestures work
- [ ] Forms display correctly

---

## Part 6: Performance & Rendering

### 6.1 Responsive Image & Asset Loading

**Procedure:**
1. Open DevTools Network tab
2. Set throttling to "Fast 3G"
3. Load page and measure:
   - [ ] First Contentful Paint (FCP) < 2s
   - [ ] Largest Contentful Paint (LCP) < 2.5s
   - [ ] Cumulative Layout Shift (CLS) < 0.1

### 6.2 JavaScript Performance

**Measure Interaction to Next Paint (INP):**
```javascript
// Measure interaction latency
const observer = new PerformanceObserver((entryList) => {
  const entries = entryList.getEntries();
  entries.forEach((entry) => {
    console.log('INP:', entry.duration, 'ms for', entry.name);
  });
});

observer.observe({ type: 'event', durable: true });
```

---

## Part 7: Test Documentation

### Test Session Template

**File:** `test_results/[app]-[date]-[viewport].md`

```markdown
# Test Session Report

## Metadata
- **Application:** Aurora CAS Calculator
- **Date:** 2026-04-30
- **Tester:** [Your Name]
- **Viewport:** 375px
- **Browser:** Chrome 126
- **Device:** MacBook / iPhone 14

## Layout Tests
- **Status:** PASS / FAIL
- **Issues Found:**
  - [Issue 1]
  - [Issue 2]
- **Screenshots:** [Attached]

## Accessibility Tests
- **Axe DevTools Violations:** 0
- **Keyboard Navigation:** PASS / FAIL
- **Screen Reader Test:** PASS / FAIL
- **Focus Indicators:** PASS / FAIL

## Error Handling Tests
- **Toast Component:** WORKING / PARTIAL / MISSING
- **Test Results:**
  - [Test 1]: PASS / FAIL
  - [Test 2]: PASS / FAIL

## Touch Target Sizing
- **Status:** PASS / FAIL
- **Buttons < 44px:** [List if any]

## Overall Status
- **PASS** / **FAIL**
- **Blockers:** None / [List]
- **Recommendations:** [If any]

## Sign-off
Tested by: ____________________
Date: ____________________
```

---

## Part 8: Automated Testing (CI/CD Integration)

### Lighthouse CI

**Setup:**
```bash
npm install -g @lhci/cli@*
lhci autorun
```

**lighthouse-ci.json:**
```json
{
  "ci": {
    "collect": {
      "url": ["http://localhost:5000", "http://localhost:8000"],
      "numberOfRuns": 3,
      "settings": {
        "formFactor": "mobile"
      }
    },
    "assert": {
      "preset": "lighthouse:recommended",
      "assertions": {
        "categories:accessibility": ["error", { "minScore": 0.90 }],
        "categories:best-practices": ["error", { "minScore": 0.90 }]
      }
    }
  }
}
```

### Axe-Core Testing

**Node Test:**
```javascript
// test/accessibility.test.js
const { AxePuppeteer } = require('@axe-core/puppeteer');
const puppeteer = require('puppeteer');

describe('Accessibility', () => {
  let browser;

  beforeAll(async () => {
    browser = await puppeteer.launch();
  });

  afterAll(async () => {
    await browser.close();
  });

  test('Calculator WCAG AA compliance', async () => {
    const page = await browser.newPage();
    await page.goto('http://localhost:5000');

    const results = await new AxePuppeteer(page).analyze();
    expect(results.violations.length).toBe(0);
  });
});
```

---

## Part 9: Deployment Checklist

Before deploying:
- [ ] All tests pass (responsive, accessibility, error handling)
- [ ] Axe DevTools: 0 violations
- [ ] Lighthouse: ≥90 accessibility score
- [ ] Touch targets: all ≥44px
- [ ] Focus indicators: visible on all interactive elements
- [ ] Toast component: integrated and tested
- [ ] No console errors or warnings
- [ ] Mobile device testing complete (real device)
- [ ] Screenshot documentation complete
- [ ] Browser compatibility verified

---

## References

- [WCAG 2.1 Level AA](https://www.w3.org/WAI/WCAG21/quickref/)
- [ARIA Authoring Practices](https://www.w3.org/WAI/ARIA/apg/)
- [MDN: Responsive Design](https://developer.mozilla.org/en-US/docs/Learn/CSS/CSS_layout/Responsive_Design)
- [Apple Human Interface Guidelines](https://developer.apple.com/design/human-interface-guidelines/)
- [Material Design 3](https://m3.material.io/)

---

End of Testing Integration Guide
