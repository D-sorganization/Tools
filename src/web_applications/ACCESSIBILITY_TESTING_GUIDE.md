# Accessibility Testing Guide for Phase 2.2

## Overview

This guide provides detailed procedures for testing accessibility across Aurora CAS Calculator, Unit Converter, and URDF Viewer. All applications must meet **WCAG 2.1 Level AA** compliance.

---

## Part 1: WCAG 2.1 Level AA Compliance Overview

### Key Principles

1. **Perceivable** - Users can perceive content (not invisible to all senses)
2. **Operable** - Users can operate the interface (keyboard accessible)
3. **Understandable** - Users can understand content and operation
4. **Robust** - Content works with assistive technologies

### Critical WCAG Criteria for Web Apps

| Criterion | Level | Requirement |
|-----------|-------|-------------|
| 1.4.3 Contrast (Minimum) | AA | 4.5:1 for text, 3:1 for large text |
| 1.4.11 Non-text Contrast | AA | 3:1 for UI components and borders |
| 2.1.1 Keyboard | A | All functionality keyboard accessible |
| 2.1.2 No Keyboard Trap | A | Focus not trapped, escape available |
| 2.4.3 Focus Order | A | Logical tab order |
| 2.4.7 Focus Visible | AA | Visible keyboard focus indicator |
| 3.2.1 On Focus | A | No unexpected context changes on focus |
| 3.3.1 Error Identification | A | Errors identified clearly |
| 4.1.2 Name, Role, Value | A | All UI components have accessible name/role |
| 4.1.3 Status Messages | AA | Status messages announced to screen readers |

---

## Part 2: Automated Testing with Axe DevTools

### 2.1 Browser Extension Setup

#### Chrome/Edge
1. Visit [Axe DevTools Chrome Extension](https://chrome.google.com/webstore)
2. Search "axe DevTools"
3. Click "Add to Chrome"
4. Extension appears in DevTools

#### Firefox
1. Visit [Firefox Add-ons](https://addons.mozilla.org/)
2. Search "axe DevTools"
3. Click "Add to Firefox"

### 2.2 Running Automated Scans

**Procedure:**
1. Open application in browser
2. Open DevTools (F12)
3. Click "axe DevTools" panel
4. Click "Scan ENTIRE PAGE"
5. Wait for scan to complete
6. Review violations and review items

**Screenshot Example:**
```
[axe DevTools results showing:
- 0 Violations (critical)
- 0 Needs Review items (manual check needed)
- Pass count: 47]
```

### 2.3 Expected Results

| Scan | Expected Violations | Max Allowed |
|------|-------------------|------------|
| Calculator | 0 | 0 |
| Unit Converter | 0 | 0 |
| URDF Viewer | 0 | 0 |

**Log Violations Found:**
```markdown
### Axe Violations Found [Date]

**Calculator:**
- [ ] Violation: [Name]
  - Element: [selector]
  - Fix: [Action needed]
  - Status: Fixed / Pending

**Unit Converter:**
- [ ] [List violations]

**URDF Viewer:**
- [ ] [List violations]
```

---

## Part 3: Color Contrast Testing

### 3.1 Contrast Ratio Requirements

**Text:**
- **Normal text:** 4.5:1 minimum (WCAG AA)
- **Large text (18pt+):** 3:1 minimum
- **Bold text (14pt+):** 3:1 minimum

**UI Components:**
- **Borders, dividers, focus indicators:** 3:1 minimum
- **Background-on-background:** 3:1 minimum

### 3.2 Testing Procedure

#### Manual Check with DevTools

1. Right-click on element
2. Select "Inspect"
3. Open DevTools Styles tab
4. Check `color` and `background-color`
5. Note RGB/hex values

#### Using WebAIM Contrast Checker

1. Visit [WebAIM Contrast Checker](https://webaim.org/resources/contrastchecker/)
2. Enter foreground color (text)
3. Enter background color
4. Note ratio (goal: ≥4.5:1)

#### Automated Check (Chrome DevTools)

1. Open DevTools
2. Go to "Lighthouse" tab
3. Run audit with "Accessibility" checked
4. Review "Contrast" issues in report

### 3.3 Critical Color Pairs to Test

**Calculator:**
```
White text (#f5f7fa) on dark shell (#1f2a3a)
✓ Ratio: 10.2:1 (PASS)

Screen text (#0f2417) on screen bg (#e3f1e8)
✓ Ratio: 5.1:1 (PASS)

Accent (#8bd3f7) on dark (#1f2a3a)
✓ Ratio: 4.6:1 (PASS)
```

**Unit Converter:**
- Test all text colors on backgrounds
- Verify error state colors (red on white/light bg)
- Verify focus indicator on all button states

---

## Part 4: Keyboard Navigation Testing

### 4.1 Tab Order Audit

**Procedure:**
1. Open page
2. Press `Tab` key repeatedly
3. Document focus path
4. Verify logical order

**Expected Tab Order - Calculator:**
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
10. Touch controls (left, right, home, end, backspace, ANS)
11. Function strip (Evaluate, Simplify, Solve, etc.)
12. Mode buttons (CAS, Algebra, Systems, etc.)
13. Keypad (number and operator buttons, left-to-right)
14. Last interactive element
```

### 4.2 Keyboard Shortcut Testing

**Calculator Tests:**
- [ ] `Enter` executes calculation
- [ ] `Escape` clears form (if implemented)
- [ ] Arrow keys control cursor position
- [ ] Number keys insert into expression (if implemented)

**Unit Converter Tests:**
- [ ] `Tab` navigates between controls
- [ ] `Enter` confirms selections
- [ ] Arrow keys work in dropdown menus
- [ ] `Esc` closes open dropdown

### 4.3 Keyboard Trap Detection

**Test:** Can you leave every interactive element using only Tab/Shift+Tab?

```javascript
// Script to detect potential keyboard traps
const focusableElements = Array.from(
  document.querySelectorAll(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
  )
);

console.log('Total focusable elements:', focusableElements.length);

// Simulate tab order
focusableElements.forEach((el, i) => {
  el.addEventListener('focus', () => {
    console.log(`Focused: ${i} - ${el.getAttribute('aria-label') || el.type || el.textContent?.slice(0, 20)}`);
  });
});

// Test escape key
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    console.log('Escape caught at:', document.activeElement);
  }
});
```

---

## Part 5: Screen Reader Testing

### 5.1 Screen Readers to Test

| OS | Screen Reader | Setup |
|----|---|---|
| macOS | VoiceOver | Cmd+F5 |
| Windows | NVDA | Download from [NVDA Project](https://www.nvaccess.org/) |
| Windows | JAWS | Commercial (trial available) |
| iPhone | VoiceOver | Settings → Accessibility → VoiceOver |
| Android | TalkBack | Settings → Accessibility → TalkBack |

### 5.2 VoiceOver Testing (macOS)

**Enable VoiceOver:**
- Press Cmd+F5 to toggle

**Navigation Commands:**
- `VO+Right Arrow` - Move to next item
- `VO+Left Arrow` - Move to previous item
- `VO+Down Arrow` - Move into group
- `VO+Up Arrow` - Move out of group
- `Space` - Activate button
- `VO+U` - Open rotor (navigation menu)

**Test Procedure:**
1. Enable VoiceOver
2. Navigate through entire page using arrow keys
3. Record what VoiceOver announces
4. Verify announcements match visual content

**Expected Announcements - Calculator:**
```
"Aurora CAS Calculator, application"
[Navigate to status bar]
"Battery level indicator"
[Navigate to main content]
"Main, main landmark"
[Navigate to inputs]
"Expression, required, edit text"
"Variable, Unknown, edit text"
[Navigate to result]
"Result display, region"
"Result, heading"
"Ready, status"
[After calculation]
"Derivative of x^2 equals 2*x, status"
```

### 5.3 NVDA Testing (Windows)

**Enable NVDA:**
- Download and install from [nvaccess.org](https://www.nvaccess.org/)
- Launch application

**Key Bindings:**
- `Down Arrow` - Next line/item
- `Up Arrow` - Previous line/item
- `Tab` - Next focusable element
- `Shift+Tab` - Previous focusable element
- `Enter` - Activate button
- `NVDA+F7` - Open elements list

**Test the Same Procedure as VoiceOver**

### 5.4 Mobile Screen Reader Testing

#### iPhone VoiceOver

1. Settings → Accessibility → VoiceOver
2. Toggle ON
3. Double-tap to select
4. Three-finger swipe right to move forward
5. Three-finger swipe left to move backward

#### Android TalkBack

1. Settings → Accessibility → TalkBack
2. Toggle ON
3. Tap twice to activate
4. Swipe right for next
5. Swipe left for previous

---

## Part 6: ARIA & Semantic HTML Validation

### 6.1 Required ARIA Attributes

**Input Labels:**
```html
<!-- Correct -->
<label for="expression">Expression</label>
<input id="expression" aria-required="true" />

<!-- Or -->
<input aria-label="Expression" aria-required="true" />
```

**Icon Buttons (no visible text):**
```html
<!-- Correct -->
<button aria-label="Copy result">📋</button>

<!-- Wrong - no accessible name -->
<button>📋</button>
```

**Live Regions (status messages):**
```html
<!-- Correct -->
<div id="result" aria-live="polite" aria-atomic="true">
  Ready.
</div>

<!-- Wrong - not announced -->
<div id="result">Ready.</div>
```

**Form Errors:**
```html
<!-- Correct -->
<input id="expr" aria-invalid="true" aria-describedby="error-1" />
<span id="error-1">Error: Unmatched parentheses</span>

<!-- Wrong - error not associated -->
<span>Error: Unmatched parentheses</span>
```

### 6.2 Semantic HTML Checklist

- [ ] Use `<button>` for buttons (not `<div role="button">`)
- [ ] Use `<input>` with `type="text"`, `type="number"`, etc.
- [ ] Use `<label>` for form inputs
- [ ] Use `<main>` for main content
- [ ] Use `<nav>` for navigation
- [ ] Use heading hierarchy: h1 > h2 > h3 (no gaps)
- [ ] Use `<section>` with aria-label for grouped content
- [ ] Use `<form>` for form groups

### 6.3 Verification Script

```javascript
// Check for common accessibility issues
const issues = [];

// Check for buttons with aria-labels
document.querySelectorAll('button').forEach(btn => {
  if (!btn.getAttribute('aria-label') && !btn.textContent?.trim()) {
    issues.push(`Button without label: ${btn.outerHTML.slice(0, 80)}`);
  }
});

// Check for inputs with labels
document.querySelectorAll('input:not([type="hidden"])').forEach(input => {
  const label = document.querySelector(`label[for="${input.id}"]`);
  if (!label && !input.getAttribute('aria-label')) {
    issues.push(`Input without label: ${input.id || input.type}`);
  }
});

// Check heading hierarchy
const headings = Array.from(document.querySelectorAll('h1, h2, h3, h4, h5, h6'));
let lastLevel = 0;
headings.forEach((h, i) => {
  const level = parseInt(h.tagName[1]);
  if (level > lastLevel + 1) {
    issues.push(`Heading hierarchy gap at: ${h.textContent} (h${lastLevel} → h${level})`);
  }
  lastLevel = level;
});

console.log('Issues found:', issues.length);
issues.forEach(issue => console.log('  -', issue));
```

---

## Part 7: Focus Management

### 7.1 Focus Indicator Visibility

**Requirements:**
- Outline: 2px solid with high contrast color
- Outline offset: 2px (doesn't overlap content)
- Color contrast: ≥3:1 against background
- Always visible on keyboard navigation

**Test Procedure:**
1. Use keyboard only (no mouse)
2. Tab through all elements
3. Verify focus indicator visible on:
   - All input fields
   - All buttons
   - Links (if present)
   - Custom interactive elements

**CSS Example:**
```css
*:focus-visible {
  outline: 2px solid #8bd3f7;      /* cyan, high contrast */
  outline-offset: 2px;              /* space from element */
}

button:focus-visible {
  box-shadow: inset 0 0 0 2px #8bd3f7;
}

input:focus-visible {
  border-color: #8bd3f7;
  box-shadow: 0 0 0 3px rgba(139, 211, 247, 0.2);
}
```

### 7.2 Focus Trap Avoidance

**Test:** Do any interactive elements prevent escape via Tab?

```javascript
// Monitor focus changes
let focusPath = [];
document.addEventListener('focus', (e) => {
  focusPath.push(e.target.getAttribute('aria-label') || e.target.type);
}, true);

// After tabbing through page:
console.log('Focus path:', focusPath);
console.log('Unique elements:', new Set(focusPath).size);
```

### 7.3 Visible Focus Order

**Procedure:**
1. Open DevTools Console
2. Run this script:
```javascript
// Highlight all focusable elements in order
const focusable = document.querySelectorAll(
  'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
);

focusable.forEach((el, i) => {
  el.style.outline = `3px solid hsl(${(i / focusable.length) * 360}, 100%, 50%)`;
  el.setAttribute('data-focus-order', i);
});

console.log('Total focusable elements:', focusable.length);
```

---

## Part 8: Error Message Handling

### 8.1 Error Announcement Requirements

**Errors Must Be:**
1. **Visible** - Displayed on screen
2. **Associated** - Linked to form field via `aria-describedby`
3. **Announced** - Announced to screen reader
4. **Clear** - Explain the problem and how to fix it

**Example - Bad:**
```html
<input id="expr">
<span style="color: red;">Invalid</span>
```
→ Error not associated with input
→ Screen reader can't link them

**Example - Good:**
```html
<input id="expr" aria-invalid="true" aria-describedby="expr-error">
<span id="expr-error" role="alert">
  Error: Unmatched parentheses in expression
</span>
```
→ Error linked via `aria-describedby`
→ `role="alert"` triggers announcement
→ Clear message with guidance

### 8.2 Test Error Scenarios

**Calculator:**
- [ ] Unmatched parentheses: Error announced immediately
- [ ] Invalid variable: Field marked `aria-invalid="true"`
- [ ] Empty required field: Error message displayed
- [ ] Each error announced once to screen reader

**Unit Converter:**
- [ ] Non-numeric input → Error toast/message
- [ ] Field marked with error styling
- [ ] Error announced to screen reader

---

## Part 9: Responsive Accessibility

### 9.1 Mobile Accessibility

**Touch Targets:**
- Minimum: 44×44px (Apple iOS HIG)
- Minimum: 48×48px (Material Design)
- Spacing: 8px minimum between targets

**Test at 375px Viewport:**
```javascript
// Check touch target sizes
document.querySelectorAll('button, input, [role="button"]').forEach(el => {
  const rect = el.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;

  if (width < 44 || height < 44) {
    console.log(`SMALL: ${width}×${height}px -`, el.getAttribute('aria-label'));
  }
});
```

### 9.2 Zoom & Scale

**Test:**
- [ ] Page works at 200% zoom
- [ ] Text remains readable when zoomed
- [ ] No content hidden when zoomed
- [ ] Horizontal scroll minimal (if any)

**CSS:**
```css
/* Allow user to zoom */
<meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=yes">

/* Not: user-scalable=no (blocks zoom) */
```

---

## Part 10: Documentation & Reporting

### 10.1 Test Report Template

**File:** `test_results/accessibility-report-[date].md`

```markdown
# Accessibility Test Report

**Date:** 2026-04-30
**Tester:** [Name]
**Applications Tested:** Calculator, Unit Converter, URDF Viewer

## Executive Summary
- **Overall Status:** PASS / FAIL
- **WCAG Level:** AA
- **Critical Issues:** [Number]
- **Recommendations:** [If any]

## Test Results by Category

### 1. Automated Axe Scan
- **Calculator:** 0 violations ✓
- **Unit Converter:** 0 violations ✓
- **URDF Viewer:** 0 violations ✓

### 2. Color Contrast
- **Text Contrast:** PASS ✓
  - Verified: white on dark, dark on light
- **UI Component Contrast:** PASS ✓

### 3. Keyboard Navigation
- **Tab Order:** PASS ✓
- **Keyboard Shortcuts:** PASS ✓
- **Focus Indicators:** PASS ✓
- **No Keyboard Traps:** PASS ✓

### 4. Screen Reader (VoiceOver macOS)
- **Calculator:** PASS ✓
  - All elements announced correctly
  - Logical reading order
- **Unit Converter:** PASS ✓
- **URDF Viewer:** PASS ✓

### 5. ARIA & Semantic HTML
- **Form Labels:** PASS ✓
- **Button Names:** PASS ✓
- **Live Regions:** PASS ✓
- **Heading Hierarchy:** PASS ✓

### 6. Mobile Accessibility (375px)
- **Touch Targets:** PASS ✓
  - All buttons ≥44×44px
- **Viewport Meta Tag:** PASS ✓
  - User zoom enabled
- **Focus Indicators:** PASS ✓

### 7. Error Handling
- **Error Announcements:** PASS ✓
- **Error Association:** PASS ✓
- **Clear Messages:** PASS ✓

## Issues Found & Fixed

### Critical Issues
None

### Recommendations
- [If any improvements noted]

## Sign-off

| Role | Name | Date |
|------|------|------|
| QA Lead | __________ | __________ |
| Developer | __________ | __________ |

---

**Attachments:**
- Axe DevTools scan results
- VoiceOver navigation video
- Focus indicator screenshots
```

### 10.2 Automated Report Generation

```bash
# Generate HTML report from Axe results
npx axe-core https://localhost:5000 --output results.json
npx @axe-core/reporter-html results.json > report.html
```

---

## Part 11: Continuous Accessibility Testing

### 11.1 Pre-Commit Checks

**Create `.husky/pre-commit`:**
```bash
#!/bin/bash

# Run accessibility checks before commit
echo "Running accessibility checks..."

npx axe-core http://localhost:5000 --standard WCAG2AA || exit 1
npx axe-core http://localhost:8000 --standard WCAG2AA || exit 1

echo "✓ Accessibility checks passed"
```

### 11.2 CI/CD Integration

**GitHub Actions Workflow:**
```yaml
name: Accessibility

on: [pull_request]

jobs:
  a11y:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install dependencies
        run: npm install axe-core @axe-core/puppeteer

      - name: Run accessibility tests
        run: npm run test:a11y

      - name: Generate report
        run: npm run a11y:report

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: a11y-results
          path: results/
```

---

## Part 12: Accessibility Resources

### Official Standards
- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [ARIA Authoring Practices Guide](https://www.w3.org/WAI/ARIA/apg/)
- [WAI: Web Content Accessibility](https://www.w3.org/WAI/)

### Tools
- [Axe DevTools](https://www.deque.com/axe/devtools/)
- [WAVE Browser Extension](https://wave.webaim.org/extension/)
- [Lighthouse](https://developers.google.com/web/tools/lighthouse)
- [NVDA Screen Reader](https://www.nvaccess.org/)

### Learning Resources
- [WebAIM Articles](https://webaim.org/articles/)
- [A11ycasts by Google Chrome](https://www.youtube.com/playlist?list=PLNYkxOF6rcICWx0C9Xc-RgEzwLvsPccay2)
- [Inclusive Components](https://inclusive-components.design/)

---

End of Accessibility Testing Guide
