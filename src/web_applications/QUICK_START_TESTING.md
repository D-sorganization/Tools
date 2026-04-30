# Phase 2.2 - Quick Start Testing Guide

## Start Here

This is your entry point for Phase 2.2 Frontend Polish & Integration testing. Complete these steps to begin.

---

## 5-Minute Setup

### 1. Verify Test Documents Are Available

Check that these files exist in `/web_applications/`:
- ✓ MOBILE_TESTING_CHECKLIST.md
- ✓ TESTING_INTEGRATION_GUIDE.md
- ✓ ACCESSIBILITY_TESTING_GUIDE.md
- ✓ RESPONSIVE_DESIGN_DECISIONS.md
- ✓ PHASE_2_2_IMPLEMENTATION_STATUS.md (this directory)

If any are missing, create them first.

### 2. Set Up Test Environment

**Create test results directory:**
```bash
cd /home/user/Tools/src/web_applications
mkdir -p test_results/{reports,screenshots,axe-results}
cd test_results
touch TESTING_LOG.md
```

### 3. Start Development Servers

**Terminal 1 - Aurora Calculator:**
```bash
cd /home/user/Tools/src/web_applications/calculator
flask --app webapp run
# Opens on http://localhost:5000
```

**Terminal 2 - Unit Converter:**
```bash
cd /home/user/Tools/src/web_applications/unit_converter/unit-converter-app
python -m http.server 8000
# Opens on http://localhost:8000
```

**Terminal 3 - URDF Viewer (if testing):**
```bash
cd /home/user/Tools/src/web_applications/urdf_viewer
uvicorn app:app --reload
# Opens on http://localhost:8000 or configured port
```

### 4. Prepare Browser DevTools

**Chrome/Edge:**
1. Open each application in a browser tab
2. Press `Ctrl+Shift+M` (Cmd+Shift+M on Mac) to enable mobile simulation
3. Install Axe DevTools extension:
   - Chrome Web Store → Search "axe DevTools"
   - Click "Add to Chrome"

**Firefox:**
1. Install Axe DevTools:
   - Firefox Add-ons → Search "axe DevTools"
   - Click "Add to Firefox"

---

## Testing Checklist (Quick Version)

### Phase 1: Mobile Responsive (375px) - 1 Hour

**For Each App (Calculator, Unit Converter, URDF):**

1. Open DevTools (F12)
2. Toggle mobile mode (Ctrl+Shift+M)
3. Select iPhone SE preset (375px)
4. Check:
   - [ ] No horizontal scroll (scroll width = viewport width)
   - [ ] All content visible and readable
   - [ ] Buttons/inputs ≥44px tall
   - [ ] Text ≥12px size
5. Take screenshot
6. Repeat for 768px and 1024px

### Phase 2: Accessibility Audit - 30 minutes

**For Each App:**

1. Open app in browser
2. Open DevTools → Axe DevTools tab
3. Click "Scan ENTIRE PAGE"
4. Record results:
   - Violations count
   - Critical issues
5. Fix any violations found
6. Save screenshot of results

### Phase 3: Keyboard Navigation - 30 minutes

**For Each App:**

1. Open app
2. Close DevTools (don't distract)
3. Press Tab key repeatedly
4. Verify:
   - [ ] Every interactive element receives focus
   - [ ] Focus order is logical
   - [ ] Focus indicator is visible
   - [ ] No elements are unreachable
5. Document any issues

### Phase 4: Error Handling - 30 minutes

**For Each App (where applicable):**

1. Try invalid inputs
2. Verify error message appears
3. Check error is visible and readable
4. Verify it's announced to screen reader (if testing)

**Test Cases:**
- **Calculator:** Unmatched parentheses `((1+2)`
- **Unit Converter:** Non-numeric input `abc`
- **URDF Viewer:** Invalid file upload

### Phase 5: Documentation - 30 minutes

1. Create test report for each app
2. Compile all screenshots
3. Record findings
4. Update TESTING_LOG.md

---

## Testing Command Reference

### Run Mobile Test (375px)
```javascript
// Paste in DevTools Console
const shell = document.querySelector('.calculator-shell');
console.log('Width:', shell.offsetWidth);
console.log('Has horizontal scroll:', document.documentElement.scrollWidth > document.documentElement.clientWidth);
```

### Run Accessibility Check
```javascript
// Check for unlabeled buttons
const unlabeled = [];
document.querySelectorAll('button').forEach(btn => {
  if (!btn.getAttribute('aria-label') && !btn.textContent?.trim()) {
    unlabeled.push(btn);
  }
});
console.log('Unlabeled buttons:', unlabeled.length, unlabeled);
```

### Check Touch Target Sizes
```javascript
// Find buttons smaller than 44x44px
document.querySelectorAll('button').forEach(btn => {
  const w = btn.offsetWidth;
  const h = btn.offsetHeight;
  if (w < 44 || h < 44) {
    console.log(`SMALL: ${w}x${h}px`, btn.getAttribute('aria-label'));
  }
});
```

### Monitor Focus
```javascript
// Log all focus events
document.addEventListener('focus', (e) => {
  console.log('Focused:', e.target.getAttribute('aria-label') || e.target.type || e.target.textContent?.slice(0, 20));
}, true);
```

---

## Test Report Template

Create a file: `test_results/reports/[app]-report-[date].md`

```markdown
# Test Report: [App Name]

**Date:** [YYYY-MM-DD]
**Tester:** [Your Name]
**Duration:** [X hours]

## Results Summary

| Category | Status | Issues |
|----------|--------|--------|
| Responsive (375px) | PASS/FAIL | 0 |
| Responsive (768px) | PASS/FAIL | 0 |
| Responsive (1024px) | PASS/FAIL | 0 |
| Accessibility (Axe) | PASS/FAIL | 0 violations |
| Keyboard Navigation | PASS/FAIL | 0 traps |
| Touch Targets | PASS/FAIL | 0 < 44px |
| Error Handling | PASS/FAIL | 0 missing |

## Details

### Responsive Testing
- 375px: [Details]
- 768px: [Details]
- 1024px: [Details]

### Accessibility
- Axe violations: 0
- Critical issues: None
- Focus indicators: Visible on all elements
- Keyboard: All elements reachable

### Issues Found
1. [Issue 1 - if any]
2. [Issue 2 - if any]

## Screenshots
- [app]-375px.png
- [app]-768px.png
- [app]-1024px.png
- [app]-axe-results.png

## Overall Status: PASS / FAIL

**Sign-off:** _________________ Date: _______
```

---

## Troubleshooting

### Issue: DevTools Mobile View Doesn't Match Real Device
**Solution:** Test on actual device using same WiFi network
- Find your IP: `ipconfig getifaddr en0` (Mac) or `ipconfig` (Windows)
- Visit: `http://[your-ip]:5000`

### Issue: Font Too Small on Mobile
**Check:** Is font size ≥12px?
```javascript
window.getComputedStyle(document.body).fontSize
```

### Issue: Buttons Hard to Tap
**Check:** Are buttons ≥44×44px?
```javascript
document.querySelectorAll('button').forEach(btn => {
  console.log(btn.offsetWidth, 'x', btn.offsetHeight);
});
```

### Issue: Horizontal Scroll on Mobile
**Check:** Is page width = viewport width?
```javascript
document.documentElement.scrollWidth <= document.documentElement.clientWidth
```

### Issue: Focus Outline Not Visible
**Check:** CSS for `:focus-visible`
```javascript
window.getComputedStyle(document.activeElement, ':focus-visible').outline
```

---

## Tool Checklist

Before starting, ensure you have:
- [ ] Chrome/Chromium browser with Axe extension
- [ ] Firefox browser with Axe extension (optional)
- [ ] Mobile device (iPhone/Android) connected to WiFi
- [ ] Terminal access to start servers
- [ ] Text editor for documentation
- [ ] Screenshot tool (built into OS or snipping tool)

---

## Critical Paths (Must Complete)

### Minimum Testing (4 hours)
1. ✓ Responsive test at 375px only
2. ✓ Axe accessibility scan
3. ✓ Keyboard tab test
4. ✓ Error message test

### Full Testing (8 hours)
1. ✓ Responsive test all viewports (375, 768, 1024px)
2. ✓ Axe accessibility scan + fixes
3. ✓ Keyboard navigation + focus test
4. ✓ Error handling + screen reader
5. ✓ Real device testing
6. ✓ Full documentation

---

## Next Steps

1. **Read:** Start with MOBILE_TESTING_CHECKLIST.md for detailed procedures
2. **Setup:** Run the 5-minute setup above
3. **Test:** Follow the checklist for each application
4. **Document:** Record findings in test_results/
5. **Report:** Compile results and share findings

---

## Support & Questions

For detailed guidance, see:
- **Mobile Layout Issues:** RESPONSIVE_DESIGN_DECISIONS.md
- **Accessibility Details:** ACCESSIBILITY_TESTING_GUIDE.md
- **Integration Procedures:** TESTING_INTEGRATION_GUIDE.md
- **Overall Status:** PHASE_2_2_IMPLEMENTATION_STATUS.md

---

## Estimated Time Breakdown

| Task | Time | Status |
|------|------|--------|
| Setup (servers, tools) | 15 min | [  ] |
| Calculator Testing | 1.5 hrs | [  ] |
| Unit Converter Testing | 1.5 hrs | [  ] |
| URDF Viewer Testing | 1 hr | [  ] |
| Documentation | 1 hr | [  ] |
| **Total** | **5.5 hrs** | |

---

Good luck! Start with the MOBILE_TESTING_CHECKLIST.md next.
