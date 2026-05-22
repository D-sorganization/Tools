# Phase 2.1 Responsive Implementation - Testing Checklist

**Status:** Ready for QA
**Date:** April 30, 2026
**Issue:** #2408

---

## Pre-Testing Setup

### Environment Setup

- [ ] Navigate to: `/home/user/Tools/src/data_processing/data_processor/web`
- [ ] Run: `npm install` (if dependencies not already installed)
- [ ] Run: `npm run dev`
- [ ] Open browser to: `http://localhost:5173` (check console for actual port)

### Chrome DevTools Setup

- [ ] Open Chrome DevTools: `F12`
- [ ] Enable Device Emulation: `Ctrl+Shift+M` (Cmd+Shift+M on Mac)
- [ ] Select "Edit device" for custom viewports if needed

---

## Test Case 1: Mobile (375px - iPhone SE)

### Setup

- [ ] Device: iPhone SE
- [ ] Width: 375px, Height: 667px
- [ ] Rotate: Portrait

### Viewport Adaptation

- [ ] Hamburger menu visible (Menu icon in header)
- [ ] Sidebar not visible by default
- [ ] Click hamburger → sidebar slides from left (smooth animation)
- [ ] Sidebar width ~300px, doesn't exceed screen
- [ ] Click outside sidebar → closes
- [ ] Layout properly adapts (no text overflow)

### Header & Navigation

- [ ] Data Processor icon visible
- [ ] Title hidden (shows on tablet+)
- [ ] Subtitle hidden (responsive text)
- [ ] Signal count shows as "5S" format (not full text)
- [ ] Hamburger button easily tappable (48x48px+)

### Main Content Area

- [ ] Charts display with reduced height (~300px)
- [ ] No horizontal scroll needed
- [ ] Tab buttons span full width (Chart | Table)
- [ ] Tab text shows icons + abbreviations (no full text)
- [ ] Chart legend positioned horizontally below (not overlapping)
- [ ] Chart margins are compact (top/bottom 25px)

### Sidebar (Opened)

- [ ] File upload section visible
- [ ] Tab buttons visible (Signals | Adv | Rsp | Tm)
- [ ] Tab text abbreviated (no overflow)
- [ ] Filter panel responsive
- [ ] All buttons easily tappable
- [ ] Closes when switching tabs (UX convenience)

### Touch Targets

- [ ] All buttons: min 48x48px
- [ ] Tab buttons: 48px height
- [ ] Hamburger: 48x48px
- [ ] No adjacent buttons too close (<8px gap minimum)

### Typography & Readability

- [ ] Text readable without zoom
- [ ] Font size >= 16px on inputs (mobile OS keyboard)
- [ ] Line spacing adequate
- [ ] Color contrast meets WCAG AA (4.5:1+)

### Performance

- [ ] Page loads quickly (<2s on 4G)
- [ ] No layout shift when sidebar opens
- [ ] Animations smooth (60fps)
- [ ] No console errors

---

## Test Case 2: Tablet (768px - iPad)

### Setup

- [ ] Device: iPad
- [ ] Width: 768px, Height: 1024px
- [ ] Rotate: Portrait

### Layout Changes

- [ ] Hamburger menu NOT visible (removed at md: breakpoint)
- [ ] Sidebar visible on left (~25% width)
- [ ] Main content takes remaining width (~75%)
- [ ] Right sidebar NOT visible (hidden until 1024px)
- [ ] No drawer behavior (sidebar always on-screen)

### Sidebar Visibility

- [ ] Sidebar fixed position, doesn't scroll with content
- [ ] All sidebar content accessible without toggle
- [ ] Sidebar tabs working (Signals, Advanced, Resample, Time)
- [ ] Panel content responsive

### Charts

- [ ] Charts responsive to available width
- [ ] Legend horizontal positioning (bottom of chart)
- [ ] Margins adjusted for tablet
- [ ] Mode bar still hidden (touch device)
- [ ] Chart height responsive (~350px)

### Content Area

- [ ] Charts display properly in available space
- [ ] Tab buttons properly sized
- [ ] Text labels fully visible (not abbreviated)
- [ ] Full feature set available

### Tab Navigation

- [ ] Chart View tab works, displays chart
- [ ] Table View tab works, displays data table
- [ ] Switching tabs smooth, no layout shift

### Touch Targets

- [ ] All elements at least 44px (WCAG guideline)
- [ ] Most elements 48px (better for touch)
- [ ] No touch target crowding

---

## Test Case 3: Desktop (1024px+ - Laptop)

### Setup

- [ ] Device: Desktop/Laptop
- [ ] Width: 1024px or larger
- [ ] Height: 768px or larger
- [ ] Rotate: Landscape

### Three-Column Layout

- [ ] Left sidebar visible (col-span-3, ~25%)
- [ ] Main content visible (col-span-6, ~50%)
- [ ] Right sidebar visible (col-span-3, ~25%)
- [ ] All columns properly aligned
- [ ] Gaps between columns visible (gap-6 = 1.5rem)

### No Hamburger Menu

- [ ] Hamburger button NOT visible
- [ ] Sidebar permanently visible
- [ ] No drawer behavior needed

### Charts

- [ ] Full quality rendering at desktop size
- [ ] Mode bar visible with all controls
- [ ] Legend positioned vertically on right
- [ ] Spacious margins (t:40, r:20, b:40, l:60)
- [ ] Charts take full available width

### Right Sidebar

- [ ] Right panel visible and functional
- [ ] Tabs visible (Stats, Analytics, Trendline, Export, Help)
- [ ] All panels accessible
- [ ] Proper spacing maintained

### Full Feature Set

- [ ] All features accessible
- [ ] No content hidden
- [ ] Optimal viewing experience
- [ ] Advanced features available

---

## Responsive Behavior Tests

### Window Resize Test

1. Start at 1024px width
2. Slowly resize down to 375px

   - [ ] Right sidebar disappears at <1024px
   - [ ] Hamburger appears at <768px
   - [ ] Layout smoothly adapts
   - [ ] No layout shift or jump
   - [ ] Charts resize smoothly
   - [ ] Text remains readable

3. Resize back to 1024px
   - [ ] Right sidebar reappears
   - [ ] Hamburger disappears
   - [ ] Layout returns to three-column
   - [ ] Smooth animation throughout

### Orientation Change Test

1. Start portrait (375x667)
2. Rotate to landscape (667x375)

   - [ ] Content reflows properly
   - [ ] Charts resize
   - [ ] Sidebar behavior unchanged
   - [ ] No content hidden
   - [ ] All interactive elements accessible

3. Rotate back to portrait
   - [ ] Content reflowed back
   - [ ] Original layout restored
   - [ ] Smooth transition

---

## Feature Tests

### File Upload

- [ ] [ ] Mobile (375px): Can load file through sidebar
- [ ] [ ] Tablet (768px): File upload working
- [ ] [ ] Desktop (1024px): File upload working
- [ ] [ ] File size validation works
- [ ] [ ] File type validation works
- [ ] [ ] Success notification appears

### Signal Selection

- [ ] [ ] Signals list displays on all viewports
- [ ] [ ] Can select/deselect signals
- [ ] [ ] Chart updates when signals selected
- [ ] [ ] List responsive (no overflow)

### Chart Interaction

- [ ] [ ] Charts render data correctly
- [ ] [ ] Desktop: Mode bar visible, can interact
- [ ] [ ] Mobile: Mode bar hidden, minimal controls
- [ ] [ ] Legend not overlapping data
- [ ] [ ] Hover tooltips work (desktop)
- [ ] [ ] Touch interaction works (mobile/tablet)

### Filter Panel

- [ ] [ ] Filter options display
- [ ] [ ] Parameter inputs responsive
- [ ] [ ] Validation errors show with red border
- [ ] [ ] Apply button disabled on errors
- [ ] [ ] Apply button works when valid
- [ ] [ ] Error messages clear and helpful

### Tab Switching

- [ ] [ ] Chart View tab works
- [ ] [ ] Table View tab works
- [ ] [ ] Switching smooth on all viewports
- [ ] [ ] No layout shift when switching
- [ ] [ ] Content loads quickly

### Sidebar Toggle (Mobile)

- [ ] [ ] Hamburger opens sidebar
- [ ] [ ] Sidebar slides smoothly from left
- [ ] [ ] Overlay appears behind sidebar
- [ ] [ ] Click overlay closes sidebar
- [ ] [ ] Click hamburger again closes sidebar
- [ ] [ ] No animation jank

---

## Lighthouse Audit

### Mobile Audit (375px)

1. Open DevTools → Lighthouse
2. Select "Mobile" mode
3. Make sure "Clear storage" is UNCHECKED
4. Click "Analyze page load"

#### Expected Results:

- [ ] Performance: >= 75 (target: 80+)
- [ ] Accessibility: >= 90
- [ ] Best Practices: >= 85
- [ ] SEO: >= 85

#### Core Web Vitals:

- [ ] LCP (Largest Contentful Paint): < 2.5s (GREEN)
- [ ] FID (First Input Delay): < 100ms (GREEN)
- [ ] CLS (Cumulative Layout Shift): < 0.1 (GREEN)

#### Key Metrics to Document:

- [ ] Viewport configured properly: ✅
- [ ] Touch-friendly size: ✅
- [ ] Responsive design: ✅
- [ ] No layout shift: ✅

**Screenshot Required:** Yes, save Lighthouse result

### Desktop Audit (1024px+)

1. Select "Desktop" mode
2. Run same audit
3. Compare metrics

#### Expected Results:

- [ ] Performance: >= 85
- [ ] Accessibility: >= 95
- [ ] Best Practices: >= 90
- [ ] SEO: >= 90

---

## Accessibility Tests

### Keyboard Navigation

- [ ] [ ] Tab key navigates through all controls
- [ ] [ ] Focus visible on all interactive elements
- [ ] [ ] Enter/Space activates buttons
- [ ] [ ] Can close sidebar with Escape (if implemented)

### Color Contrast

- [ ] [ ] Text on dark background: >= 4.5:1
- [ ] [ ] Active tab indicator: >= 3:1
- [ ] [ ] Error messages: Visible and contrasty
- [ ] [ ] Icons: Proper contrast

### Screen Reader (if available)

- [ ] [ ] Page title announced: "Data Processor"
- [ ] [ ] Major sections announced as landmarks
- [ ] [ ] Button labels clear
- [ ] [ ] Chart title announced
- [ ] [ ] Form labels associated with inputs

### ARIA Labels

- [ ] [ ] Hamburger button: "Toggle sidebar"
- [ ] [ ] Close button: "Close sidebar"
- [ ] [ ] Tab buttons: Properly labeled
- [ ] [ ] Form inputs: Associated labels

---

## Cross-Browser Testing

### Chrome

- [ ] [ ] All tests pass on latest Chrome
- [ ] [ ] Lighthouse audit works
- [ ] [ ] DevTools emulation accurate

### Firefox (if available)

- [ ] [ ] Responsive layout works
- [ ] [ ] Charts render properly
- [ ] [ ] Touch targets visible

### Safari (if available)

- [ ] [ ] iOS Safari responsive
- [ ] [ ] Charts render
- [ ] [ ] Performance acceptable

---

## Actual Device Testing (If Possible)

### iOS Device

- [ ] [ ] Load on iPhone (6S or later)
- [ ] [ ] Safari browser works
- [ ] [ ] Responsive layout adapts
- [ ] [ ] Touch interaction smooth
- [ ] [ ] Notch support (viewport-fit=cover)

### Android Device

- [ ] [ ] Load on Android phone
- [ ] [ ] Chrome browser works
- [ ] [ ] Responsive layout adapts
- [ ] [ ] Touch interaction smooth
- [ ] [ ] Status bar theme correct (dark)

---

## Performance Stress Tests

### Large Dataset

1. Load CSV with 10,000+ rows
   - [ ] [ ] Charts render without lag
   - [ ] [ ] Scrolling smooth on mobile
   - [ ] [ ] Memory usage reasonable
   - [ ] [ ] Charts responsive on resize

### Multiple Signals

1. Select 10+ signals
   - [ ] [ ] Charts render all signals
   - [ ] [ ] Legend displays properly
   - [ ] [ ] No performance degradation
   - [ ] [ ] Still responsive to interaction

### Extended Session

1. Use app for 5+ minutes
   - [ ] [ ] No memory leaks
   - [ ] [ ] No slowdown over time
   - [ ] [ ] All features remain responsive
   - [ ] [ ] Animations still smooth

---

## Edge Case Tests

### Extreme Viewports

- [ ] [ ] 320px (small phone): Content fits
- [ ] [ ] 480px: Readable and usable
- [ ] [ ] 1920px (4K): Proper scaling, not too spread out
- [ ] [ ] 2560px: Still usable

### Orientation Edge Cases

- [ ] [ ] 375x812 (iPhone X portrait): Notch supported
- [ ] [ ] 812x375 (iPhone X landscape): Notch supported
- [ ] [ ] iPad landscape (1024x768): Three-column visible
- [ ] [ ] Rotating rapidly: No visual glitches

### Interaction Edge Cases

- [ ] [ ] Double-tap on mobile: Zoom disabled (viewport-fit)
- [ ] [ ] Pinch zoom on charts: Smooth
- [ ] [ ] Sidebar open, then resize: Closes properly
- [ ] [ ] Rapid tab switching: All animations complete

---

## Issue Reporting Template

If issues found, use this format:

```
## Issue: [Title]

**Severity:** [Critical/High/Medium/Low]
**Viewport:** [375px/768px/1024px/other]
**Browser:** [Chrome/Firefox/Safari/Other]
**Steps to Reproduce:**
1. Load app at [viewport]
2. [Action]
3. [Expected vs Actual]

**Expected Behavior:**
[What should happen]

**Actual Behavior:**
[What actually happens]

**Screenshot/Video:**
[Attach if possible]

**Environment:**
- Device: [iPhone/iPad/Desktop/etc]
- OS: [iOS/Android/Windows/Mac/etc]
- Browser: [Chrome/Safari/Firefox]
- Version: [Browser version]
```

---

## Sign-Off

### Test Completion Checklist

- [ ] All test cases executed
- [ ] No critical issues found
- [ ] Lighthouse audit passed (score >= 80 on mobile)
- [ ] Accessibility verified
- [ ] Cross-browser tested
- [ ] Performance acceptable
- [ ] Documentation complete

### QA Sign-Off

- [ ] Tester Name: **\*\***\_\_\_\_**\*\***
- [ ] Date: **\*\***\_\_\_\_**\*\***
- [ ] Status: ☐ PASS ☐ FAIL ☐ CONDITIONAL

### Notes:

```
[Document any issues, observations, or recommendations]
```

---

**End of Testing Checklist**

This checklist ensures comprehensive testing of all Phase 2.1 responsive implementation features.
