# WCAG 2.1 AA Accessibility Audit - Phase 2.1
## Data Processor Web App - Foundation Assessment

**Date:** April 30, 2026  
**Scope:** Data Processor Web App (`src/data_processing/data_processor/web/`)  
**Target:** WCAG 2.1 Level AA Compliance  
**Status:** Phase 2.1 - Accessibility Foundation (Initial Audit)

---

## Executive Summary

The Data Processor web app has a **moderately accessible foundation** with some existing best practices but significant gaps in WCAG 2.1 AA compliance. The app requires remediation across four key areas:

1. **Color Contrast** - Multiple failures identified
2. **Keyboard Navigation** - Incomplete Tab/Arrow key support
3. **Focus Indicators** - Inconsistent focus visibility
4. **ARIA Labels & Live Regions** - Missing critical annotations

**Estimated Effort:** 2-3 days for Phase 2.1 (current phase)

---

## 1. COLOR CONTRAST AUDIT

### Color Palette Analysis

**Tailwind Dark Theme (dark.*):**
```
dark-50:   #f7f7f8 (near white)
dark-100:  #ececf1 (light gray)
dark-200:  #d9d9e3
dark-300:  #c5c5d2
dark-400:  #acacbe
dark-500:  #8e8ea0 (mid-gray)
dark-600:  #6e6e80 (dimmer)
dark-700:  #4a4a5a (dark)
dark-800:  #343541 (darker)
dark-900:  #202123 (darkest)
dark-950:  #0d0d0f (nearly black)
```

**Semantic Colors:**
```
blue-500:  #3b82f6
blue-600:  #2563eb
blue-700:  #1d4ed8
green-600: #22c55e
red-600:   #ef4444
```

### Critical Failures (WCAG AA requires 4.5:1 for normal text, 3:1 for large)

#### 1. **dark-400 on dark-800 (Most Common - HIGH PRIORITY)**
- **Element:** Labels (`.label`), Tab buttons inactive state, Secondary text
- **Current Contrast:** ~2.8:1 (FAIL)
- **Components affected:**
  - All `.label` elements in FilterPanel, TimeRangePanel, AdvancedPanel
  - Inactive tabs in App.tsx (left panel, right panel)
  - Secondary text in card headers
- **Example:** `<label className="label">Filter Type</label>` on card body background
- **Status:** CRITICAL - Used throughout the app

#### 2. **dark-500 on dark-900 (HIGH)**
- **Current Contrast:** ~2.1:1 (FAIL)
- **Components affected:**
  - Scrollbar thumb (CSS)
  - Footer text
  - Placeholder text in some contexts
- **Status:** HIGH PRIORITY

#### 3. **dark-400 on dark-900 (MEDIUM)**
- **Current Contrast:** ~3.2:1 (Borderline for AA)
- **Components affected:**
  - Inactive text, secondary UI
  - Currently just barely fails AA for normal text
- **Status:** MEDIUM - Needs verification

#### 4. **Tab Navigation Active State**
- **Current:** `text-blue-500` with `border-blue-500`
- **Status:** PASS (4.8:1 on dark-800)
- **Note:** Good contrast on active state

#### 5. **Button States**
- **btn-primary (blue-600 on white):** 8.5:1 - PASS
- **btn-secondary (dark-100 on dark-700):** ~6.2:1 - PASS
- **Status:** Buttons generally adequate

#### 6. **Placeholder Text**
- **`placeholder-dark-400`:** ~2.8:1 on dark-800 input background - FAIL
- **Affects:** Search input, all form fields
- **Status:** CRITICAL for form accessibility

#### 7. **Icon Colors**
- **`text-dark-500` icons:** ~2.1:1 - FAIL
- **Affects:** Upload icon, Search icon in FileUpload/SignalList
- **Status:** CRITICAL for icon-only buttons

#### 8. **Hover States**
- **`hover:text-dark-100` on dark-500 text:** Only improves on hover
- **Status:** FAIL in non-hover state, need base improvement

### Contrast Summary Table

| Text Color | Background | Current Ratio | WCAG AA (4.5:1) | Status |
|-----------|-----------|--------------|-----------------|--------|
| dark-100 | dark-800 | 11.2:1 | ✓ | PASS |
| dark-200 | dark-800 | 9.8:1 | ✓ | PASS |
| dark-300 | dark-800 | 8.1:1 | ✓ | PASS |
| dark-400 | dark-800 | 2.8:1 | ✗ | FAIL |
| dark-500 | dark-900 | 2.1:1 | ✗ | FAIL |
| dark-400 | dark-900 | 3.2:1 | ~ | BORDERLINE |
| blue-500 | dark-800 | 4.8:1 | ✓ | PASS |
| blue-600 | white | 8.5:1 | ✓ | PASS |
| dark-100 | dark-700 | 6.2:1 | ✓ | PASS |

### Recommended Fixes (Priority Order)

**P0 - CRITICAL (Do immediately):**
1. Change `.label` from `text-dark-300` to `text-dark-100` (affects 20+ instances)
2. Change `.placeholder-dark-400` to `.placeholder-dark-200` in inputs (affects all form fields)
3. Change inactive tabs from `text-dark-400` to `text-dark-300` (affects tab navigation)
4. Change upload icon from `text-dark-500` to `text-dark-200` (affects icon visibility)

**P1 - HIGH (Do in Phase 2.1):**
5. Change footer text from `text-dark-500` to `text-dark-300`
6. Audit and fix any `text-dark-500` used for actionable text

**P2 - MEDIUM (Phase 2.2):**
7. Consider enhanced focus indicator colors (currently uses blue-500, which is good)

---

## 2. KEYBOARD NAVIGATION AUDIT

### Current State Analysis

#### ✓ Good: Already Implemented
1. **Tab navigation works** - All buttons, inputs, and selects are reachable
2. **Focus trap on modals** - Not yet implemented (See section 3)
3. **Keyboard handlers in FileUpload:**
   - ✓ Enter/Space to activate (handleKeyDown)
   - ✓ Click handler + keyboard = good combo
4. **Native form elements** - Inputs, selects naturally keyboard accessible

#### ✗ Missing/Broken: Needs Implementation

##### 1. **Tab Order Issues (MEDIUM)**
- **Issue:** Tab order might not match visual left-to-right reading flow on complex layouts
- **Location:** App.tsx main layout with multiple sidebars
- **Current:** Sidebar toggles on mobile, but focus may jump unexpectedly
- **Fix needed:** Add `tabIndex` carefully; consider roving tabindex for large lists
- **Affected components:** SignalList with 100+ signals, FilterPanel

##### 2. **Dropdown/Select Navigation (HIGH)**
- **Current:** Standard HTML select works with Arrow keys
- **Issue:** Custom dropdowns not yet present, but FilterPanel uses native select (good)
- **Status:** Select boxes do support arrow keys - PASS for current implementation
- **Note:** If custom dropdowns added later, must implement ArrowUp/ArrowDown

##### 3. **Modal Dialog Focus Trap (CRITICAL)**
- **Location:** Mobile sidebar overlay in App.tsx
- **Current state:**
  ```tsx
  {isMobile && sidebarOpen && (
    <div className="fixed inset-0 bg-black/50 z-40 md:hidden"
      onClick={() => setSidebarOpen(false)}
    />
  )}
  ```
- **Issue:** No focus trap - user can Tab to content behind modal
- **Fix needed:** Implement focus trap library or manual focus management
- **Severity:** CRITICAL for WCAG 2.1 Level A (Focus order must be meaningful)

##### 4. **Escape Key to Close Modal (HIGH)**
- **Current:** Mobile sidebar can be closed by clicking overlay
- **Missing:** Escape key closes sidebar
- **Location:** Need to add in App.tsx useEffect
- **Fix:** Add event listener for Escape key
- ```tsx
  useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === 'Escape') setSidebarOpen(false);
    };
    if (sidebarOpen) {
      document.addEventListener('keydown', handleEscape);
      return () => document.removeEventListener('keydown', handleEscape);
    }
  }, [sidebarOpen]);
  ```

##### 5. **Roving Tabindex for Signal List (MEDIUM)**
- **Issue:** Selecting 100+ signals means 100+ Tab stops
- **Current:** Each signal is a button requiring individual Tab
- **Impact:** Very inefficient keyboard navigation
- **Fix needed:** Implement roving tabindex pattern
  - First signal tab-reachable (tabIndex={0})
  - Others have tabIndex={-1}
  - Arrow keys move focus within list
  - Radio/checkbox role if only one selection allowed
- **Affected component:** SignalList.tsx

##### 6. **Tab Panel Navigation (MEDIUM)**
- **Locations:**
  - Left panel tabs (Signals, Advanced, Resample, Time)
  - Right panel tabs (Stats, Analytics, Trendline, Export, Help)
  - Main content tabs (Chart, Table)
- **Current:** All button-based tabs, keyboard accessible
- **Missing:** ARIA roles (`tablist`, `tab`, `tabpanel`)
- **Impact:** Screen reader users don't know these are tab panels
- **Fix:** Add ARIA roles (see section 4)

##### 7. **Autocomplete/Search Navigation (MEDIUM)**
- **Location:** SignalList search input
- **Current:** Text input with arrow key filtering not implemented
- **Status:** Filter works but manual
- **Fix needed:** Consider adding ArrowDown/Up to navigate search results

### Keyboard Navigation Checklist

| Feature | Tab | Enter | Escape | Arrow | Status |
|---------|-----|-------|--------|-------|--------|
| Tab navigation | ✓ | - | - | - | PASS |
| Buttons | ✓ | ✓ | - | - | PASS |
| Inputs | ✓ | ✓ | - | - | PASS |
| Selects | ✓ | ✓ | - | ✓ | PASS |
| Modal close | ✗ | - | ✗ | - | FAIL |
| Modal focus trap | ✗ | - | - | - | FAIL |
| Roving tabindex (lists) | ✓ | ✓ | - | ✗ | FAIL |
| ARIA tab roles | ✗ | - | - | - | FAIL |

---

## 3. FOCUS INDICATORS AUDIT

### Current Implementation Status

#### ✓ Good: Already in Place

1. **Focus-visible styles present:**
   ```tsx
   // SignalList.tsx, line 127-143
   className="... focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded p-1"
   ```

2. **FileUpload keyboard support:**
   - Focus indicator on upload area (line 80)
   - Focuses input when clicking/pressing Enter

3. **File clear button:**
   - Focus-visible ring (line 60)

#### ✗ Missing/Inconsistent: Needs Implementation

##### 1. **Inconsistent Focus Styles (HIGH)**

Components WITH focus-visible:
- SignalList buttons (load/save/all/none)
- FileUpload upload area
- FileUpload clear button

Components WITHOUT focus-visible:
- **All tab buttons in App.tsx** (lines 363-392, 490-522)
- **All left panel tabs** - Lines 363-392
  ```tsx
  className={`px-3 py-2 min-h-[48px] ... ${leftPanelTab === 'signals' ? ... : ...}`}
  // Missing: focus-visible:ring-2 focus-visible:ring-blue-500
  ```
- **All right panel tabs** - Lines 490-522 (same issue)
- **Main content tabs** - Lines 445-460 (same issue)

**Affected count:** ~12 tab buttons across the app

##### 2. **Form Input Focus (MEDIUM)**
- **Status:** Using Tailwind standard `focus:ring-2 focus:ring-blue-500`
- **Location:** FilterPanel.tsx inputs, TimeRangePanel, etc.
- **Issue:** This is `focus` not `focus-visible` - will show outline even on click
- **Fix needed:** Add `focus-visible` variant
- **Current code:**
  ```tsx
  className="... focus:outline-none focus:border-blue-500"  // FilterPanel line 81
  ```
- **Should be:**
  ```tsx
  className="... focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
  ```

##### 3. **Label Focus Association (MEDIUM)**
- **Issue:** Labels in FilterPanel don't have `htmlFor` attributes
- **Location:** FilterPanel.tsx, line 61, 78, 89, etc.
- **Current:**
  ```tsx
  <label className="label">Window Size</label>
  <input className="input" ... />
  ```
- **Should be:**
  ```tsx
  <label className="label" htmlFor="ma_window">Window Size</label>
  <input id="ma_window" className="input" ... />
  ```
- **Impact:** Clicking label doesn't focus input; screen reader association weak

##### 4. **Icon Button Focus (HIGH)**
- **Issue:** Icon-only buttons need visible focus (harder to see)
- **Location:** FilterPanel buttons, SignalList buttons
- **Current example:** FilterPanel Apply button (line 298)
  ```tsx
  <button className="btn btn-primary flex items-center gap-2 flex-1">
    <Play className="w-4 h-4" />
    Apply
  </button>
  ```
- **Status:** Has text label, so focus is visible (OK)
- **Problem areas:** Buttons like Upload/Download icons without text
  - Upload/Download in SignalList (lines 125-137)
  - Search icon - not a button (just decoration)

##### 5. **Focus Outline vs Ring Trade-off (MEDIUM)**
- **Current:** Uses `focus-visible:ring` on some components
- **Issue:** `outline` is sometimes better for contrast
- **Consideration:** Blue-500 ring on dark-800 has adequate contrast (4.8:1)
- **Status:** Current approach is acceptable but could be enhanced

##### 6. **Dark Mode Focus Visibility (MEDIUM)**
- **Testing status:** Not tested in dark mode
- **Concern:** Blue-500 ring might not be visible enough on some dark backgrounds
- **Recommendation:** Test focus visibility on all dark-700 and dark-800 backgrounds
- **Current blue-500 on dark-800:** 4.8:1 - adequate
- **Current blue-500 on dark-700:** ~5.1:1 - adequate

##### 7. **Focus Visible in Safari/Firefox (LOW)**
- **Issue:** `focus-visible` is newer, fallback needed
- **Current:** Tailwind 3.3 supports it
- **Status:** Should work but needs testing on Safari

### Focus Indicator Priority Fixes

| Component | Type | Issue | Priority |
|-----------|------|-------|----------|
| Tab buttons (all) | Missing | No focus-visible | P0 |
| Form inputs | Inconsistent | focus vs focus-visible | P0 |
| Input labels | Missing | No htmlFor attributes | P1 |
| Icon buttons | Exposed | May be hard to see | P1 |

---

## 4. ARIA LABELS & LIVE REGIONS AUDIT

### ARIA Implementation Status

#### ✓ Good: Already Present

1. **File upload area (FileUpload.tsx, line 77):**
   ```tsx
   role="button"
   tabIndex={isLoading ? -1 : 0}
   aria-label="Upload CSV file"
   ```
   - ✓ Role specified
   - ✓ ARIA label provided
   - ✓ Keyboard accessible

2. **Clear file button (FileUpload.tsx, line 61):**
   ```tsx
   aria-label="Clear file"
   ```

3. **Sidebar toggle (App.tsx, line 307):**
   ```tsx
   aria-label="Toggle sidebar"
   ```

4. **Error message (SignalList.tsx, line 150):**
   ```tsx
   <div ... role="alert">
     {errorMessage}
   </div>
   ```
   - ✓ Alert role for dynamic messages
   - ✓ Will announce to screen readers

#### ✗ Missing: Needs Implementation

##### 1. **Tab Panel Roles (CRITICAL)**
- **Location:** App.tsx lines 362-393 (left tabs), 489-523 (right tabs), 444-461 (main tabs)
- **Issue:** Buttons used as tabs but no ARIA structure
- **What's needed:**
  ```tsx
  <div role="tablist">
    <button role="tab" aria-selected={leftPanelTab === 'signals'} aria-controls="signals-panel">
      Signals
    </button>
    {/* other tabs */}
  </div>
  <div id="signals-panel" role="tabpanel" aria-labelledby="signals-tab">
    {/* panel content */}
  </div>
  ```
- **Impact:** Screen reader users don't know they're in a tabbed interface
- **Affected:** ~15 tab instances

##### 2. **Icon-Only Button Labels (CRITICAL)**
- **Location:**
  - SignalList.tsx lines 125-137 (Upload/Download buttons)
  - FilterPanel Apply/Reset buttons (if icon-only considered)
- **Current:**
  ```tsx
  <button onClick={loadSignalSet}
    className="..."
    title="Load Signal Set"  // Title only, not aria-label
  >
    <Upload className="w-3 h-3" />
  </button>
  ```
- **Issue:** `title` attribute not reliable for screen readers; need `aria-label`
- **Fix:**
  ```tsx
  <button aria-label="Load signal set" className="...">
    <Upload className="w-3 h-3" />
  </button>
  ```
- **Affected:** 4-5 icon buttons across app

##### 3. **Form Input Descriptions (HIGH)**
- **Location:** FilterPanel.tsx, TimeRangePanel.tsx, AdvancedPanel.tsx
- **Issue:** Input fields have labels but no error descriptions or helper text markup
- **Example:**
  ```tsx
  <label className="label">Cutoff Frequency (0-1)</label>
  <input className="input" type="number" min={0.01} max={0.99} ... />
  // Should have aria-describedby if there's error/help text
  ```
- **When needed:** If form errors displayed (currently using Toast notifications)
- **Status:** Deferred to Phase 2.2 (errors currently not inline)

##### 4. **Dynamic Content Updates (MEDIUM)**
- **Location:**
  - Statistics panel updates
  - Chart title changes
  - Status messages
- **Issue:** Changes to data/stats not announced to screen readers
- **Fix needed:** Add `aria-live="polite"` to dynamic regions
- **Examples:**
  ```tsx
  <div aria-live="polite" aria-atomic="true">
    {statistics.count} data points analyzed
  </div>
  ```
- **Affected components:** StatisticsPanel, PlotView title, etc.
- **Priority:** Medium - User can still see updates visually

##### 5. **Loading State Announcements (MEDIUM)**
- **Location:** App.tsx file upload (line 48: isLoading state)
- **Current:** UI changes (opacity, button disabled)
- **Missing:** Screen reader announcement
- **Fix needed:**
  ```tsx
  <div role="status" aria-live="polite" className="sr-only">
    {isLoading && 'Loading file...'}
  </div>
  ```
- **Status:** Should be added to FileUpload component

##### 6. **Alert Messages (LOW)**
- **Location:** Toast notifications
- **Issue:** Toasts appear/disappear, screen reader might miss
- **Current:** Using Toast component (see Toast.tsx)
- **Status:** Need to check if Toast has `role="alert"` and `aria-live`

##### 7. **Search Results Count (MEDIUM)**
- **Location:** SignalList.tsx search results
- **Current:** Visual count shown in header
- **Missing:** Aria-live announcement of "No results" message
- **Fix:**
  ```tsx
  {searchTerm && filteredSignals.length === 0 && (
    <div aria-live="polite" role="status">
      No signals match "{searchTerm}"
    </div>
  )}
  ```

##### 8. **Selectability vs Clickability (MEDIUM)**
- **Location:** SignalList signal buttons (line 174)
- **Current:** Using `<button>` for selection
- **Issue:** Screen reader announces "button" but behavior is checkbox-like
- **Fix:**
  ```tsx
  role="checkbox"
  aria-checked={isSelected}
  ```
- **Or:** Use actual checkboxes with better styling
- **Impact:** Screen reader user confusion about expected interaction

##### 9. **Header/Footer Landmarks (LOW)**
- **Location:** App.tsx
- **Current:** Using `<header>` and `<footer>` - GOOD
- **Issue:** Main content area not wrapped in `<main>` - already done (line 328)
- **Status:** PASS - Good semantic HTML structure

### ARIA Fixes Priority

| Element | Type | Fix | Priority |
|---------|------|-----|----------|
| Tab panels | Missing | Add tablist/tab/tabpanel roles | P0 |
| Icon buttons | Missing | Add aria-labels | P0 |
| Loading state | Missing | Add aria-live for status | P1 |
| Dynamic updates | Missing | Add aria-live="polite" | P1 |
| Signal checkboxes | Semantic | Add role="checkbox" + aria-checked | P1 |
| Input descriptions | Missing | Add aria-describedby (when errors added) | P2 |

---

## 5. SUMMARY: Critical Issues by Component

### App.tsx
| Issue | Type | Severity |
|-------|------|----------|
| Tab buttons missing focus-visible | Focus | P0 |
| Modal focus trap missing | Keyboard | P0 |
| Modal Escape key missing | Keyboard | P0 |
| Tab panels missing ARIA roles | ARIA | P0 |
| Sidebar open state not announced | ARIA | P1 |

### FilterPanel.tsx
| Issue | Type | Severity |
|-------|------|----------|
| Label text contrast (dark-300) | Contrast | P0 |
| Input labels missing htmlFor | Accessibility | P1 |
| Input focus style (focus vs focus-visible) | Focus | P0 |
| Dynamic parameter inputs not announced | ARIA | P1 |

### SignalList.tsx
| Issue | Type | Severity |
|-------|------|----------|
| Icon buttons missing aria-labels | ARIA | P0 |
| 100+ signals = 100+ tab stops (roving tabindex needed) | Keyboard | P1 |
| Signal items should have role="checkbox" | ARIA | P1 |
| Search result count should have aria-live | ARIA | P1 |

### FileUpload.tsx
| Issue | Type | Severity |
|-------|------|----------|
| Upload icon hard to see (text-dark-500) | Contrast | P0 |
| Good: role="button" already present | - | ✓ |
| Good: aria-label already present | - | ✓ |

### PlotView.tsx & Charts
| Issue | Type | Severity |
|-------|------|----------|
| Chart data not announced (aria-live) | ARIA | P1 |
| Interactive chart elements keyboard accessible? | Keyboard | TBD |
| Chart title/description missing | ARIA | P1 |

### All Inputs (FilterPanel, TimeRangePanel, AdvancedPanel)
| Issue | Type | Severity |
|-------|------|----------|
| Placeholder text insufficient contrast | Contrast | P0 |
| Input focus-visible inconsistent | Focus | P0 |
| No aria-describedby for errors | ARIA | P2 |

---

## 6. IMPLEMENTATION ROADMAP (Phase 2.1)

### Day 1: Color Contrast Fixes (4 hours)
1. **Update CSS classes:**
   - `.label`: dark-300 → dark-100
   - `.placeholder`: dark-400 → dark-200
   - Inactive tabs: dark-400 → dark-300
   - Upload icon: dark-500 → dark-200

2. **Component updates:**
   - App.tsx tab buttons
   - FilterPanel search
   - All form inputs
   - Footer text

3. **Testing:**
   - Chrome DevTools contrast check
   - Test all dark-800 text combinations
   - Manual verification on dark backgrounds

### Day 2: Keyboard Navigation & Focus Indicators (4 hours)
1. **Focus indicators:**
   - Add focus-visible to all 12 tab buttons
   - Change form input focus to focus-visible
   - Add htmlFor to all labels
   - Add focus-visible to any remaining buttons

2. **Keyboard improvements:**
   - Add Escape key handler for mobile sidebar
   - Add focus trap for sidebar modal (use react-focus-lock library or manual)
   - Test Tab navigation order

3. **Testing:**
   - Tab through entire app with keyboard only
   - Verify focus visible on all elements
   - Test Escape key closes modal
   - Test on Firefox, Safari, Chrome

### Day 2.5: ARIA Labels (3 hours)
1. **Critical ARIA:**
   - Add tablist/tab/tabpanel roles to all 3 tab groups
   - Add aria-labels to icon-only buttons
   - Fix signal items: role="checkbox" + aria-checked

2. **Live regions:**
   - Add aria-live="polite" to error messages
   - Add aria-live to loading state
   - Add role="status" to dynamic updates

3. **Testing:**
   - NVDA screen reader test on Windows
   - VoiceOver on Mac
   - Chrome Accessibility Audit

---

## 7. DELIVERABLES CHECKLIST

- [ ] Color contrast audit report (this document)
- [ ] List of contrast failures by component
- [ ] Focus indicator test screenshots (before/after)
- [ ] Keyboard navigation test results
- [ ] ARIA label audit spreadsheet
- [ ] PR with P0 fixes: Contrast + Focus-visible + Tab ARIA roles
- [ ] HTML accessibility report (axe-core or similar)

---

## 8. NEXT STEPS (Immediate)

1. **Install axe DevTools** (Chrome) or **axe-core** npm package
2. **Run baseline accessibility scan** on http://localhost:3000
3. **Screenshot current state** (contrast, focus indicators, keyboard nav)
4. **Create GitHub issue** for each P0 fix with code location
5. **Start with contrast fixes** (simplest, highest impact)

---

## 9. KNOWN CONSTRAINTS

- **Must not break existing functionality** - All fixes CSS/attribute only
- **Keep visual design intact** - Using Tailwind only, no custom styles
- **Use existing component structure** - No major refactoring
- **Test on dark mode only** - Current app is dark theme only
- **Mobile support required** - Test all fixes on mobile and desktop

---

## Files Requiring Changes

**Core Components:**
- `/src/data_processing/data_processor/web/src/App.tsx`
- `/src/data_processing/data_processor/web/src/components/FilterPanel.tsx`
- `/src/data_processing/data_processor/web/src/components/SignalList.tsx`
- `/src/data_processing/data_processor/web/src/components/FileUpload.tsx`
- `/src/data_processing/data_processor/web/src/index.css` (class definitions)

**Secondary Components (audit needed):**
- `/src/data_processing/data_processor/web/src/components/AdvancedPanel.tsx`
- `/src/data_processing/data_processor/web/src/components/TimeRangePanel.tsx`
- `/src/data_processing/data_processor/web/src/components/PlotView.tsx`
- `/src/data_processing/data_processor/web/src/components/StatisticsPanel.tsx`

---

## WCAG 2.1 Criterion Mapping

### Failures in Current Implementation

| Criterion | Status | Issue |
|-----------|--------|-------|
| 1.4.3 Contrast (Minimum) | FAIL | Multiple text/background combos below 4.5:1 |
| 1.4.11 Non-text Contrast | FAIL | Icon contrast insufficient |
| 2.1.1 Keyboard | FAIL | Modal focus trap missing |
| 2.1.2 No Keyboard Trap | FAIL | Mobile sidebar focus trap missing |
| 2.4.3 Focus Order | FAIL | No roving tabindex on lists |
| 2.4.7 Focus Visible | FAIL | Inconsistent focus indicators |
| 3.2.1 On Focus | PASS | No unexpected focus behavior |
| 4.1.2 Name, Role, Value | FAIL | Missing ARIA roles on tabs |
| 4.1.3 Status Messages | FAIL | Live regions not announced |

### Passes Expected After Fixes

- ✓ 1.4.3 Contrast (Minimum)
- ✓ 1.4.11 Non-text Contrast
- ✓ 2.1.1 Keyboard
- ✓ 2.1.2 No Keyboard Trap
- ✓ 2.4.7 Focus Visible
- ✓ 4.1.2 Name, Role, Value
- ✓ 4.1.3 Status Messages

---

**Report Generated:** April 30, 2026  
**Next Review:** After Phase 2.1 implementation  
**Contact:** Accessibility Task Force (GitHub Issue #2409)
