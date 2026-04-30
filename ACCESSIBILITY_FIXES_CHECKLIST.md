# Accessibility Phase 2.1 - Quick Fix Checklist

## CRITICAL FIXES (P0) - Must Do First

### 1. Color Contrast Fixes in CSS

**File:** `/src/data_processing/data_processor/web/src/index.css`

- [ ] **Line 60-62: `.label` class**
  ```css
  .label {
    @apply block text-sm font-medium text-dark-100 mb-1; /* Change dark-300 to dark-100 */
  }
  ```
  **Impact:** Fixes ~25+ label contrast failures across entire app

- [ ] **Line 74-79: Input/select font size for mobile**
  - Add focus-visible to this media query
  ```css
  @media (max-width: 640px) {
    .input,
    .select {
      @apply text-base focus-visible:ring-2 focus-visible:ring-blue-500;
    }
  }
  ```

### 2. Placeholder Text Color Fix

**Files:** `/index.css`, affected components

- [ ] Update placeholder color in `.input` and `.select` classes
  ```css
  .input {
    @apply w-full px-3 py-2 bg-dark-800 border border-dark-600 rounded-lg
           text-dark-100 placeholder-dark-200  /* Change from dark-400 */
           focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500;
  }
  
  .select {
    @apply w-full px-3 py-2 bg-dark-800 border border-dark-600 rounded-lg
           text-dark-100 placeholder-dark-200  /* Change from dark-400 */
           cursor-pointer
           focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500;
  }
  ```

### 3. Tab Navigation Focus Indicators

**File:** `/src/data_processing/data_processor/web/src/index.css`

- [ ] Update `.tab` class to include focus-visible
  ```css
  .tab {
    @apply px-4 py-2 min-h-[48px] min-w-[48px] flex items-center justify-center 
           text-dark-400 hover:text-dark-100 border-b-2 border-transparent
           transition-colors duration-200 cursor-pointer
           focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500;
  }
  ```

- [ ] Update `.tab-active` to ensure contrast
  ```css
  .tab-active {
    @apply text-blue-500 border-blue-500; /* Already good, keep as is */
  }
  ```

- [ ] Change inactive tab text color to dark-300 (from dark-400)
  ```css
  .tab {
    @apply ... text-dark-300 ... /* Update this line */
  }
  ```

### 4. Form Input Focus Fix

- [ ] Change all input `.focus:` to `.focus-visible:`
  - **Location:** `.input` class line 39
  ```css
  /* BEFORE */
  focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent;
  
  /* AFTER */
  focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus:border-transparent;
  ```

### 5. Icon Color Fix (Upload Icon)

**File:** `/src/data_processing/data_processor/web/src/components/FileUpload.tsx`

- [ ] **Line 94: Upload icon visibility**
  ```tsx
  /* BEFORE */
  <Upload className="w-12 h-12 text-dark-500 mb-4" />
  
  /* AFTER */
  <Upload className="w-12 h-12 text-dark-200 mb-4" />
  ```
  **Impact:** Makes upload icon visible for users with low vision

---

## KEYBOARD NAVIGATION FIXES (P0)

### 1. Tab Button Focus Indicators

**File:** `/src/data_processing/data_processor/web/src/App.tsx`

- [ ] **Lines 363-392: Left panel tabs - Add focus-visible**
  ```tsx
  {/* BEFORE */}
  <button
    onClick={() => setLeftPanelTab('signals')}
    className={`px-3 py-2 min-h-[48px] flex items-center whitespace-nowrap transition-colors ${leftPanelTab === 'signals' ? 'border-b-2 border-blue-500 text-blue-400' : 'text-dark-400'}`}
  >
  
  {/* AFTER */}
  <button
    onClick={() => setLeftPanelTab('signals')}
    className={`px-3 py-2 min-h-[48px] flex items-center whitespace-nowrap transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded ${leftPanelTab === 'signals' ? 'border-b-2 border-blue-500 text-blue-400' : 'text-dark-400'}`}
  >
  ```
  **Count:** 4 buttons in left panel

- [ ] **Lines 490-522: Right panel tabs - Add focus-visible**
  Same fix as above
  **Count:** 5 buttons in right panel

- [ ] **Lines 445-460: Main content tabs - Add focus-visible**
  Same fix as above
  **Count:** 2 buttons

### 2. Mobile Sidebar Modal Escape Key

**File:** `/src/data_processing/data_processor/web/src/App.tsx`

- [ ] **Add useEffect for Escape key handler** (after line 287, in existing useEffect section)
  ```tsx
  // Close sidebar on Escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isMobile && sidebarOpen) {
        setSidebarOpen(false);
      }
    };
    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [sidebarOpen, isMobile]);
  ```

### 3. Mobile Sidebar Focus Trap (ADVANCED)

**File:** `/src/data_processing/data_processor/web/src/App.tsx`

- [ ] Install focus trap library (optional but recommended)
  ```bash
  npm install focus-trap-react
  ```

- [ ] **Option A: Using library** (simpler)
  ```tsx
  import FocusTrap from 'focus-trap-react';
  
  {isMobile && sidebarOpen && (
    <FocusTrap>
      <aside className="...">
        {/* sidebar content */}
      </aside>
    </FocusTrap>
  )}
  ```

- [ ] **Option B: Manual implementation** (if library not preferred)
  - Track first and last focusable elements
  - On Tab at end, loop back to first
  - On Shift+Tab at start, loop to last
  - See: https://www.w3.org/WAI/ARIA/apg/patterns/dialogmodal/

---

## ARIA LABELS & ROLES FIXES (P0-P1)

### 1. Tab Panel ARIA Roles

**File:** `/src/data_processing/data_processor/web/src/App.tsx`

- [ ] **Lines 362-393: Left panel tabs - Add ARIA roles**
  ```tsx
  {/* BEFORE */}
  <div className="flex overflow-x-auto border-b border-dark-700 text-xs">
    <button onClick={() => setLeftPanelTab('signals')} ...>
  
  {/* AFTER */}
  <div className="flex overflow-x-auto border-b border-dark-700 text-xs" role="tablist" aria-label="Left panel navigation">
    <button 
      role="tab" 
      aria-selected={leftPanelTab === 'signals'} 
      aria-controls="signals-panel"
      id="signals-tab"
      onClick={() => setLeftPanelTab('signals')} 
      ...
    >
      Signals
    </button>
    {/* Repeat for Advanced, Resample, Time */}
  </div>
  ```

- [ ] Wrap each panel content in `role="tabpanel"`
  ```tsx
  {leftPanelTab === 'signals' && (
    <div id="signals-panel" role="tabpanel" aria-labelledby="signals-tab">
      {/* content */}
    </div>
  )}
  ```

- [ ] **Repeat for right panel tabs (lines 489-523)** with appropriate IDs

### 2. Icon-Only Button ARIA Labels

**File:** `/src/data_processing/data_processor/web/src/components/SignalList.tsx`

- [ ] **Lines 125-138: Upload/Download buttons**
  ```tsx
  {/* BEFORE */}
  <button
    onClick={loadSignalSet}
    className="text-xs text-blue-500 hover:text-blue-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded p-1"
    title="Load Signal Set"
  >
    <Upload className="w-3 h-3" />
  </button>
  
  {/* AFTER */}
  <button
    onClick={loadSignalSet}
    aria-label="Load signal set"
    className="text-xs text-blue-500 hover:text-blue-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded p-1"
  >
    <Upload className="w-3 h-3" />
  </button>
  ```

- [ ] **Line 133: Save button**
  ```tsx
  aria-label="Save signal set"
  ```

### 3. Signal List Checkbox Semantics

**File:** `/src/data_processing/data_processor/web/src/components/SignalList.tsx`

- [ ] **Lines 174-196: Signal items - Add checkbox role**
  ```tsx
  {/* BEFORE */}
  <button
    key={signal}
    onClick={() => toggleSignal(signal)}
    className={`...`}
  >
  
  {/* AFTER */}
  <button
    key={signal}
    onClick={() => toggleSignal(signal)}
    role="checkbox"
    aria-checked={isSelected}
    aria-label={`${signal}, ${isSelected ? 'selected' : 'not selected'}`}
    className={`...`}
  >
  ```

### 4. Input Label Association

**File:** `/src/data_processing/data_processor/web/src/components/FilterPanel.tsx`

- [ ] **Line 61: Window Size label**
  ```tsx
  {/* BEFORE */}
  <label className="label">Window Size</label>
  <input type="number" className="input" value={parameters.ma_window} ... />
  
  {/* AFTER */}
  <label htmlFor="ma-window" className="label">Window Size</label>
  <input id="ma-window" type="number" className="input" value={parameters.ma_window} ... />
  ```

- [ ] Repeat for ALL input labels in FilterPanel, TimeRangePanel, AdvancedPanel
  - **Approx. 20+ instances**
  - Use IDs like: `bw-order`, `bw-cutoff`, `ma-window`, etc.

### 5. Dynamic Content - Aria-Live

**File:** `/src/data_processing/data_processor/web/src/components/FileUpload.tsx`

- [ ] **Add loading status announcement**
  ```tsx
  {/* Add after upload area */}
  <div role="status" aria-live="polite" aria-atomic="true" className="sr-only">
    {isLoading && 'Loading file...'}
  </div>
  ```

- [ ] Add CSS class for screen reader only text
  ```css
  /* Add to index.css */
  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border-width: 0;
  }
  ```

---

## HIGH PRIORITY FIXES (P1) - Do Next

### 1. Search Results Announcement

**File:** `/src/data_processing/data_processor/web/src/components/SignalList.tsx`

- [ ] **Line 200-203: Add aria-live to no results message**
  ```tsx
  {searchTerm && filteredSignals.length === 0 && (
    <div aria-live="polite" role="status" className="text-dark-400 text-center py-4 text-sm">
      No signals match "{searchTerm}"
    </div>
  )}
  ```

### 2. Error Message in SignalList

- [ ] **Line 149-152: Already has role="alert"** ✓
  - But should add aria-atomic="true" for clarity
  ```tsx
  <div className="mb-3 p-2 bg-red-500/10 border border-red-500/50 rounded text-red-400 text-sm" 
       role="alert"
       aria-atomic="true"
  >
  ```

### 3. Chart/Plot Title Accessibility

**File:** `/src/data_processing/data_processor/web/src/components/PlotView.tsx`

- [ ] Add aria-label to chart
  ```tsx
  <div 
    className="..." 
    aria-label={`Chart showing ${title}. Selected signals: ${selectedSignals.join(', ')}`}
  >
    {/* plot content */}
  </div>
  ```

### 4. Tab Contrast Fix

**File:** `/src/data_processing/data_processor/web/src/App.tsx`

- [ ] Ensure inactive tab contrast is 4.5:1
  - Currently: `text-dark-400` on `dark-800` = ~2.8:1 (FAIL)
  - Fix: Change to `text-dark-300` in App.tsx
  ```tsx
  className={`... ${leftPanelTab === 'signals' ? ... : 'text-dark-300'}`}
  ```

---

## TESTING CHECKLIST

### Keyboard Navigation Testing (No Mouse)
- [ ] **Tab through entire app:**
  - Can reach header buttons?
  - Can reach all form inputs?
  - Can reach all tab buttons?
  - Is tab order logical (left-to-right, top-to-bottom)?

- [ ] **Escape key:**
  - Does Escape close mobile sidebar?
  - Does Escape close any modals (if added)?

- [ ] **Arrow keys:**
  - Can navigate select dropdowns?
  - Can navigate signal list with arrow keys? (currently no, by design)

- [ ] **Enter key:**
  - Can activate buttons?
  - Can submit form inputs?

### Focus Indicator Testing
- [ ] **All interactive elements visible:**
  - Tab buttons show blue ring?
  - Form inputs show blue ring on focus-visible?
  - Buttons show blue ring?
  - Mobile hamburger button shows ring?

- [ ] **Dark mode visibility:**
  - Blue-500 ring visible on dark-800 background?
  - Blue-500 ring visible on dark-700 background?
  - Ring is at least 2px thick?

- [ ] **Test in different browsers:**
  - Chrome
  - Firefox
  - Safari
  - Edge

### Color Contrast Testing
- [ ] **Using Chrome DevTools:**
  - Open DevTools > More Tools > Accessibility
  - Inspect each element
  - Check contrast ratio shown

- [ ] **Critical elements:**
  - [ ] All labels (should be dark-100 on dark-800)
  - [ ] All placeholder text (should be dark-200 on dark-800)
  - [ ] Inactive tabs (should be dark-300 on dark-800)
  - [ ] Upload icon (should be dark-200)

- [ ] **All elements should show:** ✓ or higher ratio than required

### Screen Reader Testing (NVDA on Windows / VoiceOver on Mac)
- [ ] **Tab panels announced correctly:**
  - Should announce "Signals tab, selected" or similar
  - Should announce tabpanel region

- [ ] **Icon buttons announced:**
  - "Load signal set button" instead of just icon description

- [ ] **Signal selection:**
  - "Temperature checkbox, checked"
  - "Pressure checkbox, not checked"

- [ ] **Error/status messages:**
  - Live regions announced
  - Alerts announced immediately

- [ ] **Form inputs:**
  - Label properly associated
  - Can activate with Enter key

---

## Files to Modify (Summary)

| File | Changes | Lines |
|------|---------|-------|
| `src/index.css` | Label color, focus-visible, placeholder color | 60-62, 36-45, 74-79 |
| `App.tsx` | Tab focus-visible, Escape handler, ARIA roles, tab contrast | Multiple |
| `FilterPanel.tsx` | Label htmlFor attributes | ~20 instances |
| `SignalList.tsx` | Icon button aria-labels, checkbox roles | 125-196 |
| `FileUpload.tsx` | Upload icon color, aria-live status | 94, after 100 |
| `TimeRangePanel.tsx` | Label htmlFor attributes | TBD |
| `AdvancedPanel.tsx` | Label htmlFor attributes | TBD |
| `PlotView.tsx` | Chart aria-label | TBD |

---

## Estimated Time per Fix

| Fix | Time | Priority |
|-----|------|----------|
| CSS color changes | 15 min | P0 |
| Focus-visible on tabs | 30 min | P0 |
| Escape key handler | 20 min | P0 |
| Tab ARIA roles | 1 hour | P0 |
| Icon button aria-labels | 20 min | P0 |
| Signal checkbox roles | 30 min | P0 |
| Input label htmlFor | 1.5 hours | P1 |
| Test all fixes | 1.5 hours | Required |

**Total P0 (Critical):** ~4.5 hours  
**Total P1 (High):** ~2.5 hours  
**Total Testing:** 1.5 hours

---

## References

- [WCAG 2.1 Tab Pattern](https://www.w3.org/WAI/ARIA/apg/patterns/tabs/)
- [WCAG 2.1 Focus Visible](https://www.w3.org/WAI/WCAG21/Understanding/focus-visible.html)
- [WCAG 2.1 Contrast Minimum](https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html)
- [Aria-live regions](https://www.w3.org/WAI/ARIA/apg/practices/live-regions/)
- [Focus management in modals](https://www.w3.org/WAI/ARIA/apg/patterns/dialogmodal/)

---

**Version:** 1.0  
**Last Updated:** April 30, 2026  
**Status:** Ready for Implementation
