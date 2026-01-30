# Architecture Refactoring Plan: Modular Tools & Components

**Date**: 2026-01-30
**Objectives**:

1. Decouple the **Data Processor** from its monolithic Tkinter/CustomTkinter implementation into a modular UI-agnostic core library and separate UI wrappers.
2. Establish a **Shared Tools Library** in `Tools/src/shared` to house reusable components like the URDF Generator, C3D Reader, and Polynomial Generator.
3. Enable cross-repository usage (Golf Suite, Gasification Model) via standard Python package imports.

---

## 1. The Core Strategy: "Core vs. Shell" Architecture

To allow tools like the Data Processor to function as a "widget" in PyQt, a dedicated app in Tkinter, and potentially a backend service for React, we must strictly separate **logic** from **presentation**.

### 1.1 Structural Changes

We will reorganize `Tools` to promote reusability.

```text
Tools/
  src/
    shared/                   <-- NEW: The source of truth for all shared logic
      python/
        data_processing/      <-- Core logic for Data Processor (No GUI code)
          __init__.py
          processor.py        # processing classes
          filters.py          # signal processing filters
          io.py               # file readers/writers (migrated from file_utils.py)
        c3d/                  <-- Migrated Core logic for C3D
        urdf/                 <-- Migrated Core logic for URDF
        polynomials/          <-- Migrated Core logic for Polynomials
    
    ui/                       <-- NEW: Reusable UI Components
      python/
        widgets/              <-- PyQt/Tkinter wrappers for the Core logic
          data_processor_qt.py   # PyQt6 Widget wrapper 
          data_processor_tk.py   # Tkinter Widget wrapper (refactored app)
          polynomial_qt.py       # PyQt6 Widget (migrated)
```

## 2. Component Migration Plan

### 2.1 Data Processor Refactor

**Current Status**: Monolithic `Data_Processor_Integrated.py` mixing `customtkinter` with pandas logic.
**The Plan**:

1. **Extract Logic**: Move `process_single_csv_file`, `_poly_derivative`, and file IO logic to `Tools/src/shared/python/data_processing/`.
2. **Create PyQt Widget**: Build `DataProcessorWidget` (PyQt6) that consumes the shared logic. This allows it to be embedded directly into the **Golf Modeling Suite** (which is PyQt).
3. **Update Legacy App**: Refactor `Data_Processor_Integrated.py` to simply import the shared logic, maintaining the standalone tool for now.
4. **React Integration**: For the **Gasification Model** (React), exposing the `data_processing` library via FastAPI endpoints is the clean solution, rather than embedding a Python GUI.

### 2.2 Polynomial Generator

**Current Status**: PyQt6 widget inside `Golf_Modeling_Suite`.
**The Plan**:

1. **Move**: Move `polynomial_generator.py` to `Tools/src/shared/python/ui/widgets/polynomial_qt.py`.
2. **Decouple**: Ensure it depends only on standard libraries (numpy, matplotlib, PyQt6) and not Golf-Suite specific logging/paths.
3. **Reuse**: Import this widget back into Golf Suite and any new tools.

### 2.3 C3D Reader

**Current Status**: `c3d_reader.py` inside Golf Suite.
**The Plan**:

1. **Move**: Move to `Tools/src/shared/python/c3d/reader.py`.
2. **Standardize**: Ensure it meets the rigorous docstring/type-hinting standards of the Tools repo.

### 2.4 URDF Generator

**Current Status**: Separate tool in `Golf_Modeling_Suite`.
**The Plan**:

1. **Move**: Move core logic to `Tools/src/shared/python/urdf/`.
2. **Componentize**: If it has a UI, split it into a reusable Widget.

## 3. Implementation Steps

### Phase 1: Foundation (Immediate)

1. Create `Tools/src/shared/python/{data_processing, c3d, urdf}` packages.
2. Migrate `file_utils.py` and `c3d_reader.py` first as they are low-hanging fruit.

### Phase 2: Data Processor Extraction (High Effort)

1. Refactor `Data_Processor_Integrated.py` to strip out the `process_single_csv_file` and filter functions into the shared library.
2. Verify the standalone app still works using the new imports.

### Phase 3: UI Componentization

1. Move `polynomial_generator.py` to shared widgets.
2. Create a "Proof of Concept" PyQt application in `Tools` that imports both the Data Processor logic (maybe just a simple runner for now) and the Polynomial Widget to prove they work in a neutral environment.

## 4. User Action Required

Approve this plan to begin the reorganization. The "Bastardized" growth will be tamed by enforcing the **Presentation-Abstraction-Control** pattern.

Do you want me to start with **Phase 1 (Foundation)** immediately?
