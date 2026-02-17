# Master Architecture Plan: "UpstreamDrift" & Shared Tools Ecosystem

**Date**: 2026-01-30
**Status**: DRAFT - PENDING APPROVAL
**Executive Summary**: This plan outlines the strategic reorganization of the repository fleet to establish a centralized "Standard Library" in the `Tools` repository. This enables the **Data Processor** and other generic engineering engines to be shared across the **Gasification Model** (moving to React/API) and **UpstreamDrift** (formerly Golf Modeling Suite, keeping PyQt).

**Constraint Checklist**:

- [x] Repository name remains `Tools`.
- [x] Python package name: `upstream_drift_tools` (installed via `pip install -e Tools`).
- [x] Code Quality: Ruff, Black, MyPy compliant.
- [x] Pragmatic Programmer: DRY (Don't Repeat Yourself), Orthogonality (Decoupled components).

---

## 🛑 Phase 0: Preparation & Safety (Pre-Work)

**Objective**: Ensure no work begins without tracking and safety nets.

### 0.1 Create Tracking Issues

Run the following commands to create the tracking issues in each repository:

**Tools (`D-sorganization/Tools`)**

- [ ] `gh issue create --title "Architecture: Establish Shared Tools Library Structure" --body "Create Tools/src/shared/python structure for reusable logic."`
- [ ] `gh issue create --title "Refactor: Data Processor Core Extraction" --body "Decouple logic from Data_Processor_Integrated.py into shared/data_processing."`
- [ ] `gh issue create --title "Feat: Data Processor PyQt Widget" --body "Create a PyQt6 widget wrapper for the Data Processor core."`

**Gasification Model (`D-sorganization/Gasification_Model`)**

- [ ] `gh issue create --title "Refactor: Extract Calculators to Tools" --body "Move Unit Converter, Steam Calculator, and Thermo Props to shared Tools repo."`
- [ ] `gh issue create --title "Feat: Consume Shared Tools in API" --body "Update FastAPI backend to import calculators from upstream_drift_tools."`

**UpstreamDrift (`D-sorganization/Golf_Modeling_Suite`)**

- [ ] `gh issue create --title "Rebrand: Rename to UpstreamDrift" --body "Update branding, diagrams, and eventually repo name in future."`
- [ ] `gh issue create --title "Refactor: Move Polynomial Generator & C3D Reader" --body "Move these components to shared Tools repo."`

### 0.2 Safe Backups

- [ ] Create a "Pre-Refactor" tag in all three repos: `git tag -a refactor-start-v1 -m "State before Monolith breakdown"`

---

## 🏗️ Phase 1: The Shared Library Foundation

**Objective**: Convert `Tools` into a distributable Python package.

### 1.1 Directory Structure (`Tools`)

We maintain the `Tools` folder but structure the internals to be a valid Python package.

```text
Tools/
  pyproject.toml              # Defines 'upstream_drift_tools' package
  src/
    shared/                   # THE STANDARD LIBRARY
      python/
        upstream_drift_tools/ # The actual importable package
          __init__.py
          data_processing/    # Data Processor Core (No UI)
            processing.py
            filtering.py
            io.py
          calculators/        # Generic Engineering Calcs
            conversion/       # Unit & Flow Rate conversion
            thermo/           # Steam & Thermodynamic properties
            fluid_dynamics/   # Pressure drop, etc.
          robotics/           # Robotics Tools
            urdf/
            c3d/

    ui/                       # UI WRAPPERS (PyQt/Tkinter)
      python/
        widgets/
          data_processor_qt.py
          polynomial_qt.py
      tk_apps/
          data_processor_app.py
```

### 1.2 Package Configuration

- Update `Tools/pyproject.toml` to make `src/shared/python` a pip-installable package.
- This allows other repos to use: `from upstream_drift_tools.calculators.thermo import SteamEngine`.

---

## 🧪 Phase 2: Component Extraction (The Great Migration)

**Objective**: Move specific, generic components to the Shared Library.

### 2.1 Confirmed Candidates for Migration

These components have been analyzed and confirm to be generic enough for sharing:

**From Gasification Model:**

1. **Steam Calculator** (`steam_engine.py`):
   - _Why_: Generic IAPWS-97 steam table implementation. Useful for any thermal modeling.
   - _Destination_: `upstream_drift_tools.calculators.thermo.steam`
2. **Unit Converter Components** (`flow_rate_converter.py`, `scfm_acfm_converter.py`):
   - _Why_: Pure physics constants and conversion math.
   - _Destination_: `upstream_drift_tools.calculators.conversion`
3. **Thermodynamic Properties** (`thermodynamic_properties_calculator.py`):
   - _Why_: Generic Ideal Gas / NIST / JANAF property lookups.
   - _Destination_: `upstream_drift_tools.calculators.thermo.properties`

**From UpstreamDrift (Golf Suite):** 4. **C3D Reader** (`c3d_reader.py`): \*_Why_: Standard biomechanics file format reader.

- _Destination_: `upstream_drift_tools.robotics.c3d` 5. **Polynomial Generator** (`polynomial_generator.py`): \*_Why_: Generic math/fitting logic. (UI stays in widgets, logic moves to core).
- _Destination_: `upstream_drift_tools.math.polynomials` 6. **URDF Generator**: \*_Why_: Robot description format generation. \* _Destination_: `upstream_drift_tools.robotics.urdf`

---

## 🧠 Phase 3: Data Processor 3.0 (Rewritten)

**Objective**: A single "Brain" with multiple "Foundations".

### 3.1 Core Logic Extraction

- Refactor `Data_Processor_Integrated.py`.
- Extract `pandas` logic (filtering, smoothing, derivatives) to `upstream_drift_tools.data_processing`.
- Ensure **Zero UI Dependencies** in this core layer.

### 3.2 The PyQt Widget

- Create `class DataProcessorWidget(QWidget)` in `Tools/src/ui/python/widgets/`.
- Connects signals/slots to the Core Logic.
- **Benefit**: Native integration into UpstreamDrift's PyQt interface.

### 3.3 The Headless API Wrapper

- For the **Gasification Model (React/FastAPI)**:
- Create a new router in Gasification API: `src/integrated_process_simulator/api/routers/processing.py`.
- This router simply calls `upstream_drift_tools.data_processing.process_csv(...)`.

---

## 🔗 Phase 4: Integration & Consumption

**Objective**: Wire everything back together.

### 4.1 Dependency Management

- In **Gasification_Model** and **UpstreamDrift**:
- install via `pip install -e ../Tools`.

### 4.2 Gasification Model (React Transition)

- **Action**: Modify `api/main.py` in the Gasification Model to register the new Generic Calculator routes.
- **Benefit**: The React frontend calls the API, which uses the Shared Tools.

### 4.3 UpstreamDrift (Parity Check)

- Import `DataProcessorWidget` and add it as a new Tab.
- **Rebrand**: Update window titles, icons, and diagrams to "UpstreamDrift" where applicable.

---

## ✅ Implementation Checklist for User

1. **Approve Plan**: Confirm this structure meets your needs.
2. **Execute Creation**: Use the "Create Issues" commands listed in Phase 0.
3. **Start Phase 1**: Authorize me to begin the folder structure creation and initial file moves.
