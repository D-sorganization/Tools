# Architecture Ruling: Impact-Interval PyQt Tab Disposition

**Issue:** [#4946](https://github.com/D-sorganization/Tools/issues/4946)  
**Parent Program:** [Repository_Management#1505](https://github.com/D-sorganization/Repository_Management/issues/1505) (Fleet Readiness Program 2026-Q4)  
**Status:** Approved & Ratified  
**Date:** 2026-09-21  
**Scope:** `src/swing_sim/impact_interval/`, `src/swing_sim/impact/types.py`, and Rate-of-Closure PyQt UI tabs  
**Disposition:** Drop proposed standalone tab as superseded by #4473; retain headless physics calculation engine  

---

## 1. Context & Problem Statement

Issue [#4946](https://github.com/D-sorganization/Tools/issues/4946) requested an owner architectural ruling on the disposition of the proposed `impact-interval` PyQt tab:

1. **Premise Verified on Main:**
   - The underlying mathematical and physics package `src/swing_sim/impact_interval/` was delivered and qualified in PR #4945.
   - The standalone UI view `impact_interval_view*` was never created and is absent from `main`.
   - The `ImpactModelType` enum (`src/shared/python/swing_sim/impact/types.py:42-47`) contains `RIGID_BODY`, `SPRING_DAMPER`, and `FINITE_TIME`. It does not contain an `IMPACT_INTERVAL` variant.
2. **Evolution of the UI Architecture:**
   - Epics [#4433](https://github.com/D-sorganization/Tools/issues/4433) and [#4473](https://github.com/D-sorganization/Tools/issues/4473) established the visual-first tab family and multi-view workspace compositor (PR #5264, Issue #4225), integrating synchronized Impact, Swing, and Flight viewports, run selection, and layout presets across desktop (PyQt6) and web (React).
   - Adding a standalone `impact-interval` PyQt tab would duplicate existing multi-viewport impact presentations, fragment the user experience, and force a breaking migration of `SimulationRun` models to accommodate an unnecessary enum member.

---

## 2. Decision & Formal Ruling

**Ruling: Drop the standalone `impact-interval` PyQt tab as superseded; retain the headless calculation engine.**

1. **Drop Standalone PyQt Tab:**
   - The proposed standalone `impact_interval_view` PyQt tab is formally **dropped**. No separate tab shall be added to the launcher or `RateOfClosureMainWindow`.
   - The `ImpactModelType` enum shall **not** be modified to add an `IMPACT_INTERVAL` member, preserving the stability of the canonical `SimulationRun` serialized contract.
2. **Retain Headless Physics Engine:**
   - The `src/swing_sim/impact_interval/` calculation package is **retained as an authoritative computational module**.
   - It remains available for script-driven analysis, headless optimization routines, CLI parameter studies, and future integration as an optional solver backend under the existing unified impact models.
3. **Consolidation into Visual-First Architecture:**
   - Any future visualization of impact-interval intervals or restitution kinematics shall be presented through the existing visual-first tab architecture (`SynchronizedSimulationView`, `StrikeView`, and the #4473 workspace compositor) rather than creating separate siloed tabs.

---

## 3. Rationale

- **DRY & UX Cohesion:** The visual-first UX initiative (#4433 / #4473) established that users require unified, synchronized multi-view workspaces rather than disparate, isolated tabs for each physical sub-phase.
- **Contract Stability:** Keeping `ImpactModelType` restricted to its current verified members avoids breaking serialized run artifacts and eliminates schema migration risks across Python, Rust, and TypeScript layers.
- **Clean Separation of Concerns (LoD):** The mathematical formulation in `src/swing_sim/impact_interval/` is self-contained and functions effectively as a headless computational library without coupling to UI widgets.

---

## 4. Invariants & Guardrails

1. **No Standalone Tab Introduction:** Pull requests introducing an isolated `impact_interval_view` or adding `IMPACT_INTERVAL` to `ImpactModelType` without a new ratified architectural specification shall be rejected.
2. **Headless Engine Maintenance:** `src/swing_sim/impact_interval/` remains covered by unit and numerical regression tests to ensure computational integrity.

---

## 5. Disposition

Resolves and closes Issue [#4946](https://github.com/D-sorganization/Tools/issues/4946).
