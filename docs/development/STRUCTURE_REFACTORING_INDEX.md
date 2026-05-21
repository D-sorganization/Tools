# Structure Refactoring Index — Issue #2405

**Quick navigation for all Phase 2 refactoring documentation.**

---

## Phase 2.1: Structure Audit & Documentation ✓ COMPLETE

Phase 2.1 audited the current state and created documentation without making code changes.

### Start Here

- **[PHASE_2_1_SUMMARY.md](./PHASE_2_1_SUMMARY.md)** — Executive summary, key findings, roadmap

### Reference Documents (Phase 2.1 Deliverables)

1. **[docs/TOOL_STRUCTURE.md](./docs/TOOL_STRUCTURE.md)** — Canonical Structure Definition

   - Current state overview
   - Canonical directory layout
   - Tool variants (simple, GUI, engine-based)
   - Shared libraries and services
   - Import path guidelines
   - Validation checklist for each tool

2. **[AUDIT_REPORT_2405.md](./AUDIT_REPORT_2405.md)** — Current State Audit

   - Executive summary
   - Tool registration status (20/21 registered)
   - Module structure analysis
   - Namespace collision analysis (all safe)
   - Manifest coverage report (95.2%)
   - Directory structure summary
   - Test coverage status
   - Complete refactoring roadmap (Phases 2.2-4)
   - Risk assessment
   - Recommendations

3. **[COLLISION_REPORT_2405.md](./COLLISION_REPORT_2405.md)** — Namespace Collision Analysis

   - Summary of 69 duplicate filenames
   - Detailed analysis of each collision type
   - Risk level assessment
   - Why collisions are safe (scoping analysis)
   - Deprecation timeline
   - Import pattern analysis
   - Migration path documentation
   - Recommendations per phase

4. **[MANIFEST_AUDIT_2405.md](./MANIFEST_AUDIT_2405.md)** — Manifest Completeness Report
   - Coverage summary (20/21 tools)
   - All 20 registered tools listed
   - Unregistered tools identified (lower_body_model)
   - Manifest schema validation
   - Module path validation (spot checks)
   - Category distribution
   - Stale entry detection (none found)
   - Port assignment validation (no conflicts)
   - Recommendations for Phase 2.2

### Data Files

- **tools_inventory.csv** (in /tmp/)
  - 21-row spreadsheet of all tools
  - Tool name, current location, canonical location, registration status, action needed
  - Use for tracking Phase 2.2+ refactoring

---

## Phase 2.2: Register Missing Tools (Next: 1-2 days)

Register lower_body_model in tool_manifest.yaml to achieve 100% coverage.

### Action Items

1. Inspect src/lower_body_model/launch_pyqt6.py
2. Inspect src/lower_body_model/gui_registration.py
3. Add entry to src/shared/python/gui_launcher/tool_manifest.yaml
4. Validate in CI and run tests

### Reference

- See: MANIFEST_AUDIT_2405.md → "Unregistered Tools" section
- See: MANIFEST_AUDIT_2405.md → "Recommended Manifest Entry"
- See: docs/TOOL_STRUCTURE.md → Validation Checklist

### Timeline

- Estimated: 2-3 hours
- Risk: Very low (inspection and metadata addition)
- Blocker: None

---

## Phase 2.3: Deprecate Legacy Files (After 2.2: 2-3 days)

Mark gui_registration.py and launch_pyqt6.py as deprecated.

### Action Items

1. Add deprecation comments to all 21 launch_pyqt6.py files
2. Add deprecation comments to all 21 gui_registration.py files
3. Create developer migration guide
4. Add manifest validation to CI pipeline

### Reference

- See: COLLISION_REPORT_2405.md → "Critical Duplicates" sections
- See: docs/TOOL_STRUCTURE.md → Deprecation Timeline
- See: PHASE_2_1_SUMMARY.md → Phase 2.3 Tasks

### Timeline

- Estimated: 2-4 hours
- Risk: Low (comments only, no functional changes)
- Blocker: Phase 2.2 must complete first

---

## Phase 3: Reorganize by Category (Future: 1+ weeks)

_Optional_: Group tools by domain/category for better organization.

### Proposed Structure

```
src/
├── analysis/           # Multi-parameter analysis, etc.
├── biomechanics/       # C3D viewer, motion capture
├── calculators/        # Financial, pressure drop, steam, inertia
├── data_processing/    # PDF renamer, data processor
├── engineering/        # P&ID generator, URDF builder, Vessel drafter
├── optimization/       # Optimizer, ODE solver, function generator
├── robotics/           # Humanoid builder, rotation converter
├── signal_processing/  # Signal studio, flow rate converter
├── simulation/         # Pendulum simulator, RRT path planner
├── tools/              # Folder tools, low-level utilities
└── shared/
    └── python/         # Platform libraries and services
```

### Action Items

1. Create category directories
2. Move tools (automated script)
3. Update import paths
4. Update tool_manifest.yaml
5. Coordinate with downstream repos (UpstreamDrift, Gasification_Model)

### Reference

- See: docs/TOOL_STRUCTURE.md → "Categories (Recommended)"
- See: PHASE_2_1_SUMMARY.md → "Phase 3: Reorganize by Category"

### Timeline

- Estimated: 1+ weeks
- Risk: HIGH (breaking import paths)
- Blocker: Requires downstream coordination

---

## Phase 4: Remove Deprecated Files (6+ months after Phase 2)

_Final cleanup_: Delete deprecated files after sufficient deprecation period.

### Action Items

1. Delete all 21 gui_registration.py files
2. Delete all 21 launch_pyqt6.py files
3. Update tool loading to use manifest exclusively
4. Simplify CI rules

### Reference

- See: COLLISION_REPORT_2405.md → Deprecation Timeline
- See: PHASE_2_1_SUMMARY.md → "Phase 4: Remove Deprecated Files"

### Timeline

- Estimated: 2+ weeks
- Risk: Medium (after deprecation period)
- Blocker: Phase 2.3 must have completed 6+ months prior

---

## Key Findings Summary

### 1. Namespace Collisions: NOT BREAKING

- 69 duplicate filenames identified
- All are tool-scoped (safe)
- Manifest centralization resolves GUI metadata duplication (#1863)

### 2. Tool Registration: NEARLY COMPLETE

- 20/21 tools in manifest (95.2% coverage)
- 1 unregistered: lower_body_model
- Action: 2-hour fix in Phase 2.2

### 3. Module Organization: SOUND

- No breaking issues detected
- Current structure properly scoped
- Ready for Phase 2.2+

---

## Decision Tree

### "I'm a developer, what do I need to know?"

→ Read: **docs/TOOL_STRUCTURE.md** (canonical reference)

### "I need to understand the current issues"

→ Read: **AUDIT_REPORT_2405.md** (comprehensive summary)

### "What are the namespace collisions and are they dangerous?"

→ Read: **COLLISION_REPORT_2405.md** (detailed analysis)

### "Is the manifest complete and valid?"

→ Read: **MANIFEST_AUDIT_2405.md** (validation report)

### "What's the overall plan for Phases 2-4?"

→ Read: **PHASE_2_1_SUMMARY.md** (roadmap and timeline)

### "I need to track which tools need what work"

→ Use: **tools_inventory.csv** (spreadsheet)

---

## Quick Reference: Action Items by Role

### Project Lead

1. Review PHASE_2_1_SUMMARY.md (5 min)
2. Approve Phase 2.2 implementation (2.2 section, 10 min)
3. Schedule Phase 2.3 and beyond

### Developer (Implementing Phase 2.2)

1. Read MANIFEST_AUDIT_2405.md → "Unregistered Tools" (5 min)
2. Inspect lower_body_model implementation (15 min)
3. Add manifest entry (10 min)
4. Test and validate (30 min)

### Developer (Implementing Phase 2.3)

1. Read COLLISION_REPORT_2405.md → "Critical Duplicates" (10 min)
2. Add deprecation comments to 42 files (script can automate)
3. Create migration guide
4. Update CI pipeline

### Architect (Planning Phase 3)

1. Read docs/TOOL_STRUCTURE.md (15 min)
2. Review Phase 3 section in PHASE_2_1_SUMMARY.md (10 min)
3. Plan category reorganization
4. Coordinate with downstream repos

---

## Glossary

**Tool:** A standalone application (GUI or CLI) that users can launch. Examples: financial_calculator, c3d_viewer.

**Shared Library:** Reusable code shared by multiple tools. Located in src/shared/python/. Examples: signal_toolkit, upstream_drift_tools.

**Platform Service:** Infrastructure shared across tools. Examples: gui_launcher, calc_backend.

**Manifest:** Central registry of tools (tool_manifest.yaml). Contains metadata: tool name, description, module paths, dependencies, etc.

**Namespace Collision:** Two files with the same name in different locations. Example: src/c3d_viewer/launch_pyqt6.py and src/financial_calculator/launch_pyqt6.py. Safe if scoped to specific packages.

**Deprecation:** Marking a feature as no longer recommended, with a timeline for removal.

**Breaking Change:** Modification that requires changes in downstream code. Example: moving a tool changes import paths.

---

## File Organization

```
/home/user/Tools/
├── docs/
│   └── TOOL_STRUCTURE.md                 ← Canonical reference
├── STRUCTURE_REFACTORING_INDEX.md        ← This file (navigation)
├── PHASE_2_1_SUMMARY.md                  ← Phase summary & roadmap
├── AUDIT_REPORT_2405.md                  ← Current state audit
├── COLLISION_REPORT_2405.md              ← Namespace analysis
├── MANIFEST_AUDIT_2405.md                ← Manifest validation
└── /tmp/
    └── tools_inventory.csv               ← Tool spreadsheet
```

---

## Next Steps

### Immediate (This Week)

1. Review PHASE_2_1_SUMMARY.md
2. Approve Phase 2.2 plan
3. Schedule implementation

### Short-term (Next Week)

1. Implement Phase 2.2 (register lower_body_model)
2. Update CI for manifest validation
3. Run full test suite

### Medium-term (2-4 Weeks)

1. Implement Phase 2.3 (deprecation notices)
2. Create migration guide
3. Begin discussion on Phase 3

### Long-term (Ongoing)

1. Plan Phase 3 (category reorganization)
2. Coordinate with downstream repos
3. Schedule Phase 4 (6+ months out)

---

## Related Issues

- **#1863** — Duplicate GUI_INFO dict in 20 files (resolved by tool_manifest.yaml)
- **#2405** — Inconsistent module structure and namespace collisions (this work)

---

## Questions?

Refer to the appropriate document:

- **Architecture/Structure:** docs/TOOL_STRUCTURE.md
- **Current Issues:** AUDIT_REPORT_2405.md or COLLISION_REPORT_2405.md
- **Manifest Status:** MANIFEST_AUDIT_2405.md
- **Roadmap/Timeline:** PHASE_2_1_SUMMARY.md

Or contact the project lead / architecture team.

---

**Status:** Phase 2.1 Complete | Phase 2.2 Ready | No Code Changes Made
**Generated:** 2026-04-30
**Scope:** Issue #2405 Structure Refactoring Phases 2.1-4
