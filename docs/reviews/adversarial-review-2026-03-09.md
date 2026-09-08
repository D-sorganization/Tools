# Adversarial Code Review — 2026-03-09

## Overview

This document summarises the adversarial code review performed on the **Tools** repository on 2026-03-09 and records all quick-win fixes that were implemented in the same session. A companion DOCX report, [`Adversarial_Code_Review_UpstreamDrift_Tools_2026-03-09.docx`](Adversarial_Code_Review_UpstreamDrift_Tools_2026-03-09.docx) beside this file, contains the full issue catalogue with severity ratings (the C-/H-/M- identifiers used below). It was written to the session workspace root and committed to `D-sorganization/Repository_Management` under `docs/` by the root-level script migration; it was relocated here under [Repository_Management#1561](https://github.com/D-sorganization/Repository_Management/issues/1561).

## Scope

The review covered:

- **C3D reader** (`upstream_drift_tools/lab/bio/c3d_reader.py`)
- **URDF writer** (`model_generation/builders/urdf_writer.py`)
- **URDF generator** (`humanoid_character_builder/generators/urdf_generator.py`)
- **MJCF converter** (`model_generation/converters/mjcf_converter.py`)
- **Body parameters** (`humanoid_character_builder/core/body_parameters.py`)
- **Shared contracts** (`contracts.py`)
- Cross-cutting concerns: DRY, DbC, TDD, security, error handling

## Fixes Implemented

### 1. C3D Reader — Array Bounds (C-04)

**Problem:** `points_dataframe` indexed `points[3, :, :]` without checking that four channels exist. Files with only XYZ data (3 channels) caused an `IndexError`.

**Fix:** Added a bounds check on `points.shape[0]`. When fewer than 4 channels exist, residuals are filled with `NaN` and a warning is logged.

**Test:** `TestC3DReaderBoundsChecking` (2 tests)

### 2. C3D Reader — DRY Unit Conversion (M-06)

**Problem:** Two identical `to_meters` dictionaries existed in the same file, one containing a dead `mm^2` entry.

**Fix:** Consolidated into a single `_unit_scale(from_unit, to_unit)` class method with a canonical conversion table.

**Test:** `TestC3DUnitScaleDRY` (4 tests)

### 3. C3D Reader — Export Path Validation (H-07)

**Problem:** The export path validator checked whether `"pytest"` appeared in the file path string to detect test environments. This was fragile and could be bypassed.

**Fix:** Replaced with an explicit `C3D_ALLOW_ANY_EXPORT_PATH=1` environment variable. When unset, paths outside the current working directory are rejected with a `ValueError`.

**Test:** `TestC3DExportPathValidation` (2 tests)

### 4. C3D Reader — DbC Preconditions (M-07)

**Problem:** Public methods lacked input validation contracts.

**Fix:** Added `require()` calls (with graceful fallback if the contracts module is absent) to:

- `__init__` — rejects empty file paths
- `force_plate_dataframe` — rejects `plate_number < 1`

**Test:** `TestC3DDbCIntegration` (2 tests)

### 5. C3D Reader — CSV Formula Injection

**Problem:** Metadata values exported to CSV were not sanitised, allowing formula-injection attacks (`=cmd()`, `+1+1`, etc.).

**Fix:** Added `_sanitize_for_csv()` which prefixes dangerous characters (`=`, `+`, `-`, `@`) with a single quote. Applied to all metadata values during CSV export.

**Test:** `TestC3DMetadataSanitization` (1 test)

### 6. URDF Writer — Graph Validation (H-04)

**Problem:** `URDFWriter.write()` accepted any list of links and joints without checking that they form a valid tree. Cyclic or disconnected graphs produced invalid URDF.

**Fix:** Added `_validate_graph()` which:

- Builds a child-set from joints and identifies root links (links that are never a child)
- Raises `ValueError` if no root exists (cycle detected)
- Logs a warning if multiple roots exist
- Runs BFS from the first root and warns about unreachable links

Called at the start of `write()`.

**Test:** `TestURDFWriterGraphValidation` (2 tests)

### 7. URDF Writer — Material Name Collision (H-03)

**Problem:** Two links could define the same material name with different RGBA colours. The URDF spec defines materials globally by name, so the second definition silently won, producing incorrect visuals.

**Fix:** `_collect_materials()` now detects colour conflicts for the same material name and logs a warning.

**Test:** `TestURDFWriterMaterialCollision` (1 test)

### 8. URDF Writer — Mesh Path Traversal

**Problem:** Mesh filenames containing `..` could reference files outside the intended asset directory.

**Fix:** Added a warning when a mesh filename contains `..` and does not use the `package://` prefix.

### 9. URDF Writer — XML Escaping (C-01, pre-existing)

**Verified:** The existing `_escape()` method correctly handles `<`, `>`, `&`, `"` in robot names and other user-supplied strings. A regression test was added.

**Test:** `TestURDFWriterXMLEscaping` (1 test)

### 10. Body Parameters — Hard Bounds (H-02)

**Problem:** `BodyParameters` had soft validation (`validate()` returns warnings) but no hard enforcement. Physics engines could receive impossible values like negative height or mass.

**Fix:** Added `validate_strict()` which raises `ValueError` immediately on:

- `height_m` outside [0.3, 3.5] m
- `mass_kg` outside [1, 700] kg
- Any proportion factor negative or > 5.0

Also extended `validate()` to cover `neck_length_factor`, `hand_scale_factor`, and `foot_scale_factor` (previously missing from the factor check loop).

**Test:** `TestBodyParametersStrictValidation` (6 tests)

### 11. URDF Generator — Safety Hardening

**Problem:** The generator did not validate parameters before starting expensive computation, and directory creation lacked permission error handling.

**Fix:** Added `params.validate_strict()` call at the start of `generate()`, and wrapped `output_path.parent.mkdir()` in a try/except for `PermissionError`.

### 12. MJCF Converter — Capsule Parsing (H-14)

**Problem:** `_parse_mjcf_geom()` split the `fromto` attribute and indexed elements 0–5 without checking the count. Files with malformed `fromto` (e.g., only 3 values) caused `IndexError`. Zero-length capsules (identical endpoints) caused division-by-zero when computing length.

**Fix:**

- Added value-count check (requires ≥ 6 values)
- Added zero-length detection (falls back to sphere geometry)
- Logs warnings for both edge cases

**Test:** `TestMJCFCapsuleParsing` (3 tests)

## Test Summary

| Suite                      | Tests  | Status       |
| -------------------------- | ------ | ------------ |
| C3D Bounds Checking        | 2      | ✅ Pass      |
| C3D Export Path Validation | 2      | ✅ Pass      |
| C3D DbC Integration        | 2      | ✅ Pass      |
| C3D Metadata Sanitisation  | 1      | ✅ Pass      |
| C3D Unit Scale DRY         | 4      | ✅ Pass      |
| URDF Graph Validation      | 2      | ✅ Pass      |
| URDF XML Escaping          | 1      | ✅ Pass      |
| URDF Material Collision    | 1      | ✅ Pass      |
| Body Parameters Strict     | 6      | ✅ Pass      |
| MJCF Capsule Parsing       | 3      | ✅ Pass      |
| **Total**                  | **24** | **24/24 ✅** |

## Remaining Work (IDE Agent Tasks)

The following items from the full review require deeper refactoring and are better suited for an IDE-based agent session:

1. **DRY — Duplicate inertia primitives:** `model_generation/inertia/primitives.py` and `humanoid_character_builder/generators/mesh/primitive_inertia.py` contain overlapping implementations. Consolidate into the `model_generation` package.

2. **Rust parity:** Many pure-Python computation modules (inertia, anthropometry, kinematics) should have Rust equivalents in `rust_core/tools-core` with PyO3 bindings.

3. **Platform parity tracking:** Create `PLATFORM_PARITY.md` documenting feature coverage across PyQt6, React, and Tauri targets.

4. **Test coverage threshold:** Current minimum is 10% (pytest-cov). Consider raising progressively to 40% → 60% → 80%.

5. **Mypy strictness:** Several modules are excluded from type checking. Gradually enable `--strict` for core packages.

6. **Third-party integration stubs:** OpenSim, MyoSuite, Drake, Pinocchio integrations are skeletal. Flesh out with proper error handling and test fixtures.

7. **defusedxml adoption:** The MJCF converter imports `defusedxml` but some XML parsing paths still use `xml.etree.ElementTree` directly.
