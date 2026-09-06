# Chapter 10 — Development and Utility Tools

**Parent Document:** [Tools User Manual](./TOOLS_USER_MANUAL.md)

---

## 10.1 Folder Tool

**Source:** `src/folder_tool/`
**Status:** ✅ Implemented

### 10.1.1 Purpose

Directory management utility for organizing, sorting, and managing file structures.

### 10.1.2 Capabilities

- Directory tree visualization
- File categorization by type
- Batch file operations
- Directory comparison
- Size analysis

---

## 10.2 Folder Packer Pro

**Source:** `src/folder_packer_pro/`
**Status:** ✅ Implemented

### 10.2.1 Purpose

Advanced directory packing tool for creating compressed archives of project directories with configurable exclusion patterns.

### 10.2.2 Features

- Intelligent file filtering (.gitignore compatible)
- Compression optimization
- Executable builder (`build_exe.py` for PyInstaller)
- Progress tracking

---

## 10.3 Project Packer

**Source:** `src/project_packer/`
**Status:** ✅ Implemented

### 10.3.1 Purpose

Packages entire project directories into distributable formats, handling dependencies and configuration.

### 10.3.2 Components

| Module              | Description                     |
| ------------------- | ------------------------------- |
| `constants.py`      | Packing configuration constants |
| `build_exe.py`      | Executable build script         |
| `project_packer.py` | Core packing logic              |

---

## 10.4 Quality Utilities

**Source:** `src/tools/quality_utils.py`
**Status:** ✅ Fully Implemented

### 10.4.1 Purpose

Code quality scanning utilities used by CI/CD pipelines and pre-commit hooks to detect placeholder code, security issues, and code smell patterns.

### 10.4.2 Scan Patterns

| Category    | Pattern                     | Description                  |
| ----------- | --------------------------- | ---------------------------- |
| Placeholder | `TRACKED_TASK`              | TRACKED_TASK comment found   |
| Placeholder | `TRACKED_DEFECT`            | TRACKED_DEFECT comment found |
| Placeholder | `^\s*\.\.\.\s*$`            | Ellipsis placeholder         |
| Placeholder | `NotImplementedError`       | Unimplemented function       |
| Placeholder | `raise NotImplementedError` | Explicitly unimplemented     |
| Placeholder | `pass` (in function)        | Empty function body          |
| Security    | `eval(`                     | Unsafe eval usage            |
| Security    | `exec(`                     | Unsafe exec usage            |
| Security    | `__import__`                | Dynamic import               |
| Quality     | `print(` (in library code)  | Debug print left in code     |

### 10.4.3 CI Integration

The quality scanner is integrated into GitHub Actions workflows to automatically flag:

- Placeholder code that should be implemented
- Security vulnerabilities
- Code patterns that need review

---

## 10.5 Dependency Utilities

**Source:** `src/tools/dependency_utils.py`
**Status:** ✅ Implemented

### 10.5.1 Purpose

Manages and validates Python package dependencies across the monorepo, checking for version conflicts and missing packages.

---

## 10.6 MATLAB Utilities

**Source:** `src/tools/matlab_quality_utils.py`
**Status:** ✅ Implemented

### 10.6.1 Purpose

Quality scanning for MATLAB (.m) files, analogous to the Python quality utilities.

---

## 10.7 Verification Tools

**Source:** `src/verification/`
**Status:** ✅ Implemented

### 10.7.1 Purpose

Verification scripts for validating theme compliance, color palette correctness, and accessibility standards.

### 10.7.2 Components

| Script                    | Description                           |
| ------------------------- | ------------------------------------- |
| `verify_palette.py`       | Validates color palette definitions   |
| `verify_palette_final.py` | Final palette verification            |
| `verify_a11y.py`          | Accessibility (WCAG 2.1) verification |

### 10.7.3 Accessibility Standards

The verification tools check compliance with WCAG 2.1 Level AA:

| Criterion                    | Requirement             | Check |
| ---------------------------- | ----------------------- | ----- |
| Contrast Ratio (Normal Text) | $\geq 4.5:1$            | ✅    |
| Contrast Ratio (Large Text)  | $\geq 3.0:1$            | ✅    |
| Color Independence           | Not sole differentiator | ✅    |

**Contrast Ratio Formula:**

$$CR = \frac{L_1 + 0.05}{L_2 + 0.05}$$

where $L_1$ and $L_2$ are the relative luminances of the lighter and darker colors respectively.

**Relative Luminance:**

$$L = 0.2126 \cdot R_{lin} + 0.7152 \cdot G_{lin} + 0.0722 \cdot B_{lin}$$

where each linear color component is computed as:

$$C_{lin} = \begin{cases} C_{sRGB} / 12.92 & C_{sRGB} \leq 0.04045 \\ \left(\frac{C_{sRGB} + 0.055}{1.055}\right)^{2.4} & C_{sRGB} > 0.04045 \end{cases}$$

---

_[← Media Processing](./09_media_processing.md) | [Back to Manual](./TOOLS_USER_MANUAL.md) | [Next: Constants & Conversions →](./11_constants_conversions.md)_
