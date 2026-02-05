# Assessment K: Reproducibility & Provenance
**Date**: 2026-02-05
**Focus**: Determinism, versioning, experiment tracking

## 1. Findings Table

| Area | Status | Notes |
| :--- | :--- | :--- |
| **Versioning** | ❌ WEAK | The project lacks a unified versioning scheme (SemVer). Tools have individual versions (e.g., `_r0`) embedded in filenames or code. |
| **Dependency Lock** | ❌ MISSING | `requirements.txt` files list packages but rarely pin exact versions or hashes, leading to "works on my machine" issues. |
| **Randomness** | ⚠️ UNCHECKED | Tools involving generation (e.g., `model_generation`) do not appear to expose seed setting for deterministic output. |
| **Data Provenance** | ❌ NONE | Generated files (PDFs, plots) do not embed metadata about how they were created (version, settings). |

## 2. Critical Path Analysis
It is difficult to reproduce a specific output (e.g., a specific robot model) from a month ago because the code version, dependencies, and random seeds are not tracked.

## 3. Score
**Grade**: 4/10
**Justification**: Reproducibility is currently an afterthought. Lack of dependency locking is a critical flaw.

## 4. Recommendations
1.  **Lock Dependencies**: Generate `requirements.lock` or use `poetry.lock` to ensure identical environments.
2.  **SemVer**: Adopt Semantic Versioning for the repository and tag releases.
3.  **Metadata Embedding**: Modify export functions (PDF, URDF) to embed the tool version and configuration parameters in the output file.
