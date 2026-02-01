# Assessment F: Installation & Deployment

## Executive Summary
**Score: 4/10**
**Severity: CRITICAL**

Environment management is a critical weakness. Missing dependencies in `requirements.txt` cause tests to fail, and the reliance on complex system dependencies (MATLAB) makes deployment fragile.

## Key Findings

### 1. Dependency Management
- **Issue**: `ModuleNotFoundError: No module named 'pandas'` consistently appears in CI logs for tests, despite `requirements.txt` listing it. This suggests `PYTHONPATH` or environment isolation issues.
- **Issue**: `trimesh`, `scikit-image`, `networkx`, `rtree` are required for collision generation but missing from the base `requirements.txt`.

### 2. Path Handling
- **Strengths**: Recent fixes improved path sanitization.
- **Weaknesses**: Some legacy scripts still assume relative paths from specific directories, breaking when run from root.

### 3. System Requirements
- **Issue**: MATLAB R2020a+ requirement is a high barrier to entry.
- **Mitigation**: `MATLAB_REQUIREMENTS.md` documents this, but fallback mechanisms are limited.

## Recommendations
1. **Fix Requirements**: Audit and consolidate all `requirements.txt` files. Ensure `pandas` and `trimesh` dependencies are correctly propagated to the test environment.
2. **Containerization**: Create a `Dockerfile` to standardize the development and testing environment, eliminating "it works on my machine" issues.
3. **Mocking**: Fully mock MATLAB engines in tests to allow CI to pass without a licensed MATLAB installation.
