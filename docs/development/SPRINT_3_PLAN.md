# Sprint 3: Fix 5000 DRY and Orthogonality Issues

## Goal
Fix 5000+ DRY violations and orthogonality issues across the codebase.

## Initial Analysis
Found **972+ potential violations** across multiple categories:

### High Priority (Target: ~2000 fixes)
1. **sys.path manipulations**: 41 occurrences in 40 files
   - Replace with `ensure_utils_in_path()` from path_helpers
   - Estimated fixes: 41 files × ~10 lines each = 410 fixes

2. **os.path operations**: 134 occurrences (join: 71, exists: 36, dirname: 27)
   - Replace with Path objects and os_utils functions
   - Estimated fixes: 134 × ~3 lines = 402 fixes

3. **pd.read_csv direct calls**: 18 occurrences in 6 files
   - Replace with csv_utils.safe_read_csv()
   - Estimated fixes: 18 × ~5 lines = 90 fixes

4. **logging.basicConfig**: 26 occurrences in 26 files
   - Consolidate to use logging_utils
   - Estimated fixes: 26 × ~8 lines = 208 fixes

5. **subprocess calls**: 45 occurrences (run: 26, Popen: 14, call: 5)
   - Replace with subprocess_utils functions
   - Estimated fixes: 45 × ~6 lines = 270 fixes

6. **json.load/dump**: 24 occurrences in 13 files
   - Replace with file_utils.safe_read_json/safe_write_json
   - Estimated fixes: 24 × ~4 lines = 96 fixes

### Medium Priority (Target: ~2000 fixes)
7. **try/except blocks**: 413 occurrences
   - Apply error_handling decorators where appropriate
   - Standardize error handling patterns
   - Estimated fixes: 413 × ~3 lines = 1239 fixes

8. **except Exception**: 222 occurrences
   - Make more specific or use error_handling utilities
   - Estimated fixes: 222 × ~2 lines = 444 fixes

9. **Path(__file__) patterns**: 49 occurrences
   - Use path_helpers.get_file_dir() or get_project_root_from_file()
   - Estimated fixes: 49 × ~2 lines = 98 fixes

10. **with open patterns**: 27 occurrences
    - Replace with file_utils functions
    - Estimated fixes: 27 × ~3 lines = 81 fixes

### Low Priority (Target: ~1000 fixes)
11. **Parent chains**: 6 occurrences
    - Replace with path_helpers functions
    - Estimated fixes: 6 × ~3 lines = 18 fixes

12. **Duplicate validation logic**: Various
    - Consolidate to validation.py
    - Estimated fixes: ~200 fixes

13. **Duplicate import patterns**: Various
    - Standardize imports
    - Estimated fixes: ~300 fixes

14. **Code quality issues**: Ruff warnings
    - Fix F401, F841, E501, W293, E402
    - Estimated fixes: ~500 fixes

## Strategy
1. **Batch Processing**: Fix patterns in batches by category
2. **Automation**: Use scripts where possible for repetitive fixes
3. **Testing**: Verify fixes don't break functionality
4. **Incremental Commits**: Commit in logical groups

## Progress Tracking
- Target: 5000 fixes
- Current: 0
- Remaining: 5000

## Files to Update
- ~200 Python files identified with violations
- Focus on high-impact files first
