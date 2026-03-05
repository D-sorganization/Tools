# Assessment N: Scalability

## Executive Summary
This assessment evaluates the repository's ability to handle an increasing number of tools, larger datasets, and more complex mathematical models.
The monorepo architecture and internal plugin system (`src/core/plugin_manager.py`) provide a strong foundation for adding new tools without cluttering the global namespace. However, as the repository has grown to 2464 files, checkout times and IDE indexing are becoming noticeably slower. From a computational perspective, scaling to larger datasets will require shifting from in-memory processing to out-of-core strategies.

## Scorecard
- **Grade: 8.0/10**

## Findings Table
| ID | Severity | Category | Location | Symptom | Root Cause | Fix | Effort |
|----|----------|----------|----------|---------|------------|-----|--------|
| N-001 | Major | Repo Scale | `.git` history | Slow shallow clones / checkouts | Large binary assets checked in | Use Git LFS or submodules for test data / media | M |
| N-002 | Medium | Computation | `src/shared/python/signal_toolkit/` | CPU bottleneck on large arrays | Python loops over numpy | Vectorize loops or use `numba` / `Cython` | M |
| N-003 | Medium | Build Scale | `build_exe.py` (Folder tools) | PyInstaller creates massive executables | Redundant packaged environments | Optimize `.spec` files to exclude `pandas`/`numpy` if unused | M |
| N-004 | Minor | Database | `src/media_processing/` | No local database scaling | Relying on JSON state (TODO) | Implement `SQLite` or `DuckDB` | L |

## Refactoring Plan
- **Short Term**: Audit the `pyinstaller` scripts in the `folder_tools` to drastically reduce the footprint of compiled `.exe` files by explicitly excluding heavy data science libraries that aren't needed for file management (N-003).
- **Medium Term**: Identify performance bottlenecks in `signal_toolkit` and apply `@numba.jit` decorators to mathematically intensive, non-vectorizable loops (N-002).
- **Long Term**: Migrate the entire repository's test data and static assets (like images, `.mat` files) to Git LFS. This will keep the core repository checkout light and fast, resolving N-001.
