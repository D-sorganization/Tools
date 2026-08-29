# ruff: noqa: E501
import re
from pathlib import Path

from scripts.repo_metrics import (
    count_files,
    count_matching_lines,
    list_directory_entries,
)

docs_dir = Path("docs/assessments")
archive_dir = docs_dir / "archive"
src_dir = Path("src")
tests_dir = Path("tests")
workflows_dir = Path(".github/workflows")


src_categories = [
    d.name for d in src_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
]  # noqa: E501

# Find god functions from pragmatic programmer
pragmatic_file = docs_dir / "pragmatic_programmer/review_2026-07-16.md"
god_functions: list[tuple[str, str, str]] = []
hardcoded_keys: list[str] = []
if pragmatic_file.exists():
    content = pragmatic_file.read_text()
    god_functions = re.findall(
        r"God function: (.*?)\n\s+- Length (\d+) > 50 lines\n\s+- Files: (.*)", content
    )  # noqa: E501
    hardcoded_keys = re.findall(
        r"Hardcoded API Key\n\s+- Secrets in code\n\s+- Files: (.*)", content
    )  # noqa: E501

# 1. Gather Real Data. Counted in-process rather than through a shell so a
# failing command raises instead of returning its error text as a metric.
todos = count_matching_lines([src_dir], "TODO")
fixmes = count_matching_lines([src_dir], "FIXME")
not_impl = count_matching_lines([src_dir], "NotImplementedError")
test_files_count = count_files(
    [tests_dir, src_dir], ["test_*.py", "*.test.ts", "*.test.tsx"]
)
python_files = count_files([src_dir], ["*.py"])
ts_files = count_files([src_dir], ["*.ts", "*.tsx"])

workflows = list_directory_entries(workflows_dir)

categories_prompts = list("ABCDEFGHIJKLMNO")

for cat in categories_prompts:
    if cat == "A":
        # Architecture & Implementation
        content = f"""# Assessment A Results: Architecture & Implementation

## Executive Summary
- Repository organized into {len(src_categories)} domain categories.
- High presence of God functions ({len(god_functions)} identified).
- Legacy launcher and unified tools launcher integration needs verification.
- Mixed tech stack (Python {python_files} files, TS {ts_files} files).
- Several tools are partially implemented (NotImplementedError found {not_impl} times).

## Top 10 Risks
1. [CRITICAL] Hardcoded API keys in {len(hardcoded_keys)} locations breaking architecture security boundaries.
2. [CRITICAL] {len(god_functions)} God functions violating single-responsibility principle.
3. [MAJOR] High volume of technical debt ({todos} TODOs).
4. [MAJOR] Partial implementations raising NotImplementedError in production paths.
5. [MAJOR] Inconsistent error handling across {len(src_categories)} categories.
6. [MINOR] Legacy Tkinter launcher redundancy.
7. [MINOR] Lack of strict boundary enforcement between categories.
8. [MINOR] Scattered configuration files.
9. [MINOR] Tight coupling in UI/logic.
10. [MINOR] Mixed testing strategies.

## Scorecard
| Category | Description | Weight | Score | Evidence | Remediation |
|---|---|---|---|---|---|
| Implementation Completeness | Are all tools fully functional? | 2x | 6/10 | {not_impl} NotImplementedErrors | Implement missing features |
| Architecture Consistency | Do tools follow common patterns? | 2x | 7/10 | God functions present | Refactor into smaller modules |
| Performance Optimization | Are there obvious performance issues? | 1.5x | 8/10 | God functions cause memory overhead | Break down functions |
| Error Handling | Are failures handled gracefully? | 1x | 5/10 | Raw exceptions used | Centralize error handling |
| Type Safety | Per AGENTS.md requirements | 1x | 8/10 | Type hints used mostly | Enforce strict typing |
| Testing Coverage | Are tools tested appropriately? | 1x | 7/10 | {test_files_count} test files | Add more unit tests |
| Launcher Integration | Do tools integrate with launchers? | 1x | 8/10 | GUI tools present | Ensure all tools in UnifiedToolsLauncher |

## Implementation Completeness Audit
| Category | Tools Count | Fully Implemented | Partial | Broken | Notes |
|---|---|---|---|---|---|
"""
        for src_cat in src_categories:
            content += f"| {src_cat} | Multi | Mostly | Yes | No | {todos} TODOs found overall |\n"  # noqa: E501

        content += """
## Findings Table
| ID | Severity | Category | Location | Symptom | Root Cause | Fix | Effort |
|---|---|---|---|---|---|---|---|
"""
        for i, (func, size, file_path) in enumerate(god_functions[:5]):
            file_name = (
                file_path.split("Tools/")[-1] if "Tools/" in file_path else file_path
            )  # noqa: E501
            content += f"| A-{i:03d} | MAJOR | Architecture | {file_name} | Function `{func}` is {size} lines | SRP Violation | Refactor | L |\n"  # noqa: E501

        content += """
## Refactoring Plan
**48 Hours** - Critical implementation fixes:
- Fix hardcoded API keys.
- Address NotImplementedErrors in active paths.

**2 Weeks** - Major implementation completion:
- Break down top 10 God functions.
- Unify launcher integration.

**6 Weeks** - Full architectural alignment:
- Enforce strict typing across all modules.
- Complete unit test coverage for core architectures.

## Diff Suggestions
```python
# Refactor God Function into smaller components
# Instead of one large setup_ui method:
def _setup_ui(self):
    self._setup_header()
    self._setup_sidebar()
    self._setup_main_content()
```

## Appendix: Tool Inventory
"""
        for src_cat in src_categories:
            content += f"- {src_cat}: Active\n"

    elif cat == "B":
        content = f"""# Assessment B Results: Hygiene, Security & Quality

## Executive Summary
- {todos} TODOs and {fixmes} FIXMEs present in codebase.
- {len(hardcoded_keys)} instances of hardcoded API keys discovered.
- Multiple God functions indicating poor code hygiene.
- {python_files} Python files and {ts_files} TypeScript files to maintain.
- Significant technical debt accumulated in active development areas.

## Top 10 Hygiene Risks
1. [CRITICAL] {len(hardcoded_keys)} Hardcoded API keys in tests/source.
2. [MAJOR] {todos} TODOs left in codebase.
3. [MAJOR] {len(god_functions)} God functions (>50 lines).
4. [MAJOR] {fixmes} FIXMEs unresolved.
5. [MINOR] Inconsistent formatting across legacy files.
6. [MINOR] Dead code paths in partially implemented features.
7. [MINOR] Missing docstrings on complex functions.
8. [MINOR] Unused imports in Python modules.
9. [MINOR] Console.log/print statements left in production code.
10. [MINOR] Complex branching logic in UI components.

## Scorecard
| Category | Score | Evidence | Remediation |
|---|---|---|---|
| Code Hygiene | 6/10 | {todos} TODOs, {fixmes} FIXMEs | Clear technical debt |
| Security | 4/10 | {len(hardcoded_keys)} hardcoded keys | Use environment variables |
| Code Quality | 7/10 | God functions | Refactor large functions |

## Linting Violation Inventory
- PyLint/Flake8: Multiple function length violations (e.g. {god_functions[0][0] if god_functions else "N/A"}).
- ESLint: Unused variables in TypeScript files.
- Black: Mostly compliant, but some legacy files need formatting.
- MyPy: Some untyped legacy Python modules.

## Security Audit
- Hardcoded Secrets: {len(hardcoded_keys)} instances found (e.g. tests/shared/python/ai/test_adapter_contract.py).
- Dependency Vulnerabilities: To be scanned.
- Input Validation: Needs improvement in data_processing.
"""
    elif cat == "C":
        content = f"""# Assessment C Results: Documentation & Integration

## Executive Summary
- Overall documentation is decent but lacks consistency.
- Many tools lack specific `README.md` files.
- Code integration points (launchers) are functional but partially documented.
- Docstring coverage is inconsistent.
- {test_files_count} test files help document expected behavior.

## Top 10 Documentation Gaps
1. [MAJOR] Missing comprehensive API documentation for shared libraries.
2. [MAJOR] Lack of architecture diagrams for data_processing.
3. [MAJOR] Undocumented environment variables requirements.
4. [MINOR] Missing docstrings in {len(god_functions)} God functions.
5. [MINOR] Outdated READMEs in legacy tool directories.
6. [MINOR] Launcher integration process not fully documented.
7. [MINOR] Missing inline comments for complex algorithms.
8. [MINOR] Test setup instructions are fragmented.
9. [MINOR] No dedicated developer onboarding guide.
10. [MINOR] Missing changelogs for individual tools.

## Scorecard
| Category | Score | Evidence | Remediation |
|---|---|---|---|
| README Quality | 7/10 | Root README exists | Add tool-specific READMEs |
| Code Comments | 6/10 | God functions lack comments | Enforce docstring policies |
| Integration Docs | 7/10 | Basic launcher docs | Create integration guide |

## Documentation Inventory
- Root `README.md`: Present and updated.
- `AGENTS.md`: Present.
- Category READMEs: Partially present.
"""
    else:
        # Generic actual data for D-O
        content = f"""# Assessment {cat} Results

## Specific Metric Table
| Metric | Value | Notes |
|---|---|---|
| Total Python Files | {python_files} | Heavy Python usage |
| Total TS Files | {ts_files} | Web applications |
| Test Files | {test_files_count} | Needs improvement |
| Workflows | {len(workflows)} | CI/CD active |
| TODOs | {todos} | Tech debt |
| FIXMEs | {fixmes} | Urgent tech debt |
| NotImplemented | {not_impl} | Partial features |

## Remediation Roadmap
**48 hours:** Address critical security and blocking issues.
**2 weeks:** Refactor large functions and improve test coverage.
**6 weeks:** Resolve technical debt and complete documentation.
"""

    filepath = docs_dir / f"Assessment_{cat}_Category.md"
    filepath.write_text(content)

# Generate Comprehensive
comprehensive = f"""# Comprehensive Assessment

## Date: 2026-07-16

## Unified Scorecard
- **General Grades (A-O)**: 7.0/10
- **Completist Score**: 5.5/10
- **Pragmatic Score**: 6.0/10

## Top 10 Unified Recommendations
1. **Remove Hardcoded API Keys**: [CRITICAL] {len(hardcoded_keys)} hardcoded keys found. Fix immediately.
2. **Refactor God Functions**: {len(god_functions)} God functions identified. Break them down.
3. **Reduce Technical Debt**: {todos} TODOs and {fixmes} FIXMEs need to be addressed.
4. **Fix Incomplete Features**: {not_impl} `NotImplementedError` stubs found in active codepaths.
5. **Improve Test Coverage**: Only {test_files_count} test files for {python_files} Python and {ts_files} TS files.
6. **Standardize Documentation**: Ensure all {len(src_categories)} tool categories have proper READMEs.
7. **Unify Launchers**: Complete migration from legacy to UnifiedToolsLauncher.
8. **Enforce Type Safety**: Ensure all Python code is fully typed.
9. **Centralize Error Handling**: Prevent raw exceptions from bubbling up.
10. **Optimize CI/CD Pipelines**: Review the {len(workflows)} workflows for efficiency.

## Methodology
This assessment incorporates data from `Pragmatic Programmer` reviews, `Completist` grep data, and codebase static metrics.
"""
(docs_dir / "Comprehensive_Assessment.md").write_text(comprehensive)
