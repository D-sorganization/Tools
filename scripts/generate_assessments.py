import os

ASSESSMENT_DIR = "docs/assessments"
ISSUES_DIR = "docs/assessments/issues"

os.makedirs(ASSESSMENT_DIR, exist_ok=True)
os.makedirs(ISSUES_DIR, exist_ok=True)

assessments = {
    "A": ("Code Structure", 8, """
The repository exhibits a mature and well-organized structure.
- **Source Separation**: `src/` is cleanly separated from `tests/`, `docs/`, and configuration files.
- **Domain Segmentation**: Inside `src/`, code is logically grouped by domain (e.g., `data_processing`, `scientific_modeling`).
- **Hierarchy**: The nesting level is appropriate (max depth ~3-4 for core logic).
- **Consistency**: Most modules follow the `src/category/tool/python/package` pattern.
"""),
    "B": ("Documentation", 9, """
Documentation coverage is exceptional.
- **Docstrings**: Over 6,500 docstrings found across 646 files (~10 per file).
- **READMEs**: 35 README files cover almost every tool and category.
- **Guides**: Comprehensive guides in `docs/` (Architecture, Launchers, Plugin System).

## Auto-Fixes
- Added missing module docstrings to `src/verification/verify_palette.py`, `src/verification/verify_palette_final.py`, and `src/verification/verify_a11y.py`.
"""),
    "C": ("Test Coverage", 5, """
Test coverage is the primary weakness.
- **Ratio**: 119 test files for 646 source files (~18% ratio).
- **Gaps**: Many shared utilities and complex logic in `src/shared` appear under-tested.
- **Risk**: Low coverage increases regression risk during refactoring.
"""),
    "D": ("Error Handling", 7, """
Error handling follows standard practices.
- **Mechanisms**: Use of `try...except` blocks is visible in key areas.
- **UI**: GUI applications (PyQt) handle execution loops correctly (`sys.exit(app.exec())`).
- **Safety**: Some use of `eval` is wrapped in try-blocks or commented, though the usage itself is a risk (covered in Security).
"""),
    "E": ("Performance", 6, """
Performance is adequate but unoptimized.
- **Logging**: Heavy reliance on `print()` (700+) vs `logging` (1299+) impacts runtime performance and monitoring.
- **Imports**: Standard heavy imports (pandas, numpy) are used; no obvious lazy loading in critical paths observed.
- **Concurrency**: `launch_web.py` scripts use blocking subprocess calls in some places.
"""),
    "F": ("Security", 4, """
**CRITICAL FINDINGS**:
1.  **Data Leakage**: `.msg` (Outlook email) files found in `src/shared/python/upstream_drift_tools/...`. This is a major PII/IP risk.
2.  **Unsafe Functions**: `eval()` usage detected in `Data_Processor_r0.py`, `signal_processing.py`, and `fitting.py`.
3.  **Shell Injection**: Extensive use of `shell=True` in launcher scripts.
4.  **SAST**: CodeQL workflow is present but disabled (`codeql-analysis.yml.disabled`).
"""),
    "G": ("Dependencies", 9, """
Dependency management is very strong.
- **Manifests**: Clean `requirements.txt` with inline comments explaining usage.
- **Locking**: `requirements-lock.txt` and `pnpm-lock.yaml` ensure reproducible builds.
- **Isolation**: Virtual environment usage is enforced/encouraged in docs.
"""),
    "H": ("CI/CD", 9, """
CI/CD is robust and extensive.
- **Workflows**: Over 40 GitHub Actions workflows covering everything from linting (`ci-standard.yml`) to stale issue cleanup.
- **Gates**: Strict quality gates for formatting (Black), linting (Ruff), and types (MyPy).
- **Automation**: "Jules" agent automation is highly integrated.
"""),
    "I": ("Code Style", 8, """
Code style is strictly enforced.
- **Tooling**: `ruff` and `black` are used in CI, ensuring consistent formatting.
- **Typing**: `mypy` is configured and used, though some `type: ignore` usage was spotted.
- **Conventions**: Variable naming and structure generally follow PEP 8.
"""),
    "J": ("API Design", 7, """
API design is modular but implicit.
- **Modularity**: Tools are well-separated.
- **Contracts**: `src/shared` provides reusable components, but explicit interfaces (Protocols/ABCs) could be stronger to enforce contracts.
- **REST**: Web apps use standard REST patterns.
"""),
    "K": ("Data Handling", 6, """
Data handling is mixed.
- **I/O**: Standard pandas/numpy usage for data processing.
- **Safety**: The presence of `.msg` files indicates poor hygiene regarding binary/personal data committing.
- **Validation**: Input validation in web apps is present but could be more robust.
"""),
    "L": ("Logging", 6, """
Logging is inconsistent.
- **Hybrid**: The codebase is split between `print()` (debugging style) and `logging` (production style).
- **Standardization**: Need to migrate all `print()` statements in `src/` to the shared logger.
"""),
    "M": ("Configuration", 8, """
Configuration management is good.
- **Environment**: `.env` and `.env.example` usage is documented.
- **Files**: Config files (JSON, YAML) are used appropriately.
- **Launchers**: Launchers handle configuration loading dynamically.
"""),
    "N": ("Scalability", 7, """
The architecture supports scaling.
- **Plugin System**: The `core/plugin_manager.py` allows easy addition of new tools.
- **Monorepo**: The structure supports adding many tools without clutter, though checking out the whole repo is heavy.
"""),
    "O": ("Maintainability", 5, """
Technical debt is accumulating.
- **Markers**: 445 `TODO` and 140 `FIXME` markers indicate significant unfinished work.
- **Legacy**: Existence of "legacy" launchers (`Launcher.py`, `launch_tools_main.py`) alongside `UnifiedToolsLauncher.py` creates confusion (though documented).
""")
}

# Generate Category Assessments
for code, (name, grade, analysis) in assessments.items():
    safe_name = name.replace(' ', '_').replace('/', '-')
    filename = f"Assessment_{code}_{safe_name}.md"
    filepath = os.path.join(ASSESSMENT_DIR, filename)
    with open(filepath, "w") as f:
        f.write(f"# Assessment: {name} (Category {code})\n\n")
        f.write(f"## Grade: {grade}/10\n\n")
        f.write("## Analysis\n")
        f.write(analysis.strip() + "\n")
    print(f"Generated {filepath}")

# Generate Issues for Low Grades (< 5)
issues = {
    "F": ("Security", "CRITICAL: Data Leakage and Unsafe Eval Usage", """
The security assessment identified critical vulnerabilities:
1.  **Data Leakage**: Binary Outlook `.msg` files containing email correspondence are present in `src/shared/python/upstream_drift_tools/...`.
2.  **Unsafe Code**: `eval()` is used in `Data_Processor_r0.py` and others without sufficient sanitization.
3.  **SAST**: CodeQL is disabled.

**Action Items**:
-   Remove `.msg` files from history (BFG/filter-branch).
-   Refactor `eval()` usage to use `ast.literal_eval` or a math parser library.
-   Enable CodeQL workflow.
"""),
    "C": ("Test Coverage", "Low Test Coverage (18%)", """
Test coverage is significantly below industry standards.
-   Only 119 test files for 646 source files.
-   Critical shared libraries in `src/shared` lack comprehensive unit tests.

**Action Items**:
-   Enforce strict TDD for new features.
-   Add unit tests for `src/shared/python` utilities.
-   Target 60% file coverage ratio.
"""),
    "O": ("Maintainability", "High Technical Debt (445 TODOs)", """
The codebase has accumulated significant technical debt.
-   445 `TODO` markers.
-   140 `FIXME` markers.

**Action Items**:
-   Audit all `FIXME` items and resolve high-priority ones.
-   Convert valid `TODO` items into GitHub Issues.
-   Remove obsolete code.
""")
}

for code, (name, title, body) in issues.items():
    if assessments[code][1] <= 5:
        filename = f"Issue_{code}_{name.replace(' ', '_')}.md"
        filepath = os.path.join(ISSUES_DIR, filename)
        with open(filepath, "w") as f:
            f.write("--- \nlabels: jules:assessment, needs-attention\n---\n\n")
            f.write(f"# {title}\n\n")
            f.write(body.strip() + "\n")
        print(f"Generated {filepath}")

# Generate Comprehensive Assessment
# Calculate dynamic scores
def get_avg(keys):
    values = [assessments[k][1] for k in keys]
    return sum(values) / len(values)

avg_code = get_avg(["A", "I"])
avg_test = get_avg(["C"])
avg_docs = get_avg(["B"])
avg_sec = get_avg(["F", "D"])
avg_perf = get_avg(["E"])
avg_ops = get_avg(["H", "M", "G"])
avg_design = get_avg(["J", "K", "L", "N", "O"])

final_score = (
    avg_code * 0.25 +
    avg_test * 0.15 +
    avg_docs * 0.10 +
    avg_sec * 0.15 +
    avg_perf * 0.15 +
    avg_ops * 0.10 +
    avg_design * 0.10
)

with open(os.path.join(ASSESSMENT_DIR, "Comprehensive_Assessment.md"), "w") as f:
    f.write("# Comprehensive Repository Assessment\n\n")
    f.write(f"## Weighted Score: {final_score:.2f}/10\n\n")
    f.write("The repository demonstrates high standards in automation, tooling, and code style, but is held back by specific security hygiene issues (data leakage) and low test coverage.\n\n")
    f.write("## Grade Table\n")
    f.write("| Category | Name | Grade |\n")
    f.write("|----------|------|-------|\n")
    for code, (name, grade, _) in sorted(assessments.items()):
        f.write(f"| {code} | {name} | {grade}/10 |\n")

    f.write("\n## Weighted Scoring Breakdown\n")
    f.write(f"- **Code Quality (25%)**: {avg_code:.2f}/10\n")
    f.write(f"- **Testing (15%)**: {avg_test:.2f}/10\n")
    f.write(f"- **Documentation (10%)**: {avg_docs:.2f}/10\n")
    f.write(f"- **Security (15%)**: {avg_sec:.2f}/10\n")
    f.write(f"- **Performance (15%)**: {avg_perf:.2f}/10\n")
    f.write(f"- **Operations (10%)**: {avg_ops:.2f}/10\n")
    f.write(f"- **Design (10%)**: {avg_design:.2f}/10\n")

    f.write("\n## Top 5 Recommendations\n\n")
    f.write("1.  **URGENT: Data Leakage Cleanup (Category F)**\n")
    f.write("    - **Issue**: Binary Outlook `.msg` files containing email correspondence are present in the repository.\n")
    f.write("    - **Action**: Immediately remove these files from the git history and file system.\n\n")

    f.write("2.  **Increase Test Coverage (Category C)**\n")
    f.write("    - **Issue**: Only ~18% test file ratio.\n")
    f.write("    - **Action**: Implement a requirement for unit tests for all new code in `src/shared`.\n\n")

    f.write("3.  **Secure Eval Usage (Category F)**\n")
    f.write("    - **Issue**: Unsafe `eval()` usage in data processing tools.\n")
    f.write("    - **Action**: Replace `eval()` with safer alternatives like `ast.literal_eval` or expression parsers.\n\n")

    f.write("4.  **Pay Down Technical Debt (Category O)**\n")
    f.write("    - **Issue**: 445 `TODO` markers.\n")
    f.write("    - **Action**: Conduct a specific sprint to resolve or ticket these items.\n\n")

    f.write("5.  **Standardize Logging (Category L)**\n")
    f.write("    - **Issue**: Mixed use of `print()` and `logging`.\n")
    f.write("    - **Action**: Enforce a linting rule to ban `print()` in library code.\n")

print(f"Generated {os.path.join(ASSESSMENT_DIR, 'Comprehensive_Assessment.md')}")
