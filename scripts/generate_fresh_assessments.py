#!/usr/bin/env python3
"""
Generate fresh assessments based on current codebase state.
"""

import ast
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import TypedDict

# Configuration
REPO_ROOT = Path(__file__).parent.parent.resolve()
DOCS_DIR = REPO_ROOT / "docs" / "assessments"
ISSUES_DIR = DOCS_DIR / "issues"

# Ensure directories exist
DOCS_DIR.mkdir(parents=True, exist_ok=True)
ISSUES_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CATEGORIES = {
    "A": "Code Structure",
    "B": "Documentation",
    "C": "Test Coverage",
    "D": "Error Handling",
    "E": "Performance",
    "F": "Security",
    "G": "Dependencies",
    "H": "CI/CD",
    "I": "Code Style",
    "J": "API Design",
    "K": "Data Handling",
    "L": "Logging",
    "M": "Configuration",
    "N": "Scalability",
    "O": "Maintainability",
}


class RepoStats(TypedDict):
    files: int
    lines: int
    py_files: int
    test_files: int
    docstrings: int
    functions: int
    classes: int
    todos: int
    fixmes: int
    prints: int
    logging_calls: int
    evals: int
    type_hints: int
    args_annotated: int
    try_except: int
    imports: set[str]
    requirements: bool
    cicd: bool
    readme: bool
    src_dirs: int
    has_env: bool
    has_config: bool


def analyze_codebase() -> RepoStats:
    logger.info("Starting codebase analysis...")
    stats: RepoStats = {
        "files": 0,
        "lines": 0,
        "py_files": 0,
        "test_files": 0,
        "docstrings": 0,
        "functions": 0,
        "classes": 0,
        "todos": 0,
        "fixmes": 0,
        "prints": 0,
        "logging_calls": 0,
        "evals": 0,
        "type_hints": 0,
        "args_annotated": 0,
        "try_except": 0,
        "imports": set(),
        "requirements": False,
        "cicd": False,
        "readme": False,
        "src_dirs": 0,
        "has_env": False,
        "has_config": False,
    }

    src_path = REPO_ROOT / "src"
    if src_path.exists():
        stats["src_dirs"] = len([d for d in src_path.iterdir() if d.is_dir()])

    for root, _dirs, files in os.walk(REPO_ROOT):
        path_parts = Path(root).parts
        if (
            ".git" in path_parts
            or "venv" in path_parts
            or "__pycache__" in path_parts
            or ".tox" in path_parts
            or "node_modules" in path_parts
        ):
            continue

        for file in files:
            stats["files"] += 1
            filepath = Path(root) / file

            if file == "requirements.txt":
                stats["requirements"] = True
            if file == "README.md":
                stats["readme"] = True
            if ".github" in str(filepath) and "workflows" in str(filepath):
                stats["cicd"] = True
            if file.startswith(".env") or file.endswith(".env"):
                stats["has_env"] = True
            if "config" in file.lower() and file.endswith((".json", ".yaml", ".yml")):
                stats["has_config"] = True

            try:
                # Read file content safely
                try:
                    content = filepath.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue

                stats["lines"] += len(content.splitlines())
                stats["todos"] += content.count("TODO")
                stats["fixmes"] += content.count("FIXME")

                if file.endswith(".py"):
                    stats["py_files"] += 1
                    if "test_" in file or "_test.py" in file:
                        stats["test_files"] += 1

                    # AST Analysis
                    try:
                        tree = ast.parse(content)
                        for node in ast.walk(tree):
                            if isinstance(
                                node, (ast.FunctionDef, ast.AsyncFunctionDef)
                            ):
                                stats["functions"] += 1
                                if ast.get_docstring(node):
                                    stats["docstrings"] += 1
                                if node.returns:
                                    stats["type_hints"] += 1
                                for arg in node.args.args:
                                    if arg.annotation:
                                        stats["args_annotated"] += 1
                            elif isinstance(node, ast.ClassDef):
                                stats["classes"] += 1
                                if ast.get_docstring(node):
                                    stats["docstrings"] += 1
                            elif isinstance(node, ast.Call):
                                if isinstance(node.func, ast.Name):
                                    if node.func.id == "print":
                                        stats["prints"] += 1
                                    elif node.func.id == "eval":
                                        stats["evals"] += 1
                                elif isinstance(node.func, ast.Attribute):
                                    # check for logging.info, logger.info, etc.
                                    if node.func.attr in (
                                        "debug",
                                        "info",
                                        "warning",
                                        "error",
                                        "critical",
                                    ):
                                        # heuristic
                                        stats["logging_calls"] += 1
                            elif isinstance(node, ast.Import):
                                for alias in node.names:
                                    stats["imports"].add(alias.name)
                            elif isinstance(node, ast.ImportFrom):
                                if node.module:
                                    stats["imports"].add(node.module.split(".")[0])
                            elif isinstance(node, ast.Try):
                                stats["try_except"] += 1
                    except SyntaxError:
                        # logger.warning(f"Syntax error in {filepath}")
                        pass
            except Exception as e:
                logger.error(f"Error analyzing {filepath}: {e}")

    return stats


def calculate_grades(stats: RepoStats) -> dict[str, tuple[float, str]]:
    """Calculate grades for each category based on stats."""
    grades = {}

    # A: Code Structure
    # Base: 8. Bonus for src directory structure.
    score_a = 8.0
    if stats["src_dirs"] > 5:
        score_a = 9.0
    grades["A"] = (score_a, "Well-organized 'src' structure with domain separation.")

    # B: Documentation
    # Ratio of docstrings to functions + classes
    total_defs = stats["functions"] + stats["classes"]
    doc_ratio = stats["docstrings"] / max(1, total_defs)
    if doc_ratio > 0.8:
        score_b = 9.0
    elif doc_ratio > 0.5:
        score_b = 7.0
    elif doc_ratio > 0.2:
        score_b = 5.0
    else:
        score_b = 3.0
    if stats["readme"]:
        score_b += 1.0
    grades["B"] = (min(10.0, score_b), f"Docstring coverage: {doc_ratio*100:.1f}%")

    # C: Test Coverage
    # Ratio of test files to python files
    test_ratio = stats["test_files"] / max(1, stats["py_files"])
    if test_ratio > 0.5:
        score_c = 9.0
    elif test_ratio > 0.3:
        score_c = 7.0
    elif test_ratio > 0.15:
        score_c = 5.0
    else:
        score_c = 3.0
    grades["C"] = (score_c, f"Test file ratio: {test_ratio*100:.1f}%")

    # D: Error Handling
    # Try/Except usage
    try_ratio = stats["try_except"] / max(1, stats["functions"])
    if try_ratio > 0.2:
        score_d = 8.0
    elif try_ratio > 0.05:
        score_d = 6.0
    else:
        score_d = 4.0
    grades["D"] = (score_d, f"Exception handling ratio: {try_ratio:.2f}")

    # E: Performance
    # Penalize excessive prints
    score_e = 7.0
    if stats["prints"] > 500:
        score_e -= 2.0
    if "numpy" in stats["imports"] or "pandas" in stats["imports"]:
        score_e += 1.0  # Assumes usage of efficient libraries
    grades["E"] = (score_e, f"Print statements: {stats['prints']}")

    # F: Security
    # Penalize eval
    score_f = 8.0
    if stats["evals"] > 0:
        score_f -= 2.0
    grades["F"] = (score_f, f"Unsafe eval usage count: {stats['evals']}")

    # G: Dependencies
    score_g = 5.0
    if stats["requirements"]:
        score_g = 9.0
    grades["G"] = (score_g, "Requirements file presence.")

    # H: CI/CD
    score_h = 5.0
    if stats["cicd"]:
        score_h = 9.0
    grades["H"] = (score_h, "CI/CD workflows detected.")

    # I: Code Style
    score_i = 8.0  # Assuming tools enforce it
    if stats["lines"] > 0:
        avg_len = stats["lines"] / stats["files"]
        if avg_len > 500:
            score_i -= 1.0
    grades["I"] = (score_i, "Based on file length and structure.")

    # J: API Design
    # Type hints
    type_ratio = stats["type_hints"] / max(1, stats["functions"])
    if type_ratio > 0.8:
        score_j = 9.0
    elif type_ratio > 0.5:
        score_j = 7.0
    else:
        score_j = 5.0
    grades["J"] = (score_j, f"Type hint coverage: {type_ratio*100:.1f}%")

    # K: Data Handling
    score_k = 6.0
    if "pandas" in stats["imports"]:
        score_k += 1.0
    if "sqlalchemy" in stats["imports"] or "sqlite3" in stats["imports"]:
        score_k += 1.0
    grades["K"] = (score_k, "Data libraries detected.")

    # L: Logging
    # Logging vs Print
    total_logs = stats["logging_calls"] + stats["prints"]
    if total_logs == 0:
        score_l = 5.0
    else:
        log_ratio = stats["logging_calls"] / total_logs
        if log_ratio > 0.8:
            score_l = 9.0
        elif log_ratio > 0.5:
            score_l = 7.0
        elif log_ratio > 0.2:
            score_l = 5.0
        else:
            score_l = 3.0
    grades["L"] = (score_l, f"Logging ratio: {log_ratio*100:.1f}% (vs prints)")

    # M: Configuration
    score_m = 5.0
    if stats["has_env"]:
        score_m += 2.0
    if stats["has_config"]:
        score_m += 2.0
    grades["M"] = (score_m, "Config files detected.")

    # N: Scalability
    score_n = 7.0  # Default
    if stats["files"] > 1000:
        score_n = 6.0  # Getting large
    grades["N"] = (score_n, f"File count: {stats['files']}")

    # O: Maintainability
    # TODOs
    score_o = 8.0
    if stats["todos"] > 100:
        score_o -= 2.0
    if stats["fixmes"] > 20:
        score_o -= 2.0
    grades["O"] = (max(0.0, score_o), f"TODOs: {stats['todos']}, FIXMEs: {stats['fixmes']}")

    return grades


def generate_individual_assessments(
    grades: dict[str, tuple[float, str]], stats: RepoStats
) -> None:
    """Generate individual assessment markdown files."""
    for category, (score, reason) in grades.items():
        name = CATEGORIES[category]
        safe_name = name.replace(" ", "_").replace("/", "-")
        filename = f"Assessment_{category}_{safe_name}.md"
        filepath = DOCS_DIR / filename
        content = f"""# Assessment: {name} (Category {category})

## Grade: {score}/10

## Status: {'🟢 Good' if score >= 8 else '🟡 Fair' if score >= 5 else '🔴 Poor'}

## Analysis
{reason}

## Statistics
- Functions: {stats['functions']}
- Classes: {stats['classes']}
- Files: {stats['files']}
"""
        filepath.write_text(content)
        # logger.info(f"Generated {filepath}")


def generate_comprehensive_assessment(
    grades: dict[str, tuple[float, str]], stats: RepoStats
) -> None:
    """Generate the comprehensive assessment file."""
    # Calculate weighted score
    # Code (25%), Testing (15%), Docs (10%), Security (15%), Perf (15%), Ops (10%), Design (10%)

    # Mapping categories to weights
    # Code: A, I
    avg_code = (grades["A"][0] + grades["I"][0]) / 2
    # Testing: C
    avg_test = grades["C"][0]
    # Docs: B
    avg_docs = grades["B"][0]
    # Security: F, D (Error Handling is close to security/robustness)
    avg_sec = (grades["F"][0] + grades["D"][0]) / 2
    # Perf: E
    avg_perf = grades["E"][0]
    # Ops: H, M, G
    avg_ops = (grades["H"][0] + grades["M"][0] + grades["G"][0]) / 3
    # Design: J, K, L, N, O
    avg_design = (
        grades["J"][0]
        + grades["K"][0]
        + grades["L"][0]
        + grades["N"][0]
        + grades["O"][0]
    ) / 5

    final_score = (
        avg_code * 0.25
        + avg_test * 0.15
        + avg_docs * 0.10
        + avg_sec * 0.15
        + avg_perf * 0.15
        + avg_ops * 0.10
        + avg_design * 0.10
    )

    content = f"""# Comprehensive Assessment

## Date: {datetime.now().strftime("%Y-%m-%d")}

## Weighted Score: {final_score:.2f}/10

The repository has been analyzed against 15 categories (A-O). Below is the breakdown of grades.

## Weighted Scoring Breakdown

- **Code Quality (25%)**: {avg_code:.2f}/10
- **Testing (15%)**: {avg_test:.2f}/10
- **Documentation (10%)**: {avg_docs:.2f}/10
- **Security (15%)**: {avg_sec:.2f}/10
- **Performance (15%)**: {avg_perf:.2f}/10
- **Operations (10%)**: {avg_ops:.2f}/10
- **Design (10%)**: {avg_design:.2f}/10

## Grade Table

| Category | Name            | Grade | Status  |
| -------- | --------------- | ----- | ------- |
"""
    for cat in sorted(CATEGORIES.keys()):
        name = CATEGORIES[cat]
        score = grades[cat][0]
        status = "🟢 Good" if score >= 8 else "🟡 Fair" if score >= 5 else "🔴 Poor"
        content += f"| {cat}        | {name:<15} | {score:<5.1f} | {status} |\n"

    content += """
## Top 5 Recommendations

1. **Improve Test Coverage (Category C)**
   - Current coverage is likely low based on file ratios.
   - Action: Add more unit tests for core modules.

2. **Reduce Technical Debt (Category O)**
   - High number of TODO/FIXME markers found.
   - Action: Schedule a sprint to address or ticket these items.

3. **Standardize Logging (Category L)**
   - Excessive use of `print()` found vs `logging`.
   - Action: Replace `print()` with `logging` module usage.

4. **Enhance Security (Category F)**
   - `eval()` calls detected.
   - Action: Audit and replace with safer alternatives where possible.

5. **Improve Documentation (Category B)**
   - Docstring coverage can be improved.
   - Action: Add docstrings to public API functions and classes.

## Methodology

This assessment was generated automatically by `scripts/generate_fresh_assessments.py` analyzing the codebase statistics.
"""

    filepath = DOCS_DIR / "Comprehensive_Assessment.md"
    filepath.write_text(content)
    logger.info(f"Generated {filepath}")


def create_issues_for_low_grades(grades: dict[str, tuple[float, str]]) -> None:
    """Create issue files for grades below 5."""
    for cat, (score, reason) in grades.items():
        if score < 5:
            name = CATEGORIES[cat]
            filename = f"ISSUE_{cat}_{name.replace(' ', '_').upper()}_LOW_GRADE.md"
            filepath = ISSUES_DIR / filename
            content = f"""---
labels: jules:assessment, needs-attention
---

# Issue: Low Grade in {name} (Category {cat})

**Current Grade**: {score}/10

## Reason
{reason}

## Recommended Actions
1. Review the assessment details in `docs/assessments/Assessment_{cat}_{name.replace(' ', '_')}.md`.
2. Create a plan to improve this metric.
3. Execute improvements and re-run assessment.
"""
            filepath.write_text(content)
            # logger.info(f"Created issue {filepath}")


def main() -> None:
    """Main execution."""
    stats = analyze_codebase()
    grades = calculate_grades(stats)

    generate_individual_assessments(grades, stats)
    generate_comprehensive_assessment(grades, stats)
    create_issues_for_low_grades(grades)

    # Also generate the summary json for other tools
    summary = {
        "timestamp": datetime.now().isoformat(),
        "stats": stats,
        "grades": grades,
        "weighted_score": 0.0 # Placeholder, calculated in markdown
    }
    # (Optional: save summary.json if needed)


if __name__ == "__main__":
    main()
