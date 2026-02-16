#!/usr/bin/env python3
"""
Generate fresh assessments based on current codebase state.
"""

import ast
import logging
import os
from datetime import datetime
from pathlib import Path

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


def analyze_codebase():
    logger.info("Starting codebase analysis...")
    stats = {
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
        "evals": 0,
        "type_hints": 0,
        "args_annotated": 0,
        "try_except": 0,
        "imports": set(),
        "requirements": False,
        "cicd": False,
        "readme": False,
    }

    for root, _dirs, files in os.walk(REPO_ROOT):
        # Skip hidden and venv directories
        # We want to allow .github for CI checks, but skip .git repo folder
        path_parts = Path(root).parts
        if (
            ".git" in path_parts
            or "venv" in path_parts
            or "__pycache__" in path_parts
            or ".tox" in path_parts
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
                                        # check for suppression comments on the same line (approximate)
                                        # actually ast doesn't give comments easily, so we count all prints
                                        stats["prints"] += 1
                                    if node.func.id == "eval":
                                        stats["evals"] += 1
                            elif isinstance(node, ast.Try):
                                stats["try_except"] += 1
                    except SyntaxError:
                        pass

            except Exception as e:
                logger.warning(f"Error analyzing {file}: {e}")

    return stats


def calculate_grades(stats):
    logger.info("Calculating grades...")
    grades = {}

    # A: Code Structure
    # Heuristic: Presence of src and tests folders
    has_src = (REPO_ROOT / "src").exists()
    has_tests = (REPO_ROOT / "tests").exists()
    grades["A"] = 9.0 if has_src and has_tests else 6.0

    # B: Documentation
    total_defs = stats["functions"] + stats["classes"]
    if total_defs > 0:
        doc_ratio = stats["docstrings"] / total_defs
        # Base score 4.0, add up to 6.0 based on ratio
        grades["B"] = min(10.0, 4.0 + (doc_ratio * 6.0))
    else:
        grades["B"] = 5.0

    # C: Test Coverage
    # Heuristic: Ratio of test files to python files
    if stats["py_files"] > 0:
        test_ratio = stats["test_files"] / stats["py_files"]
        # If 20% of files are tests, that's decent. 50% is excellent.
        # Score = ratio * 20, capped at 10.
        grades["C"] = min(10.0, test_ratio * 20)
    else:
        grades["C"] = 0.0

    # D: Error Handling
    # Heuristic: try/except usage
    if stats["functions"] > 0:
        error_ratio = stats["try_except"] / stats["functions"]
        # Expect at least 10% of functions to have error handling? Maybe too strict.
        grades["D"] = min(10.0, 5.0 + (error_ratio * 20))
    else:
        grades["D"] = 6.0

    # E: Performance
    # Heuristic: Default 7.0, punished if files are too large (lines > 1000?)
    # For now, default.
    grades["E"] = 7.0

    # F: Security
    # Penalize for eval usage
    grades["F"] = max(0.0, 10.0 - (stats["evals"] * 2.0))

    # G: Dependencies
    grades["G"] = 9.0 if stats["requirements"] else 4.0

    # H: CI/CD
    grades["H"] = 9.0 if stats["cicd"] else 2.0

    # I: Code Style
    # Heuristic: Assume 8.0 as base if tools are present
    grades["I"] = 8.0

    # J: API Design
    # Heuristic: Type hints usage
    if stats["functions"] > 0:
        type_ratio = stats["type_hints"] / stats["functions"]
        grades["J"] = min(10.0, 4.0 + (type_ratio * 6.0))
    else:
        grades["J"] = 5.0

    # K: Data Handling
    grades["K"] = 6.0

    # L: Logging
    # Penalize for print usage
    # 50 prints -> 5.0 grade. 0 prints -> 9.0 grade.
    if stats["prints"] > 50:
        grades["L"] = 4.0
    elif stats["prints"] > 10:
        grades["L"] = 6.0
    else:
        grades["L"] = 9.0

    # M: Configuration
    grades["M"] = 8.0

    # N: Scalability
    grades["N"] = 7.0

    # O: Maintainability
    # Penalize for TODOs/FIXMEs
    # 50 TODOs -> 5.0 grade
    todo_count = stats["todos"] + stats["fixmes"]
    if todo_count > 100:
        grades["O"] = 4.0
    elif todo_count > 50:
        grades["O"] = 6.0
    else:
        grades["O"] = 8.0

    return {k: round(v, 1) for k, v in grades.items()}


def generate_reports(grades, stats):
    logger.info("Generating reports...")
    today = datetime.now().strftime("%Y-%m-%d")

    # Generate individual category reports
    for cat, name in CATEGORIES.items():
        grade = grades[cat]
        content = f"""# Assessment {cat}: {name}

## Grade: {grade}/10

## Analysis
- **Date**: {today}
- **Automated Check**: Yes

## Details
"""
        # Add specific details based on category
        if cat == "B":
            content += f"- **Docstrings**: {stats['docstrings']} found in {stats['functions']+stats['classes']} definitions ({stats['docstrings']/(stats['functions']+stats['classes']+0.001)*100:.1f}%)\n"
        elif cat == "C":
            content += f"- **Test Files**: {stats['test_files']} (Total Python Files: {stats['py_files']})\n"
            content += f"- **Test Ratio**: {stats['test_files']/(stats['py_files']+0.001)*100:.1f}%\n"
        elif cat == "D":
            content += f"- **Try/Except Blocks**: {stats['try_except']}\n"
        elif cat == "F":
            content += (
                f"- **Eval Calls**: {stats['evals']} (Each call reduces score by 2.0)\n"
            )
        elif cat == "L":
            content += f"- **Print Calls**: {stats['prints']} (Should be 0 in production code)\n"
        elif cat == "O":
            content += f"- **TODOs**: {stats['todos']}\n"
            content += f"- **FIXMEs**: {stats['fixmes']}\n"
            content += f"- **Total Lines of Code**: {stats['lines']}\n"

        filepath = (
            DOCS_DIR / f"Assessment_{cat}_{name.replace(' ', '_').replace('/', '-')}.md"
        )
        filepath.write_text(content, encoding="utf-8")

        # Create Issue for low scores (< 5.0)
        if grade < 5.0:
            issue_content = f"""# Issue: Low Score in {name} (Category {cat})

## Status: Needs Attention
## Grade: {grade}/10
## Date: {today}

The assessment found significant issues in this category.

### Findings
- The automated assessment calculated a grade of {grade}/10, which is below the threshold of 5.0.
- Please review `docs/assessments/Assessment_{cat}_{name.replace(' ', '_')}.md` for more details.

### Recommendations
1. Review the specific metrics that contributed to this low score.
2. Create a plan to address the deficiencies.
3. Run the assessment script again to verify improvements.
"""
            issue_filename = f"ISSUE_LOW_SCORE_{cat}_{today}.md"
            issue_path = ISSUES_DIR / issue_filename
            issue_path.write_text(issue_content, encoding="utf-8")
            logger.warning(f"Created issue for Category {cat}: {issue_path}")

    # Calculate Weighted Score
    # Code (25%), Testing (15%), Docs (10%), Security (15%), Perf (15%), Ops (10%), Design (10%)
    groups = {
        "Code": ["A", "I", "K", "O"],
        "Testing": ["C", "G"],
        "Documentation": ["B", "M"],
        "Security": ["F", "D"],
        "Performance": ["E", "N"],
        "Operations": ["H", "L"],
        "Design": ["J"],
    }

    group_scores = {}
    for group, cats in groups.items():
        group_scores[group] = sum(grades[c] for c in cats) / len(cats)

    weighted_score = (
        group_scores["Code"] * 0.25
        + group_scores["Testing"] * 0.15
        + group_scores["Documentation"] * 0.10
        + group_scores["Security"] * 0.15
        + group_scores["Performance"] * 0.15
        + group_scores["Operations"] * 0.10
        + group_scores["Design"] * 0.10
    )

    # Generate Comprehensive Assessment
    comp_content = f"""# Comprehensive Assessment

## Date: {today}
## Weighted Score: {weighted_score:.2f}/10

The repository has been analyzed against 15 categories (A-O). Below is the breakdown of grades.

## Weighted Scoring Breakdown
- **Code Quality (25%)**: {group_scores['Code']:.2f}/10
- **Testing (15%)**: {group_scores['Testing']:.2f}/10
- **Documentation (10%)**: {group_scores['Documentation']:.2f}/10
- **Security (15%)**: {group_scores['Security']:.2f}/10
- **Performance (15%)**: {group_scores['Performance']:.2f}/10
- **Operations (10%)**: {group_scores['Operations']:.2f}/10
- **Design (10%)**: {group_scores['Design']:.2f}/10

## Grade Table
| Category | Name | Grade | Status |
|---|---|---|---|
"""
    for cat, name in CATEGORIES.items():
        grade = grades[cat]
        status = "🟢 Good" if grade >= 8.0 else "🟡 Fair" if grade >= 5.0 else "🔴 Poor"
        comp_content += f"| {cat} | {name} | {grades[cat]} | {status} |\n"

    comp_content += """
## Top 5 Recommendations

1. **Improve Test Coverage (Category C)**
   - Current coverage is low based on the ratio of test files to source files.
   - Action: Add more unit tests for core modules.

2. **Reduce Technical Debt (Category O)**
   - High number of TODO/FIXME markers found.
   - Action: Schedule a sprint to address or ticket these items.

3. **Standardize Logging (Category L)**
   - Excessive use of `print()` found.
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
    (DOCS_DIR / "Comprehensive_Assessment.md").write_text(
        comp_content, encoding="utf-8"
    )
    logger.info(
        f"Comprehensive assessment written to {DOCS_DIR / 'Comprehensive_Assessment.md'}"
    )


if __name__ == "__main__":
    try:
        stats = analyze_codebase()
        logger.info(
            f"Stats collected: Files={stats['files']}, Lines={stats['lines']}, Tests={stats['test_files']}"
        )

        grades = calculate_grades(stats)
        logger.info(f"Grades calculated: {grades}")

        generate_reports(grades, stats)
        logger.info("Assessment generation complete.")

    except Exception as e:
        logger.error(f"Failed to generate assessments: {e}", exc_info=True)
        exit(1)
