#!/usr/bin/env python3
"""
Generate comprehensive assessment reports for Categories A-O, Completist Audit, and Pragmatic Review.
"""

import ast
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import TypedDict

# Configuration
REPO_ROOT = Path(__file__).parent.parent.resolve()
DOCS_DIR = REPO_ROOT / "docs" / "assessments"
COMPLETIST_DIR = REPO_ROOT / ".jules" / "completist_data"
PRAGMATIC_DIR = DOCS_DIR / "pragmatic_programmer"

# Ensure output directories exist
DOCS_DIR.mkdir(parents=True, exist_ok=True)
(DOCS_DIR / "completist").mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CATEGORIES = {
    "A": {"name": "Architecture & Implementation", "weight": 2.0},
    "B": {"name": "Code Quality & Hygiene", "weight": 1.5},
    "C": {"name": "Documentation & Comments", "weight": 1.0},
    "D": {"name": "User Experience & Developer Journey", "weight": 2.0},
    "E": {"name": "Performance & Scalability", "weight": 1.5},
    "F": {"name": "Installation & Deployment", "weight": 1.5},
    "G": {"name": "Testing & Validation", "weight": 2.0},
    "H": {"name": "Error Handling & Debugging", "weight": 1.5},
    "I": {"name": "Security & Input Validation", "weight": 1.5},
    "J": {"name": "Extensibility & Plugin Architecture", "weight": 1.0},
    "K": {"name": "Reproducibility & Provenance", "weight": 1.5},
    "L": {"name": "Long-Term Maintainability", "weight": 1.0},
    "M": {"name": "Educational Resources & Tutorials", "weight": 1.0},
    "N": {"name": "Visualization & Export", "weight": 1.0},
    "O": {"name": "CI/CD & DevOps", "weight": 1.0},
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
    evals: int
    type_hints: int
    args_annotated: int
    try_except: int
    imports: set[str]
    requirements: bool
    cicd: bool
    readme: bool
    god_functions: list[str]
    dry_violations: list[str]
    not_implemented: int
    abstract_methods: int
    incomplete_docs: int
    ui_files: int
    performance_imports: int
    visualization_imports: int
    has_setup_py: bool
    has_pyproject: bool
    has_examples: bool
    has_docs_folder: bool


class AssessmentGenerator:
    def __init__(self):
        self.stats: RepoStats = {
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
            "god_functions": [],
            "dry_violations": [],
            "not_implemented": 0,
            "abstract_methods": 0,
            "incomplete_docs": 0,
            "ui_files": 0,
            "performance_imports": 0,
            "visualization_imports": 0,
            "has_setup_py": False,
            "has_pyproject": False,
            "has_examples": False,
            "has_docs_folder": False
        }
        self.scores: dict[str, float | None] = {}

    def analyze_codebase(self):
        logger.info("Starting codebase analysis...")

        if (REPO_ROOT / "docs").exists():
            self.stats["has_docs_folder"] = True

        if (REPO_ROOT / "examples").exists():
            self.stats["has_examples"] = True

        # Walk file tree
        for root, _dirs, files in os.walk(REPO_ROOT):
            # Skip hidden and venv directories
            path_parts = Path(root).parts
            if any(p.startswith(".") and p != ".github" for p in path_parts) or "venv" in path_parts or "__pycache__" in path_parts:
                continue

            for file in files:
                self.stats["files"] += 1
                filepath = Path(root) / file

                if file == "requirements.txt":
                    self.stats["requirements"] = True
                if file == "README.md":
                    self.stats["readme"] = True
                if file == "setup.py":
                    self.stats["has_setup_py"] = True
                if file == "pyproject.toml":
                    self.stats["has_pyproject"] = True

                if file.endswith((".ui", ".qml")):
                    self.stats["ui_files"] += 1

                if ".github" in str(filepath) and "workflows" in str(filepath):
                    self.stats["cicd"] = True

                try:
                    content = filepath.read_text(encoding="utf-8", errors="ignore")
                    self.stats["lines"] += len(content.splitlines())

                    # Basic grep counts
                    self.stats["todos"] += content.count("TODO")
                    self.stats["fixmes"] += content.count("FIXME")

                    if file.endswith(".py"):
                        self.analyze_python_file(filepath, content)

                except Exception as e:
                    logger.debug(f"Error reading {filepath}: {e}")

    def analyze_python_file(self, filepath: Path, content: str):
        self.stats["py_files"] += 1
        if "test_" in filepath.name or "_test.py" in filepath.name:
            self.stats["test_files"] += 1

        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.stats["functions"] += 1
                    if ast.get_docstring(node):
                        self.stats["docstrings"] += 1
                    if node.returns:
                        self.stats["type_hints"] += 1
                    for arg in node.args.args:
                        if arg.annotation:
                            self.stats["args_annotated"] += 1
                elif isinstance(node, ast.ClassDef):
                    self.stats["classes"] += 1
                    if ast.get_docstring(node):
                        self.stats["docstrings"] += 1
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id == "print":
                            self.stats["prints"] += 1
                        elif node.func.id == "eval":
                            self.stats["evals"] += 1
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        self.stats["imports"].add(alias.name)
                        self._check_special_imports(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        mod_name = node.module.split('.')[0]
                        self.stats["imports"].add(mod_name)
                        self._check_special_imports(mod_name)
                elif isinstance(node, ast.Try):
                    self.stats["try_except"] += 1
        except SyntaxError:
            logger.warning(f"Syntax error in {filepath}")

    def _check_special_imports(self, name: str):
        if name in ["cProfile", "timeit", "profile", "pstats"]:
            self.stats["performance_imports"] += 1
        if name in ["matplotlib", "seaborn", "plotly", "altair", "bokeh"]:
            self.stats["visualization_imports"] += 1
        if name == "PyQt6" or name == "tkinter":
            self.stats["ui_files"] += 1 # Count as UI related

    def load_completist_data(self):
        logger.info("Loading completist data...")
        try:
            with open(COMPLETIST_DIR / "todo_markers.txt") as f:
                self.stats["todos"] = len(f.readlines())  # Override with exact count
            with open(COMPLETIST_DIR / "not_implemented.txt") as f:
                self.stats["not_implemented"] = len(f.readlines())
            with open(COMPLETIST_DIR / "abstract_methods.txt") as f:
                self.stats["abstract_methods"] = len(f.readlines())
            with open(COMPLETIST_DIR / "incomplete_docs.txt") as f:
                self.stats["incomplete_docs"] = len(f.readlines())
        except FileNotFoundError:
            logger.warning("Completist data not found. Run completist scan first.")

    def load_pragmatic_data(self):
        logger.info("Loading pragmatic programmer review...")
        review_file = PRAGMATIC_DIR / "review_2026-02-19.md"
        if not review_file.exists():
            logger.warning("Pragmatic review not found.")
            return

        content = review_file.read_text(encoding="utf-8")

        # Extract DRY violations
        dry_matches = re.findall(r"\*\*DRY\*\* \[MAJOR\]: Duplicate code block", content)
        self.stats["dry_violations"] = dry_matches

        # Extract God Functions
        god_matches = re.findall(r"\*\*ORTHOGONALITY\*\* \[MAJOR\]: God function: (.*)", content)
        self.stats["god_functions"] = god_matches

    def calculate_scores(self):
        logger.info("Calculating scores...")

        # A: Architecture
        god_penalty = len(self.stats["god_functions"]) * 0.5
        score_a = max(0, 10 - god_penalty)
        self.scores["A"] = round(score_a, 1)

        # B: Code Quality
        type_coverage = self.stats["type_hints"] / max(1, self.stats["functions"])
        print_penalty = min(2, self.stats["prints"] / 100)
        score_b = (type_coverage * 10) - print_penalty
        self.scores["B"] = round(max(0, min(10, score_b)), 1)

        # C: Documentation
        doc_coverage = self.stats["docstrings"] / max(1, self.stats["functions"])
        score_c = doc_coverage * 10
        self.scores["C"] = round(max(0, min(10, score_c)), 1)

        # D: UX
        # Score based on UI file presence
        if self.stats["ui_files"] > 0:
             self.scores["D"] = None # Manual Review Required
        else:
             self.scores["D"] = None # N/A

        # E: Performance
        if self.stats["performance_imports"] > 0:
            self.scores["E"] = 7.0 # Optimistically score presence of tooling
        else:
            self.scores["E"] = 5.0 # Average default if no tools found

        # F: Installation
        has_install_files = self.stats["requirements"] or self.stats["has_setup_py"] or self.stats["has_pyproject"]
        self.scores["F"] = 9.0 if has_install_files else 2.0

        # G: Testing
        test_ratio = self.stats["test_files"] / max(1, self.stats["py_files"])
        score_g = min(10, test_ratio * 20)
        self.scores["G"] = round(max(0, score_g), 1)

        # H: Error Handling
        try_ratio = self.stats["try_except"] / max(1, self.stats["functions"])
        score_h = min(10, try_ratio * 50)
        self.scores["H"] = round(max(0, score_h), 1)

        # I: Security
        eval_penalty = self.stats["evals"] * 1.0
        score_i = max(0, 10 - eval_penalty)
        self.scores["I"] = round(score_i, 1)

        # J: Extensibility
        self.scores["J"] = None # Manual Review

        # K: Reproducibility
        self.scores["K"] = 8.0 if self.stats["requirements"] else 4.0

        # L: Maintainability
        dry_penalty = len(self.stats["dry_violations"]) * 0.2
        score_l = max(0, 10 - dry_penalty)
        self.scores["L"] = round(score_l, 1)

        # M: Education
        if self.stats["has_docs_folder"] or self.stats["has_examples"]:
             self.scores["M"] = 7.0
        else:
             self.scores["M"] = 3.0

        # N: Visualization
        if self.stats["visualization_imports"] > 0:
             self.scores["N"] = 8.0
        else:
             self.scores["N"] = None # Manual Review

        # O: CI/CD
        self.scores["O"] = 9.0 if self.stats["cicd"] else 2.0

    def generate_category_reports(self):
        logger.info("Generating category reports...")
        date_str = datetime.now().strftime("%Y-%m-%d")

        for cat_id, info in CATEGORIES.items():
            filename = f"Assessment_{cat_id}_Results_{date_str}.md"
            filepath = DOCS_DIR / filename

            raw_score = self.scores.get(cat_id)
            score_display = f"{raw_score}/10" if raw_score is not None else "Manual Review Required"

            content = f"""# Assessment {cat_id}: {info['name']}

**Date**: {date_str}
**Score**: {score_display}
**Weight**: {info['weight']}x

## Executive Summary
This assessment evaluates the repository's status regarding **{info['name']}**.
The calculated score is **{score_display}**.

## Key Findings

### Strengths
- [Automated] Analysis completed successfully.
"""
            # Dynamic Strengths
            if cat_id == "F" and self.stats["requirements"]:
                 content += "- [Automated] `requirements.txt` is present.\n"
            if cat_id == "O" and self.stats["cicd"]:
                 content += "- [Automated] CI/CD workflows detected.\n"
            if cat_id == "G" and self.stats["test_files"] > 10:
                 content += f"- [Automated] Found {self.stats['test_files']} test files.\n"

            content += """
### Weaknesses
- [Automated] See detailed metrics below.

## Detailed Metrics
"""
            # Add specific metrics based on category
            if cat_id == "A":
                content += f"- **God Functions**: {len(self.stats['god_functions'])} identified (See Pragmatic Review).\n"
                content += f"- **Total Files**: {self.stats['files']}\n"
                content += f"- **Total Lines**: {self.stats['lines']}\n"
            elif cat_id == "B":
                content += f"- **Type Hint Coverage**: {self.stats['type_hints'] / max(1, self.stats['functions']) * 100:.1f}%\n"
                content += f"- **Print Statements**: {self.stats['prints']} (Should use logging)\n"
            elif cat_id == "C":
                content += f"- **Docstring Coverage**: {self.stats['docstrings'] / max(1, self.stats['functions']) * 100:.1f}%\n"
                content += f"- **Incomplete Docs**: {self.stats['incomplete_docs']} files flagged.\n"
            elif cat_id == "D":
                content += f"- **UI Files Detected**: {self.stats['ui_files']}\n"
                content += "- **Note**: Manual review of UI/UX flow is required.\n"
            elif cat_id == "E":
                content += f"- **Performance Tool Imports**: {self.stats['performance_imports']} (e.g. cProfile, timeit)\n"
            elif cat_id == "F":
                content += f"- **Requirements File**: {'Yes' if self.stats['requirements'] else 'No'}\n"
                content += f"- **Setup.py**: {'Yes' if self.stats['has_setup_py'] else 'No'}\n"
                content += f"- **pyproject.toml**: {'Yes' if self.stats['has_pyproject'] else 'No'}\n"
            elif cat_id == "G":
                content += f"- **Test Files**: {self.stats['test_files']}\n"
                content += f"- **Python Files**: {self.stats['py_files']}\n"
                content += f"- **Test Ratio**: {self.stats['test_files'] / max(1, self.stats['py_files']):.2f}\n"
            elif cat_id == "H":
                content += f"- **Try/Except Blocks**: {self.stats['try_except']}\n"
                content += f"- **Exception Handling Ratio**: {self.stats['try_except'] / max(1, self.stats['functions']):.2f} per function\n"
            elif cat_id == "I":
                content += f"- **Eval Usages**: {self.stats['evals']} (Security Risk)\n"
            elif cat_id == "J":
                content += "- **Automated Metrics**: None available.\n"
                content += "- **Recommendation**: Manually review plugin architecture documentation.\n"
            elif cat_id == "K":
                content += f"- **Dependency Management**: {'Requirements.txt present' if self.stats['requirements'] else 'Missing requirements.txt'}\n"
            elif cat_id == "L":
                content += f"- **DRY Violations**: {len(self.stats['dry_violations'])} major blocks found.\n"
                content += f"- **TODO Markers**: {self.stats['todos']}\n"
                content += f"- **FIXME Markers**: {self.stats['fixmes']}\n"
            elif cat_id == "M":
                content += f"- **Docs Folder**: {'Yes' if self.stats['has_docs_folder'] else 'No'}\n"
                content += f"- **Examples Folder**: {'Yes' if self.stats['has_examples'] else 'No'}\n"
            elif cat_id == "N":
                content += f"- **Visualization Libraries**: {self.stats['visualization_imports']} imports found.\n"
            elif cat_id == "O":
                content += f"- **CI/CD Workflows Detected**: {'Yes' if self.stats['cicd'] else 'No'}\n"

            content += """
## Recommendations
1. Address the weaknesses identified above.
2. Review the Pragmatic Programmer report for specific code locations.
3. Update documentation to reflect current state.
"""
            if raw_score is None:
                content += "4. Perform a manual review to determine the score for this category.\n"

            filepath.write_text(content)

    def generate_completist_report(self):
        logger.info("Generating completist report...")
        date_str = datetime.now().strftime("%Y-%m-%d")
        filepath = DOCS_DIR / "completist" / f"Completist_Report_{date_str}.md"

        content = f"""# Completist Audit Report - {date_str}

## Overview
This report summarizes the technical debt and incompleteness markers found in the codebase.

## Metrics
| Metric | Count | Impact |
|--------|-------|--------|
| **TODOs** | {self.stats['todos']} | Feature gaps |
| **FIXMEs** | {self.stats['fixmes']} | Broken/Buggy code |
| **NotImplemented** | {self.stats['not_implemented']} | Missing implementations |
| **Abstract Methods** | {self.stats['abstract_methods']} | Unimplemented interfaces |
| **Incomplete Docs** | {self.stats['incomplete_docs']} | Documentation gaps |

## Critical Gaps
- **High TODO count**: Indicates significant planned work that is not yet started.
- **Abstract Methods**: {self.stats['abstract_methods']} abstract methods need concrete implementations.

## Recommendations
- Prioritize `FIXME` items as they likely represent bugs.
- Review `TODO` items and convert them to proper issues or delete them if obsolete.
- Ensure all abstract methods are implemented in derived classes.
"""
        filepath.write_text(content)

    def generate_comprehensive_report(self):
        logger.info("Generating comprehensive report...")
        filepath = DOCS_DIR / "Comprehensive_Assessment.md"

        # Calculate Unified Score
        valid_scores = [s for s in self.scores.values() if s is not None]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0

        content = f"""# Comprehensive Assessment Report

**Date**: {datetime.now().strftime("%Y-%m-%d")}
**Overall Score**: {avg_score:.1f}/10 (based on {len(valid_scores)} automated categories)
**Status**: {"PASS" if avg_score > 7 else "NEEDS IMPROVEMENT"}

## Executive Summary
The repository has been assessed across 15 categories (A-O), along with a Completist Audit and Pragmatic Programmer Review. The overall health is rated at **{avg_score:.1f}/10**.
Note: Some categories require manual review and are not included in the automated score.

## Unified Scorecard

| Category | Name | Weight | Score |
|----------|------|--------|-------|
"""
        for cat_id, info in CATEGORIES.items():
            s = self.scores.get(cat_id)
            s_str = f"{s}/10" if s is not None else "Manual Review"
            content += f"| **{cat_id}** | {info['name']} | {info['weight']}x | {s_str} |\n"

        content += f"""
## Top 10 Recommendations
1. **Reduce Technical Debt**: Address the {self.stats['todos']} TODOs and {self.stats['fixmes']} FIXMEs.
2. **Improve Documentation**: Increase docstring coverage (currently {self.stats['docstrings'] / max(1, self.stats['functions']) * 100:.1f}%).
3. **Enhance Testing**: Add more test files (current ratio: {self.stats['test_files'] / max(1, self.stats['py_files']):.2f}).
4. **Fix DRY Violations**: Refactor the {len(self.stats['dry_violations'])} duplicate code blocks identified in the Pragmatic Review.
5. **Refactor God Functions**: Break down the {len(self.stats['god_functions'])} complex functions identified.
6. **Security Hardening**: Audit and remove the {self.stats['evals']} usages of `eval()`.
7. **Type Safety**: Add type hints to functions (currently {self.stats['type_hints'] / max(1, self.stats['functions']) * 100:.1f}% coverage).
8. **Logging**: Replace {self.stats['prints']} print statements with proper logging.
9. **CI/CD**: Maintain and expand the existing CI/CD workflows.
10. **Error Handling**: Improve exception handling coverage.

## Conclusion
The codebase shows promise but requires significant refactoring to improve maintainability and reduce technical debt.
"""
        filepath.write_text(content)

def main():
    generator = AssessmentGenerator()
    generator.analyze_codebase()
    generator.load_completist_data()
    generator.load_pragmatic_data()
    generator.calculate_scores()

    generator.generate_category_reports()
    generator.generate_completist_report()
    generator.generate_comprehensive_report()

    logger.info("Assessment generation complete.")

if __name__ == "__main__":
    main()
