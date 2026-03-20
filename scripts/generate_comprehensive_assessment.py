#!/usr/bin/env python3
"""
Generate comprehensive assessments for the repository.
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
    loggings: int
    evals: int
    type_hints: int
    args_annotated: int
    try_except: int
    imports: set[str]
    requirements: bool
    cicd: bool
    readme: bool
    env_files: int
    config_files: int
    dirs: int
    max_depth: int


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
        "loggings": 0,
        "evals": 0,
        "type_hints": 0,
        "args_annotated": 0,
        "try_except": 0,
        "imports": set(),
        "requirements": False,
        "cicd": False,
        "readme": False,
        "env_files": 0,
        "config_files": 0,
        "dirs": 0,
        "max_depth": 0,
    }

    for root, dirs, files in os.walk(REPO_ROOT):
        # Skip hidden and venv directories
        path_parts = Path(root).parts
        if (
            any(p.startswith(".") and p != ".github" for p in path_parts)
            or "venv" in path_parts
            or "__pycache__" in path_parts
            or "node_modules" in path_parts
            or "site-packages" in path_parts
        ):
            continue

        current_depth = len(Path(root).relative_to(REPO_ROOT).parts)
        stats["max_depth"] = max(stats["max_depth"], current_depth)
        stats["dirs"] += len(dirs)

        for file in files:
            stats["files"] += 1
            filepath = Path(root) / file

            if file == "requirements.txt":
                stats["requirements"] = True
            if file.lower() == "readme.md":
                stats["readme"] = True
            if (
                ".github" in str(filepath)
                and "workflows" in str(filepath)
                and file.endswith((".yml", ".yaml"))
            ):
                stats["cicd"] = True
            if file.startswith(".env"):
                stats["env_files"] += 1
            if file.endswith((".json", ".yaml", ".toml", ".ini")):
                stats["config_files"] += 1

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
                            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
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
                                    if (
                                        isinstance(node.func.value, ast.Name)
                                        and node.func.value.id == "logging"
                                    ):
                                        stats["loggings"] += 1
                                    elif node.func.attr in [
                                        "info",
                                        "debug",
                                        "warning",
                                        "error",
                                        "critical",
                                    ]:
                                        # Heuristic for logging calls on logger objects
                                        stats["loggings"] += 1
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
    grades = {}

    # A: Code Structure
    # Metric: Reasonable depth, distinct directories
    score_a = 10.0
    if stats["max_depth"] > 8:
        score_a -= 2
    if stats["dirs"] < 5:
        score_a -= 2
    grades["A"] = (
        max(0, score_a),
        f"Directory structure depth: {stats['max_depth']}, Directories: {stats['dirs']}",
    )

    # B: Documentation
    # Metric: Docstring coverage
    total_defs = stats["functions"] + stats["classes"]
    doc_coverage = stats["docstrings"] / max(1, total_defs)
    score_b = doc_coverage * 10
    if stats["readme"]:
        score_b = min(10, score_b + 2)
    grades["B"] = (
        round(score_b, 1),
        f"Docstring coverage: {doc_coverage * 100:.1f}%, README present: {stats['readme']}",
    )

    # C: Test Coverage
    # Metric: Test file ratio (heuristic)
    test_ratio = stats["test_files"] / max(1, stats["py_files"])
    score_c = min(10, test_ratio * 20)  # 50% ratio = 10
    grades["C"] = (
        round(score_c, 1),
        f"Test file ratio: {test_ratio * 100:.1f}% ({stats['test_files']}/{stats['py_files']})",
    )

    # D: Error Handling
    # Metric: Try/Except usage vs Functions
    try_ratio = stats["try_except"] / max(1, stats["functions"])
    score_d = min(10, try_ratio * 50)  # 20% functions with try = 10 (heuristic)
    score_d = max(5, score_d)  # Baseline 5 as standard python raises exceptions
    grades["D"] = (
        round(score_d, 1),
        f"Try/Except blocks: {stats['try_except']} (Ratio: {try_ratio:.2f})",
    )

    # E: Performance
    # Metric: Print usage (bad)
    score_e = 8.0
    if stats["prints"] > 100:
        score_e -= 2
    if stats["prints"] > 500:
        score_e -= 2
    grades["E"] = (max(0, score_e), f"Print statements: {stats['prints']}")

    # F: Security
    # Metric: Eval usage (bad)
    score_f = 10.0
    if stats["evals"] > 0:
        score_f -= stats["evals"] * 2
    grades["F"] = (max(0, score_f), f"Eval usage: {stats['evals']}")

    # G: Dependencies
    # Metric: Requirements.txt
    score_g = 10.0 if stats["requirements"] else 0.0
    grades["G"] = (score_g, f"Requirements.txt present: {stats['requirements']}")

    # H: CI/CD
    # Metric: Workflow files
    score_h = 10.0 if stats["cicd"] else 0.0
    grades["H"] = (score_h, f"CI/CD Workflows present: {stats['cicd']}")

    # I: Code Style
    # Metric: Type hints
    type_coverage = stats["type_hints"] / max(1, stats["functions"])
    score_i = type_coverage * 10
    grades["I"] = (round(score_i, 1), f"Type hint coverage: {type_coverage * 100:.1f}%")

    # J: API Design
    # Metric: Classes present
    score_j = 7.0  # Baseline
    if stats["classes"] > 10:
        score_j += 1
    grades["J"] = (score_j, f"Classes defined: {stats['classes']}")

    # K: Data Handling
    # Metric: Config files
    score_k = 6.0  # Baseline
    if "pandas" in stats["imports"]:
        score_k += 2
    if "numpy" in stats["imports"]:
        score_k += 1
    grades["K"] = (
        min(10, score_k),
        f"Data libs used: {'pandas' in stats['imports']}, {'numpy' in stats['imports']}",
    )

    # L: Logging
    # Metric: Logging usage
    score_l = 0.0
    if stats["loggings"] > 0:
        ratio = stats["loggings"] / max(1, stats["prints"] + stats["loggings"])
        score_l = ratio * 10
    grades["L"] = (
        round(score_l, 1),
        f"Logging usage: {stats['loggings']} vs Prints: {stats['prints']}",
    )

    # M: Configuration
    # Metric: Config/Env files
    score_m = 5.0
    if stats["env_files"] > 0:
        score_m += 2.5
    if stats["config_files"] > 0:
        score_m += 2.5
    grades["M"] = (
        min(10, score_m),
        f"Env files: {stats['env_files']}, Config files: {stats['config_files']}",
    )

    # N: Scalability
    # Metric: Code size
    score_n = 7.0
    if stats["py_files"] > 50:
        score_n += 1
    grades["N"] = (score_n, f"Python files: {stats['py_files']}")

    # O: Maintainability
    # Metric: TODOs/FIXMEs (bad)
    score_o = 10.0
    debt = stats["todos"] + stats["fixmes"]
    if debt > 50:
        score_o -= 2
    if debt > 200:
        score_o -= 3
    if debt > 500:
        score_o -= 3
    grades["O"] = (max(0, score_o), f"Technical Debt (TODO+FIXME): {debt}")

    return grades


def generate_assessments(
    grades: dict[str, tuple[float, str]], stats: RepoStats
) -> None:
    for category, (score, justification) in grades.items():
        name = CATEGORIES[category]
        filename = (
            f"Assessment_{category}_{name.replace(' ', '_').replace('/', '-')}.md"
        )
        filepath = DOCS_DIR / filename

        content = f"""# Assessment: {name} (Category {category})

## Grade: {score}/10

## Justification
{justification}

## Statistics
- Total Python Files: {stats["py_files"]}
- Total Lines of Code: {stats["lines"]}
- Analysis Date: {datetime.now().strftime("%Y-%m-%d")}
"""
        filepath.write_text(content)
        # logger.info(f"Generated {filepath}")


def generate_issues(grades: dict[str, tuple[float, str]]) -> None:
    for category, (score, justification) in grades.items():
        if score < 5.0:
            name = CATEGORIES[category]
            filename = f"Issue_{category}_{name.replace(' ', '_').replace('/', '-')}.md"
            filepath = ISSUES_DIR / filename

            content = f"""---
labels: jules:assessment, needs-attention
---

# Issue: Low Score in {name}

## Grade: {score}/10

## Problem
The assessment for **{name}** returned a score below the acceptable threshold of 5.0.

## Justification
{justification}

## Action Items
1. Review the generated assessment in `docs/assessments/Assessment_{category}_{name.replace(" ", "_")}.md`.
2. Address the specific metrics highlighted in the justification.
3. Run `scripts/generate_comprehensive_assessment.py` to verify improvements.
"""
            filepath.write_text(content)
            logger.info(f"Generated Issue: {filepath}")


def generate_comprehensive(grades: dict[str, tuple[float, str]]) -> None:
    weighted_score = (
        (grades["A"][0] + grades["I"][0]) / 2 * 0.25
        + grades["C"][0] * 0.15
        + grades["B"][0] * 0.10
        + (grades["F"][0] + grades["D"][0]) / 2 * 0.15
        + grades["E"][0] * 0.15
        + (grades["H"][0] + grades["M"][0] + grades["G"][0]) / 3 * 0.10
        + (
            grades["J"][0]
            + grades["K"][0]
            + grades["L"][0]
            + grades["N"][0]
            + grades["O"][0]
        )
        / 5
        * 0.10
    )

    content = f"""# Comprehensive Assessment

## Date: {datetime.now().strftime("%Y-%m-%d")}

## Weighted Score: {weighted_score:.2f}/10

## Grade Table

| Category | Name | Grade |
|----------|------|-------|
"""
    for code, name in sorted(CATEGORIES.items()):
        score = grades[code][0]
        content += f"| {code} | {name} | {score}/10 |\n"

    content += """
## Top 5 Recommendations

1. **Address Critical Issues**: Review any categories with scores below 5.0 (check `docs/assessments/issues/`).
2. **Improve Test Coverage**: Current ratio is reflected in Category C.
3. **Reduce Technical Debt**: Address TODOs and FIXMEs (Category O).
4. **Enhance Documentation**: Improve docstring coverage (Category B).
5. **Standardize Logging**: Replace prints with logging (Category L).

## Methodology
This assessment was automatically generated by `scripts/generate_comprehensive_assessment.py` based on static analysis metrics.
"""

    filepath = DOCS_DIR / "Comprehensive_Assessment.md"
    filepath.write_text(content)
    logger.info(f"Generated Comprehensive Assessment: {filepath}")


def main() -> None:
    stats = analyze_codebase()
    grades = calculate_grades(stats)

    generate_assessments(grades, stats)
    generate_issues(grades)
    generate_comprehensive(grades)

    logger.info("Assessment generation complete.")


if __name__ == "__main__":
    main()
