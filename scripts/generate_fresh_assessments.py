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
    evals: int
    type_hints: int
    args_annotated: int
    try_except: int
    imports: set[str]
    requirements: bool
    cicd: bool
    readme: bool


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
                                    elif node.func.id == "eval":
                                        stats["evals"] += 1
                            elif isinstance(node, ast.Import):
                                for alias in node.names:
                                    stats["imports"].add(alias.name)
                            elif isinstance(node, ast.ImportFrom):
                                if node.module:
                                    stats["imports"].add(node.module)
                            elif isinstance(node, ast.Try):
                                stats["try_except"] += 1
                    except SyntaxError:
                        logger.warning(f"Syntax error in {filepath}")
            except Exception as e:
                logger.error(f"Error analyzing {filepath}: {e}")

    return stats


def generate_report(stats: RepoStats) -> str:
    """Generate markdown report from stats."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# Codebase Assessment - {timestamp}

## Statistics
- Total Files: {stats["files"]}
- Total Lines: {stats["lines"]}
- Python Files: {stats["py_files"]}
- Test Files: {stats["test_files"]}
- Functions: {stats["functions"]}
- Classes: {stats["classes"]}

## Quality Metrics
- Docstring Coverage: {stats["docstrings"] / stats["functions"] * 100:.1f}% (functions)
- Type Hint Coverage: {stats["type_hints"] / stats["functions"] * 100:.1f}% (returns)
- Argument Annotation: {stats["args_annotated"] / max(1, stats["functions"]) * 100:.1f}% (per function avg)

## Health Indicators
- TODOs: {stats["todos"]}
- FIXMEs: {stats["fixmes"]}
- Print Statements: {stats["prints"]}
- Eval Usage: {stats["evals"]}
- Exception Handling (try block count): {stats["try_except"]}

## Infrastructure
- Requirements.txt: {"[X]" if stats["requirements"] else "[ ]"}
- CI/CD Workflows: {"[X]" if stats["cicd"] else "[ ]"}
- README.md: {"[X]" if stats["readme"] else "[ ]"}

## Key Dependencies
{", ".join(sorted(list(stats["imports"]))[:20])}...
"""
    return report


def main() -> None:
    """Main execution."""
    stats = analyze_codebase()
    report = generate_report(stats)

    report_path = DOCS_DIR / "current_assessment.md"
    report_path.write_text(report)
    logger.info(f"Report generated: {report_path}")

    # Also generate individual issues for high priority items
    if stats["prints"] > 50:
        create_issue(
            "I001",
            "High Print Usage",
            f"Codebase contains {stats['prints']} print statements. Replace with proper logging.",
        )

    if stats["docstrings"] / max(1, stats["functions"]) < 0.5:
        create_issue(
            "B001",
            "Low Docstring Coverage",
            f"Docstring coverage is at {stats['docstrings'] / stats['functions'] * 100:.1f}%.",
        )


def create_issue(issue_id: str, title: str, description: str) -> None:
    """Create a new issue file."""
    issue_path = ISSUES_DIR / f"{issue_id}_{title.lower().replace(' ', '_')}.md"
    content = f"""# Issue {issue_id}: {title}
Date: {datetime.now().strftime("%Y-%m-%d")}
Status: Open
Category: {CATEGORIES.get(issue_id[0], "General")}

## Description
{description}

## Recommendations
- Systematically address this issue in the next refactoring cycle.
"""
    issue_path.write_text(content)
    logger.info(f"Issue created: {issue_path}")


if __name__ == "__main__":
    main()
