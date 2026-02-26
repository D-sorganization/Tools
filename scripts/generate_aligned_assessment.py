#!/usr/bin/env python3
"""
Generate aligned assessments (A-O) and comprehensive report.
Strictly follows the categories in docs/assessments/README.md.
Enhances the assessment with specific checks and detailed justifications.
"""

import ast
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import TypedDict

# Configuration
REPO_ROOT = Path(__file__).parent.parent.resolve()
DOCS_DIR = REPO_ROOT / "docs" / "assessments"
COMPLETIST_REPORT = (
    DOCS_DIR
    / "completist"
    / f"Completist_Report_{datetime.now().strftime('%Y-%m-%d')}.md"
)
PRAGMATIC_JSON = (
    DOCS_DIR
    / "pragmatic_programmer"
    / f"review_{datetime.now().strftime('%Y-%m-%d')}.json"
)
PRAGMATIC_MD = (
    DOCS_DIR
    / "pragmatic_programmer"
    / f"review_{datetime.now().strftime('%Y-%m-%d')}.md"
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CATEGORIES = {
    "A": "Architecture & Implementation",
    "B": "Code Quality & Hygiene",
    "C": "Documentation & Comments",
    "D": "User Experience & Developer Journey",
    "E": "Performance & Scalability",
    "F": "Installation & Deployment",
    "G": "Testing & Validation",
    "H": "Error Handling & Debugging",
    "I": "Security & Input Validation",
    "J": "Extensibility & Plugin Architecture",
    "K": "Reproducibility & Provenance",
    "L": "Long-Term Maintainability",
    "M": "Educational Resources & Tutorials",
    "N": "Visualization & Export",
    "O": "CI/CD & DevOps",
}


class RepoStats(TypedDict):
    files: int
    lines: int
    py_files: int
    test_files: int
    docstrings: int
    functions: int
    classes: int
    prints: int
    loggings: int
    evals: int
    try_except: int
    imports: set[str]
    max_depth: int
    dirs: int
    readme_exists: bool
    contributing_exists: bool
    install_exists: bool
    requirements_exists: bool
    setup_py_exists: bool
    dockerfile_exists: bool
    cicd_exists: bool
    env_files: int
    examples_dir: bool
    docs_dir: bool
    tests_dir: bool
    pyproject_exists: bool
    pytest_ini_exists: bool
    todo_count: int
    fixme_count: int
    critical_gaps: int
    dry_violations: int
    god_functions: int
    god_functions_list: list[str]
    secrets_found: int


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
        "prints": 0,
        "loggings": 0,
        "evals": 0,
        "try_except": 0,
        "imports": set(),
        "max_depth": 0,
        "dirs": 0,
        "readme_exists": False,
        "contributing_exists": False,
        "install_exists": False,
        "requirements_exists": False,
        "setup_py_exists": False,
        "dockerfile_exists": False,
        "cicd_exists": False,
        "env_files": 0,
        "examples_dir": False,
        "docs_dir": False,
        "tests_dir": False,
        "pyproject_exists": False,
        "pytest_ini_exists": False,
        "todo_count": 0,
        "fixme_count": 0,
        "critical_gaps": 0,
        "dry_violations": 0,
        "god_functions": 0,
        "god_functions_list": [],
        "secrets_found": 0,
    }

    # Helper to check for file existence
    stats["readme_exists"] = (REPO_ROOT / "README.md").exists()
    stats["contributing_exists"] = (REPO_ROOT / "CONTRIBUTING.md").exists() or (
        REPO_ROOT / "docs" / "CONTRIBUTING.md"
    ).exists()
    stats["install_exists"] = (REPO_ROOT / "INSTALL.md").exists()
    stats["requirements_exists"] = (REPO_ROOT / "requirements.txt").exists()
    stats["setup_py_exists"] = (REPO_ROOT / "setup.py").exists()
    stats["pyproject_exists"] = (REPO_ROOT / "pyproject.toml").exists()
    stats["pytest_ini_exists"] = (REPO_ROOT / "pytest.ini").exists()
    stats["examples_dir"] = (REPO_ROOT / "examples").exists()
    stats["docs_dir"] = (REPO_ROOT / "docs").exists()
    stats["tests_dir"] = (REPO_ROOT / "tests").exists() or (
        REPO_ROOT / "src" / "tests"
    ).exists()  # Heuristic

    secret_patterns = [
        re.compile(r"(?i)(api_key|secret_key|password|token)\s*=\s*['\"][^'\"]+['\"]"),
    ]

    for root, _dirs, files in os.walk(REPO_ROOT):
        # Exclusions
        path_parts = Path(root).parts
        if (
            any(p.startswith(".") and p != ".github" for p in path_parts)
            or "venv" in path_parts
            or "node_modules" in path_parts
            or "__pycache__" in path_parts
        ):
            continue

        current_depth = len(Path(root).relative_to(REPO_ROOT).parts)
        stats["max_depth"] = max(stats["max_depth"], current_depth)
        stats["dirs"] += 1

        if ".github" in str(Path(root)) and "workflows" in str(Path(root)):
            if any(f.endswith((".yml", ".yaml")) for f in files):
                stats["cicd_exists"] = True

        for file in files:
            stats["files"] += 1
            filepath = Path(root) / file

            if file == "Dockerfile":
                stats["dockerfile_exists"] = True
            if file.startswith(".env"):
                stats["env_files"] += 1

            try:
                # Basic line count and content check
                try:
                    content = filepath.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue

                stats["lines"] += len(content.splitlines())

                # Secret scanning
                for pattern in secret_patterns:
                    if pattern.search(content):
                        stats["secrets_found"] += 1

                if file.endswith(".py"):
                    stats["py_files"] += 1
                    if "test_" in file or "_test.py" in file:
                        stats["test_files"] += 1

                    try:
                        tree = ast.parse(content)
                        for node in ast.walk(tree):
                            if isinstance(
                                node, (ast.FunctionDef, ast.AsyncFunctionDef)
                            ):
                                stats["functions"] += 1
                                if ast.get_docstring(node):
                                    stats["docstrings"] += 1
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
                                    # Heuristic for logger.info etc
                                    elif node.func.attr in [
                                        "info",
                                        "debug",
                                        "warning",
                                        "error",
                                        "critical",
                                    ]:
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
                        pass

            except Exception:
                pass

    return stats


def parse_external_reports(stats: RepoStats) -> None:
    # Pragmatic Programmer Review
    if PRAGMATIC_JSON.exists():
        try:
            data = json.loads(PRAGMATIC_JSON.read_text())
            for issue in data.get("issues", []):
                if issue.get("principle") == "DRY":
                    # Count instances, description often says "Found in X locations"
                    desc = issue.get("description", "")
                    match = re.search(r"Found in (\d+) locations", desc)
                    if match:
                        stats["dry_violations"] += int(match.group(1))
                    else:
                        stats["dry_violations"] += 1
                if issue.get(
                    "principle"
                ) == "ORTHOGONALITY" and "God function" in issue.get("title", ""):
                    stats["god_functions"] += 1
                    stats["god_functions_list"].append(issue.get("title", ""))
        except Exception as e:
            logger.error(f"Failed to parse pragmatic report JSON: {e}")
    elif PRAGMATIC_MD.exists():
        # Fallback to MD parsing if JSON is missing
        try:
            content = PRAGMATIC_MD.read_text()
            # Simple grep for god functions
            god_funcs = re.findall(r"God function: ([\w_]+)", content)
            stats["god_functions"] = len(god_funcs)
            stats["god_functions_list"] = god_funcs
            # Grep for DRY
            dry_matches = re.findall(r"\*\*DRY\*\*", content)
            stats["dry_violations"] = len(dry_matches) * 2  # Estimate
        except Exception as e:
            logger.error(f"Failed to parse pragmatic report MD: {e}")
    else:
        logger.warning(f"Pragmatic report not found: {PRAGMATIC_JSON}")

    # Completist Report
    if COMPLETIST_REPORT.exists():
        try:
            content = COMPLETIST_REPORT.read_text()
            # Extract Critical Gaps
            match_crit = re.search(r"- \*\*Critical Gaps\*\*: (\d+)", content)
            if match_crit:
                stats["critical_gaps"] = int(match_crit.group(1))

            # Extract TODOs
            match_todo = re.search(r"- \*\*Feature Gaps \(TODO\)\*\*: (\d+)", content)
            if match_todo:
                stats["todo_count"] = int(match_todo.group(1))

            # Extract FIXMEs
            match_fixme = re.search(r"- \*\*Technical Debt\*\*: (\d+)", content)
            if match_fixme:
                stats["fixme_count"] = int(match_fixme.group(1))
        except Exception as e:
            logger.error(f"Failed to parse completist report: {e}")
    else:
        logger.warning(f"Completist report not found: {COMPLETIST_REPORT}")


def calculate_grades(stats: RepoStats) -> dict[str, tuple[float, str]]:
    grades = {}

    # A: Architecture & Implementation
    score_a = 9.0
    deductions_a = []
    if stats["max_depth"] > 8:
        score_a -= 1
        deductions_a.append(f"Deep directory structure ({stats['max_depth']})")
    if stats["god_functions"] > 0:
        pen = min(3, stats["god_functions"] * 0.1)
        score_a -= pen
        deductions_a.append(
            f"{stats['god_functions']} God Functions detected (e.g., {', '.join(stats['god_functions_list'][:3])}...)"
        )
    if stats["dry_violations"] > 20:
        score_a -= 1
        deductions_a.append(f"High DRY violations ({stats['dry_violations']})")

    just_a = (
        "Architecture seems sound." if not deductions_a else "; ".join(deductions_a)
    )
    grades["A"] = (max(0, round(score_a, 1)), just_a)

    # B: Code Quality & Hygiene
    score_b = 8.0  # Baseline
    if stats["pyproject_exists"]:
        score_b += 1
    # Type hints heuristic (future improvement: count annotated args)
    just_b = f"Standard files present: {stats['pyproject_exists']}"
    grades["B"] = (score_b, just_b)

    # C: Documentation & Comments
    score_c = 0.0
    total_defs = max(1, stats["functions"] + stats["classes"])
    doc_cov = stats["docstrings"] / total_defs
    score_c += doc_cov * 8  # Max 8 from docstrings
    if stats["readme_exists"]:
        score_c += 1
    if stats["contributing_exists"]:
        score_c += 1
    grades["C"] = (
        round(score_c, 1),
        f"Docstring coverage: {doc_cov*100:.1f}%, README: {stats['readme_exists']}",
    )

    # D: User Experience & Developer Journey
    score_d = 5.0  # Baseline
    if stats["examples_dir"]:
        score_d += 2
    if stats["install_exists"]:
        score_d += 2
    if stats["readme_exists"]:
        score_d += 1
    grades["D"] = (
        min(10, score_d),
        f"Examples: {stats['examples_dir']}, Install Doc: {stats['install_exists']}",
    )

    # E: Performance & Scalability
    score_e = 9.0
    if stats["prints"] > 100:
        score_e -= 2
    if stats["prints"] > 500:
        score_e -= 2
    grades["E"] = (
        max(0, score_e),
        f"Print statements: {stats['prints']} (Should use logging)",
    )

    # F: Installation & Deployment
    score_f = 6.0
    if stats["requirements_exists"]:
        score_f += 2
    if stats["setup_py_exists"]:
        score_f += 1
    if stats["dockerfile_exists"]:
        score_f += 1
    grades["F"] = (
        min(10, score_f),
        f"Reqs: {stats['requirements_exists']}, Docker: {stats['dockerfile_exists']}",
    )

    # G: Testing & Validation
    test_ratio = stats["test_files"] / max(1, stats["py_files"])
    score_g = min(10, test_ratio * 20)  # 0.5 ratio -> 10
    if stats["pytest_ini_exists"]:
        score_g += 0.5
    grades["G"] = (
        min(10, round(score_g, 1)),
        f"Test ratio: {test_ratio:.2f}, pytest.ini: {stats['pytest_ini_exists']}",
    )

    # H: Error Handling & Debugging
    try_ratio = stats["try_except"] / max(1, stats["functions"])
    score_h = min(10, 5 + (try_ratio * 10))
    grades["H"] = (round(score_h, 1), f"Try/Except ratio: {try_ratio:.2f}")

    # I: Security & Input Validation
    score_i = 10.0
    if stats["evals"] > 0:
        score_i -= stats["evals"] * 2
    if stats["env_files"] > 0:
        score_i -= 1  # .env in repo is bad
    if stats["secrets_found"] > 0:
        score_i -= 2
    grades["I"] = (
        max(0, score_i),
        f"Eval usage: {stats['evals']}, .env files: {stats['env_files']}, Possible Secrets: {stats['secrets_found']}",
    )

    # J: Extensibility & Plugin Architecture
    score_j = 7.0  # Baseline
    if stats["classes"] > 50:
        score_j += 1
    grades["J"] = (score_j, f"Classes defined: {stats['classes']}")

    # K: Reproducibility & Provenance
    score_k = 6.0
    if stats["requirements_exists"]:
        score_k += 2
    if "numpy" in stats["imports"] or "pandas" in stats["imports"]:
        score_k += 1  # Scientific stack usually implies checking results
    grades["K"] = (
        min(10, score_k),
        f"Standard deps used: {stats['requirements_exists']}",
    )

    # L: Long-Term Maintainability
    score_l = 10.0
    debt = stats["todo_count"] + stats["fixme_count"]
    if debt > 20:
        score_l -= 1
    if debt > 50:
        score_l -= 2
    if stats["dry_violations"] > 50:
        score_l -= 2
    grades["L"] = (
        max(0, score_l),
        f"Debt (TODO+FIXME): {debt}, DRY Violations: {stats['dry_violations']}",
    )

    # M: Educational Resources & Tutorials
    score_m = 4.0
    if stats["docs_dir"]:
        score_m += 2
    if stats["examples_dir"]:
        score_m += 2
    grades["M"] = (
        score_m,
        f"Docs: {stats['docs_dir']}, Examples: {stats['examples_dir']}",
    )

    # N: Visualization & Export
    score_n = 5.0
    if "matplotlib" in stats["imports"]:
        score_n += 2
    if "seaborn" in stats["imports"]:
        score_n += 1
    if "plotly" in stats["imports"]:
        score_n += 1
    grades["N"] = (
        min(10, score_n),
        f"Plotting libs: {', '.join(i for i in ['matplotlib', 'seaborn', 'plotly'] if i in stats['imports'])}",
    )

    # O: CI/CD & DevOps
    score_o = 10.0 if stats["cicd_exists"] else 0.0
    grades["O"] = (score_o, f"CI/CD Workflows: {stats['cicd_exists']}")

    return grades


def generate_reports(grades: dict[str, tuple[float, str]], stats: RepoStats) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate individual reports
    for cat_code, (score, just) in grades.items():
        cat_name = CATEGORIES[cat_code]
        filename = f"Assessment_{cat_code}_{cat_name.replace(' ', '_').replace('/', '_').replace('&', 'and')}.md"
        filepath = DOCS_DIR / filename

        content = f"""# Assessment: {cat_name} (Category {cat_code})

## Grade: {score}/10

## Justification
{just}

## Statistics
- **Relevant Files**: {stats['files']}
- **Metric Context**:
  - Python Files: {stats['py_files']}
  - Docstrings: {stats['docstrings']}
  - Test Files: {stats['test_files']}
  - Analysis Date: {datetime.now().strftime("%Y-%m-%d")}

## Recommendations
- Refer to `docs/assessments/Comprehensive_Assessment.md` for prioritized actions.
"""
        filepath.write_text(content)

    # Generate Comprehensive Report
    weighted_sum = sum(g[0] for g in grades.values())
    avg_score = weighted_sum / 15.0

    comp_content = f"""# Comprehensive Assessment

## Date: {datetime.now().strftime("%Y-%m-%d")}
## Unified Score: {avg_score:.2f}/10

## Scorecard (Categories A-O)

| ID | Category | Grade | Justification |
|----|----------|-------|---------------|
"""
    for code, name in sorted(CATEGORIES.items()):
        g = grades[code]
        comp_content += f"| {code} | {name} | {g[0]}/10 | {g[1]} |\n"

    comp_content += f"""
## Completist Audit Summary
- **Critical Gaps**: {stats['critical_gaps']}
- **Feature Gaps (TODO)**: {stats['todo_count']}
- **Technical Debt (FIXME)**: {stats['fixme_count']}
- **Full Report**: [Completist Report](completist/Completist_Report_{datetime.now().strftime('%Y-%m-%d')}.md)

## Pragmatic Programmer Review Summary
- **DRY Violations**: {stats['dry_violations']} occurrences
- **God Functions**: {stats['god_functions']} detected
- **Full Report**: `docs/assessments/pragmatic_programmer/review_{datetime.now().strftime('%Y-%m-%d')}.md`

## Top 10 Unified Recommendations

1. **Address Critical Gaps**: Resolve the {stats['critical_gaps']} critical implementation gaps identified in the Completist Audit.
2. **Refactor God Functions**: Split the {stats['god_functions']} complex functions identified (see Pragmatic Review).
3. **Reduce Technical Debt**: Address {stats['fixme_count']} FIXMEs and {stats['todo_count']} TODOs (Category L).
4. **Improve Documentation**: Increase docstring coverage (currently {stats['docstrings']/max(1, stats['functions']+stats['classes'])*100:.1f}%).
5. **Enforce DRY**: Refactor code to reduce {stats['dry_violations']} duplication instances.
6. **Enhance Security**: Audit and remove {stats['evals']} `eval()` calls and {stats['secrets_found']} potential secrets (Category I).
7. **Modernize Logging**: Replace {stats['prints']} `print()` calls with proper `logging` (Category E).
8. **Boost Test Coverage**: Increase test file ratio (Category G, currently {grades['G'][0]}/10).
9. **Standardize Error Handling**: Ensure consistent try/except usage (Category H).
10. **Expand Examples**: Add more usage examples to improve Developer Journey (Category D).

"""
    (DOCS_DIR / "Comprehensive_Assessment.md").write_text(comp_content)
    logger.info(
        f"Generated Comprehensive Assessment: {DOCS_DIR / 'Comprehensive_Assessment.md'}"
    )


def main() -> None:
    stats = analyze_codebase()
    parse_external_reports(stats)
    grades = calculate_grades(stats)
    generate_reports(grades, stats)


if __name__ == "__main__":
    main()
