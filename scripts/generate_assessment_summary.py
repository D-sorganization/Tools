#!/usr/bin/env python3
"""
Generate comprehensive assessment summary from individual assessment reports.

This script aggregates all A-O assessment results and creates:
1. A comprehensive markdown summary
2. A JSON file with structured metrics
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from utils.file_utils import safe_write_json
except ImportError:
    import json

try:
    from utils.file_utils import safe_read_text, safe_write_text
except ImportError:
    from pathlib import Path

    def safe_read_text(path: str | Path, encoding: str = "utf-8", default: str = "") -> str:
        try:
            return Path(path).read_text(encoding=encoding)
        except Exception:
            return default

    def safe_write_text(path: str | Path, content: str, encoding: str = "utf-8", create_parents: bool = True) -> None:
        p = Path(path)
        if create_parents:
            p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding=encoding)

    def safe_write_json(path: str | Path, data: Any, indent: int = 2, create_parents: bool = True) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def extract_score_from_report(report_path: Path) -> float:
    """Extract numerical score from assessment report."""
    try:
        content = safe_read_text(report_path, default="")

        # Look for score patterns like "Overall: 8.5" or "Score: 8.5/10"
        patterns = [
            r"Overall.*?(\d+\.?\d*)",
            r"Score.*?(\d+\.?\d*)",
            r"\*\*(\d+\.?\d*)\*\*.*?/10",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return float(match.group(1))

        # Default score if not found
        return 7.0

    except (OSError, UnicodeDecodeError, ValueError) as e:
        logger.warning(f"Could not extract score from {report_path}: {e}")
        return 7.0


def extract_issues_from_report(report_path: Path) -> list[dict[str, Any]]:
    """Extract issues/findings from assessment report."""
    issues = []

    try:
        content = safe_read_text(report_path, default="")

        # Look for severity markers
        severity_patterns = {
            "BLOCKER": r"BLOCKER:?\s*(.+)",
            "CRITICAL": r"CRITICAL:?\s*(.+)",
            "MAJOR": r"MAJOR:?\s*(.+)",
            "MINOR": r"MINOR:?\s*(.+)",
        }

        for severity, pattern in severity_patterns.items():
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                issues.append(
                    {
                        "severity": severity,
                        "description": match.group(1).strip(),
                        "source": report_path.stem,
                    }
                )

    except (OSError, UnicodeDecodeError) as e:
        logger.warning(f"Could not extract issues from {report_path}: {e}")

    return issues


# Category mapping for assessment reports
ASSESSMENT_CATEGORIES: dict[str, dict[str, Any]] = {
    "A": {"name": "Architecture & Implementation", "weight": 2.0},
    "B": {"name": "Hygiene, Security & Quality", "weight": 2.0},
    "C": {"name": "Documentation & Integration", "weight": 1.5},
    "D": {"name": "User Experience", "weight": 1.5},
    "E": {"name": "Performance & Scalability", "weight": 1.5},
    "F": {"name": "Installation & Deployment", "weight": 1.0},
    "G": {"name": "Testing & Validation", "weight": 2.0},
    "H": {"name": "Error Handling", "weight": 1.0},
    "I": {"name": "Security & Input Validation", "weight": 2.0},
    "J": {"name": "Extensibility & Plugins", "weight": 1.0},
    "K": {"name": "Reproducibility & Provenance", "weight": 1.0},
    "L": {"name": "Long-Term Maintainability", "weight": 1.5},
    "M": {"name": "Educational Resources", "weight": 1.0},
    "N": {"name": "Visualization & Export", "weight": 1.0},
    "O": {"name": "CI/CD & DevOps", "weight": 2.0},
}


def _collect_scores_and_issues(
    input_reports: list[Path],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Collect scores and issues from assessment reports.

    Args:
        input_reports: List of assessment report files

    Returns:
        Tuple of (scores dict, all_issues list)
    """
    scores: dict[str, float] = {}
    all_issues: list[dict[str, Any]] = []

    for report in input_reports:
        match = re.search(r"Assessment_([A-O])_Results", report.name)
        if match:
            assessment_id = match.group(1)
            scores[assessment_id] = extract_score_from_report(report)
            all_issues.extend(extract_issues_from_report(report))

    return scores, all_issues


def _calculate_weighted_score(scores: dict[str, float]) -> float:
    """Calculate weighted average score from category scores.

    Args:
        scores: Dictionary mapping assessment ID to score

    Returns:
        Weighted average score
    """
    total_weighted_score = 0.0
    total_weight = 0.0

    for assessment_id, score in scores.items():
        if assessment_id in ASSESSMENT_CATEGORIES:
            weight = ASSESSMENT_CATEGORIES[assessment_id]["weight"]
            total_weighted_score += score * weight
            total_weight += weight

    return total_weighted_score / total_weight if total_weight > 0 else 7.0


def _generate_markdown_content(
    scores: dict[str, float],
    overall_score: float,
    critical_issues: list[dict[str, Any]],
) -> str:
    """Generate markdown summary content.

    Args:
        scores: Dictionary mapping assessment ID to score
        overall_score: Overall weighted score
        critical_issues: List of critical issues

    Returns:
        Markdown content string
    """
    md_content = f"""# Comprehensive Assessment Summary

**Date**: {datetime.now().strftime("%Y-%m-%d")}
**Generated**: Automated via Jules Assessment Auto-Fix workflow
**Overall Score**: {overall_score:.1f}/10

## Executive Summary

Repository assessment completed across all {len(scores)} categories.

### Overall Health: {overall_score:.1f}/10

### Category Scores

| Category | Name | Score | Weight |
|----------|------|-------|--------|
"""

    for assessment_id in sorted(scores.keys()):
        if assessment_id in ASSESSMENT_CATEGORIES:
            cat_info = ASSESSMENT_CATEGORIES[assessment_id]
            score = scores[assessment_id]
            md_content += f"| **{assessment_id}** | {cat_info['name']} | {score:.1f} | {cat_info['weight']}x |\n"

    md_content += f"""
## Critical Issues

Found {len(critical_issues)} critical issues requiring immediate attention:

"""

    for i, issue in enumerate(critical_issues[:10], 1):
        md_content += f"{i}. **[{issue['severity']}]** {issue['description']} (Source: {issue['source']})\n"

    md_content += """
## Recommendations

1. Address all BLOCKER issues immediately
2. Create action plan for CRITICAL issues
3. Schedule remediation for MAJOR issues
4. Monitor trends in assessment scores

## Next Assessment

Recommended: 30 days from today

---

*Generated by Jules Assessment Auto-Fix*
"""
    return md_content


def _generate_json_data(
    scores: dict[str, float],
    overall_score: float,
    critical_issues: list[dict[str, Any]],
    all_issues: list[dict[str, Any]],
    reports_count: int,
) -> dict[str, Any]:
    """Generate JSON metrics data.

    Args:
        scores: Dictionary mapping assessment ID to score
        overall_score: Overall weighted score
        critical_issues: List of critical issues
        all_issues: List of all issues
        reports_count: Number of reports analyzed

    Returns:
        JSON-serializable dictionary
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "overall_score": round(overall_score, 2),
        "category_scores": {
            k: {
                "score": v,
                "name": ASSESSMENT_CATEGORIES[k]["name"],
                "weight": ASSESSMENT_CATEGORIES[k]["weight"],
            }
            for k, v in scores.items()
            if k in ASSESSMENT_CATEGORIES
        },
        "critical_issues": critical_issues,
        "total_issues": len(all_issues),
        "reports_analyzed": reports_count,
    }


def generate_summary(
    input_reports: list[Path],
    output_md: Path,
    output_json: Path,
) -> int:
    """Generate comprehensive summary from assessment reports.

    Args:
        input_reports: List of assessment report files
        output_md: Path to save markdown summary
        output_json: Path to save JSON metrics

    Returns:
        Exit code (0 = success, 1 = failure)
    """
    logger.info(f"Generating assessment summary from {len(input_reports)} reports...")

    # Collect scores and issues
    scores, all_issues = _collect_scores_and_issues(input_reports)

    # Calculate overall score
    overall_score = _calculate_weighted_score(scores)

    # Filter critical issues
    critical_issues = [
        i for i in all_issues if i["severity"] in ("BLOCKER", "CRITICAL")
    ]

    # Generate and save markdown
    md_content = _generate_markdown_content(scores, overall_score, critical_issues)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    safe_write_text(output_md, md_content)
    logger.info(f"✓ Markdown summary saved to {output_md}")

    # Generate and save JSON
    json_data = _generate_json_data(
        scores, overall_score, critical_issues, all_issues, len(input_reports)
    )
    safe_write_json(output_json, json_data, indent=2)
    logger.info(f"✓ JSON metrics saved to {output_json}")

    return 0


def main() -> int | None:
    """Parse CLI arguments and generate assessment summary."""
    parser = argparse.ArgumentParser(description="Generate assessment summary")
    parser.add_argument(
        "--input",
        nargs="+",
        type=Path,
        required=True,
        help="Input assessment report files (can use wildcards)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output markdown summary file",
    )
    parser.add_argument(
        "--json-output",
        required=True,
        type=Path,
        help="Output JSON metrics file",
    )

    args = parser.parse_args()

    # Expand wildcards if needed
    input_reports: list[Path] = []
    for pattern in args.input:
        if "*" in str(pattern):
            # Expand glob pattern
            input_reports.extend(Path(".").glob(str(pattern)))
        else:
            input_reports.append(pattern)

    # Filter to existing files
    input_reports = [p for p in input_reports if p.exists() and p.is_file()]

    if not input_reports:
        logger.error("No valid input reports found")
        return 1

    exit_code = generate_summary(input_reports, args.output, args.json_output)
    return exit_code


if __name__ == "__main__":
    sys.exit(main() or 0)
