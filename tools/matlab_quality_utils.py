"""Shared utilities for MATLAB quality checks."""

import argparse
import json
import logging
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from tools.compatibility import UTC
from tools.quality_utils import check_patterns_in_content

# Constants
MATLAB_SCRIPT_TIMEOUT_SECONDS: int = 300

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MATLABQualityChecker:
    """Comprehensive MATLAB code quality checker."""

    # Compiled regex patterns for performance
    LOAD_PATTERN = re.compile(r"^\s*load\s+(?:\w+|\([^)]+\))")
    ASSIGNMENT_PATTERN = re.compile(r"\w+\s*=\s*load\s*[\(]")

    BANNED_PATTERNS = [
        (r"\bTODO\b", "TODO placeholder found"),
        (r"\bFIXME\b", "FIXME placeholder found"),
        (r"\bHACK\b", "HACK comment found"),
        (r"\bXXX\b", "XXX comment found"),
        (r"<[A-Z_][A-Z0-9_]*>", "Angle bracket placeholder found"),
        (r"\{\{.*?\}\}", "Template placeholder found"),
    ]

    def __init__(self, project_root: Path) -> None:
        """Initialize the MATLAB quality checker."""
        self.project_root = project_root
        self.matlab_dir = project_root / "matlab"
        self.results: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "total_files": 0,
            "issues": [],
            "passed": True,
            "summary": "",
            "checks": {},
        }

    def check_matlab_files_exist(self) -> bool:
        """Check if MATLAB files exist in the project."""
        if not self.matlab_dir.exists():
            logger.info(
                "MATLAB directory not found: %s (skipping MATLAB checks)",
                self.matlab_dir,
            )
            return False

        m_files = [f for f in self.matlab_dir.rglob("*.m") if "archive" not in f.parts]
        self.results["total_files"] = len(m_files)

        if len(m_files) == 0:
            logger.info("No MATLAB files found (skipping MATLAB checks)")
            return False

        logger.info("Found %d MATLAB files", len(m_files))
        return True

    def run_matlab_quality_checks(self) -> dict[str, Any]:
        """Run MATLAB quality checks using the MATLAB script."""
        try:
            matlab_script = self.matlab_dir / "matlab_quality_config.m"
            if not matlab_script.exists():
                logger.info(
                    "MATLAB quality config script not found, using static analysis",
                )
                return self._static_matlab_analysis()

            try:
                return self._run_matlab_script(matlab_script)
            except Exception as e:
                logger.warning("Could not run MATLAB script directly: %s", e)
                return self._static_matlab_analysis()

        except Exception as e:
            logger.exception("Error running MATLAB quality checks")
            return {"error": str(e)}

    def _run_matlab_script(self, script_path: Path) -> dict[str, Any]:
        """Attempt to run MATLAB script from command line."""
        try:
            commands = [
                ["matlab", "-batch", f"run('{script_path}')"],
                [
                    "matlab",
                    "-nosplash",
                    "-nodesktop",
                    "-batch",
                    f"run('{script_path}')",
                ],
                ["octave", "--no-gui", "--eval", f"run('{script_path}')"],
            ]

            for cmd in commands:
                try:
                    logger.info("Trying command: %s", " ".join(cmd))
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        cwd=self.matlab_dir,
                        timeout=MATLAB_SCRIPT_TIMEOUT_SECONDS,
                        check=False,
                    )

                    if result.returncode == 0:
                        logger.info("MATLAB quality checks completed successfully")
                        return {
                            "success": True,
                            "output": result.stdout,
                            "method": "matlab_script",
                            "passed": True,
                        }
                    logger.warning(
                        "Command failed with return code %s",
                        result.returncode,
                    )
                    logger.debug("stderr: %s", result.stderr)

                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue

            logger.info("All MATLAB commands failed, falling back to static analysis")
            return self._static_matlab_analysis()

        except Exception as e:
            logger.exception("Error running MATLAB script")
            return {"error": str(e)}

    def _static_matlab_analysis(self) -> dict[str, Any]:
        """Perform static analysis of MATLAB files without running MATLAB."""
        logger.info("Performing static MATLAB file analysis")

        issues = []
        total_files = 0

        for m_file in self.matlab_dir.rglob("*.m"):
            if "archive" in m_file.parts:
                continue
            total_files += 1
            file_issues = self._analyze_matlab_file(m_file)
            issues.extend(file_issues)

        self.results["total_files"] = total_files
        self.results["issues"] = issues
        self.results["passed"] = len(issues) == 0

        return {
            "success": True,
            "method": "static_analysis",
            "total_files": total_files,
            "issues": issues,
            "passed": len(issues) == 0,
        }

    def _analyze_matlab_file(self, file_path: Path) -> list[str]:
        """Analyze a single MATLAB file for quality issues."""
        issues = []

        try:
            with file_path.open(encoding="utf-8", errors="ignore") as f:
                lines = f.read().splitlines()

            # Use shared pattern checker
            issues.extend(
                check_patterns_in_content(lines, self.BANNED_PATTERNS, file_path)
            )

            in_function = False
            nesting_level = 0

            for i, line in enumerate(lines, 1):
                line_stripped = line.strip()
                line_original = line

                if not line_stripped:
                    continue

                is_comment = line_stripped.startswith("%")

                if not is_comment:
                    if re.match(
                        r"\b(function|if|for|while|switch|try|parfor|classdef|arguments|properties|methods|events)\b",
                        line_stripped,
                    ):
                        if line_stripped.startswith("function"):
                            in_function = True
                        nesting_level += 1

                    if re.match(r"\bend\b", line_stripped):
                        nesting_level -= 1
                        if nesting_level <= 0:
                            in_function = False
                            nesting_level = 0

                if line_stripped.startswith("function") and not is_comment:
                    has_docstring = False
                    min_docstring_length = 3
                    for j in range(i, min(i + 5, len(lines))):
                        next_line = lines[j].strip()
                        if next_line and not next_line.startswith("%"):
                            break
                        if (
                            next_line.startswith("%")
                            and len(next_line) > min_docstring_length
                        ):
                            has_docstring = True
                            break

                    if not has_docstring:
                        issues.append(
                            f"{file_path.name} (line {i}): Missing function docstring",
                        )

                    has_arguments = False
                    for j in range(i, min(i + 25, len(lines))):
                        line_check = lines[j].strip()
                        if line_check.startswith("%"):
                            continue
                        if re.search(r"\barguments\b", line_check):
                            has_arguments = True
                            break

                    if not has_arguments:
                        issues.append(
                            f"{file_path.name} (line {i}): "
                            "Missing arguments validation block",
                        )

                if is_comment:
                    continue

                if re.search(r"\beval\s*\(", line_stripped):
                    issues.append(
                        f"{file_path.name} (line {i}): "
                        "Avoid using eval() - potential security risk and "
                        "performance issue",
                    )

                if re.search(r"\bassignin\s*\(", line_stripped):
                    issues.append(
                        f"{file_path.name} (line {i}): "
                        "Avoid using assignin() - violates encapsulation",
                    )

                if re.search(r"\bevalin\s*\(", line_stripped):
                    issues.append(
                        f"{file_path.name} (line {i}): "
                        "Avoid using evalin() - violates encapsulation",
                    )

                if re.search(r"\bglobal\s+\w+", line_stripped):
                    issues.append(
                        f"{file_path.name} (line {i}): "
                        "Global variable usage - consider passing as argument",
                    )

                if self.LOAD_PATTERN.search(
                    line_stripped,
                ) and not self.ASSIGNMENT_PATTERN.search(line_stripped):
                    issues.append(
                        f"{file_path.name} (line {i}): "
                        "load without output variable - use 'data = load(...)' instead",
                    )

                magic_number_pattern = r"(?<![.\w])(?:\d+\.\d+|\d+)(?![.\w])"
                magic_numbers = re.findall(magic_number_pattern, line_stripped)

                acceptable_numbers = {
                    "0",
                    "0.0",
                    "1",
                    "1.0",
                    "2",
                    "2.0",
                    "3",
                    "3.0",
                    "4",
                    "4.0",
                    "5",
                    "5.0",
                    "10",
                    "10.0",
                    "42",
                    "42.0",
                    "100",
                    "100.0",
                    "1000",
                    "1000.0",
                    "0.5",
                    "0.1",
                    "0.01",
                    "0.001",
                    "0.0001",
                }

                known_constants = {
                    "3.14159": "pi constant [dimensionless] - mathematical constant",
                    "3.1416": "pi constant [dimensionless] - mathematical constant",
                    "3.14": "pi constant [dimensionless] - mathematical constant",
                    "1.5708": "pi/2 constant [dimensionless] - mathematical constant",
                    "1.57": "pi/2 constant [dimensionless] - mathematical constant",
                    "0.7854": "pi/4 constant [dimensionless] - mathematical constant",
                    "0.785": "pi/4 constant [dimensionless] - mathematical constant",
                    "9.81": "gravitational acceleration [m/s²] - approximate standard gravity",
                    "9.8": "gravitational acceleration [m/s²] - approximate standard gravity",
                    "9.807": "gravitational acceleration [m/s²] - approximate standard gravity",
                }

                for num in magic_numbers:
                    if num in known_constants:
                        issues.append(
                            f"{file_path.name} (line {i}): Magic number {num} "
                            f"({known_constants[num]}) - define as named constant",
                        )
                    elif num not in acceptable_numbers:
                        comment_idx = line_original.find("%")
                        num_idx = line_original.find(num)
                        if comment_idx == -1 or (
                            num_idx != -1 and num_idx < comment_idx
                        ):
                            issues.append(
                                f"{file_path.name} (line {i}): Magic number {num} "
                                "should be defined as constant with units and source",
                            )

                if in_function:
                    if re.search(
                        r"\bclear\s+(all|global)\b", line_stripped, re.IGNORECASE
                    ):
                        issues.append(
                            f"{file_path.name} (line {i}): Avoid 'clear all' or "
                            "'clear global' in functions - clears all variables, "
                            "functions, and MEX links",
                        )
                    elif re.search(r"\bclear\b(?!\s+\w+)", line_stripped):
                        issues.append(
                            f"{file_path.name} (line {i}): Avoid 'clear' in functions "
                            "- can clear function variables",
                        )
                    if re.search(r"\bclc\b", line_stripped):
                        issues.append(
                            f"{file_path.name} (line {i}): Avoid 'clc' in functions "
                            "- affects user's workspace",
                        )
                    if re.search(r"\bclose\s+all\b", line_stripped):
                        issues.append(
                            f"{file_path.name} (line {i}): Avoid 'close all' in "
                            "functions - closes user's figures",
                        )

                if re.search(r"\bexist\s*\(", line_stripped):
                    issues.append(
                        f"{file_path.name} (line {i}): Consider using validation or "
                        "try/catch instead of exist()",
                    )

                if in_function and re.search(r"\baddpath\s*\(", line_stripped):
                    issues.append(
                        f"{file_path.name} (line {i}): Avoid addpath in functions "
                        "- manage paths externally",
                    )

        except Exception as e:
            issues.append(f"{file_path.name}: Could not analyze file - {e!s}")

        return issues

    def run_all_checks(self) -> dict[str, Any]:
        """Run all MATLAB quality checks."""
        logger.info("Starting MATLAB quality checks")

        if not self.check_matlab_files_exist():
            self.results["passed"] = True
            self.results["summary"] = "[SKIP] No MATLAB files to check - passed"
            return self.results

        matlab_results = self.run_matlab_quality_checks()

        if "error" in matlab_results:
            self.results["passed"] = False
            self.results["summary"] = (
                f"MATLAB quality checks failed: {matlab_results['error']}"
            )
            self.results["checks"]["matlab"] = matlab_results
        else:
            self.results["checks"]["matlab"] = matlab_results
            if matlab_results.get("passed", False):
                self.results["summary"] = (
                    f"[PASS] MATLAB quality checks PASSED "
                    f"({self.results['total_files']} files checked)"
                )
            else:
                self.results["passed"] = False
                self.results["summary"] = (
                    f"[FAIL] MATLAB quality checks FAILED "
                    f"({self.results['total_files']} files checked)"
                )

        return self.results


def run_matlab_quality_checks_cli() -> None:
    """Main entry point for command line execution."""
    parser = argparse.ArgumentParser(description="MATLAB Code Quality Checker")
    parser.add_argument("--strict", action="store_true", help="Enable strict mode")
    parser.add_argument(
        "--output-format",
        choices=["json", "text"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root directory (default: current directory)",
    )

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    if not project_root.exists():
        logger.error("Project root does not exist: %s", project_root)
        sys.exit(1)

    checker = MATLABQualityChecker(project_root)
    results = checker.run_all_checks()

    if args.output_format == "json":
        print(json.dumps(results, indent=2, default=str))
    else:
        logger.info("\n" + "=" * 60)
        logger.info("MATLAB QUALITY CHECK RESULTS")
        logger.info("=" * 60)
        logger.info(f"Timestamp: {results.get('timestamp', 'N/A')}")
        logger.info(f"Total Files: {results.get('total_files', 0)}")
        logger.info(
            f"Status: {'PASSED' if results.get('passed', False) else 'FAILED'}",
        )
        logger.info(f"Summary: {results.get('summary', 'N/A')}")

        if results.get("issues"):
            logger.info(f"\nIssues Found ({len(results['issues'])}):")
            for i, issue in enumerate(results["issues"], 1):
                logger.info(f"  {i}. {issue}")

        logger.info("\n" + "=" * 60)

    passed = results.get("passed", False)
    has_issues = bool(results.get("issues"))

    exit_code = (
        (0 if (passed and not has_issues) else 1)
        if args.strict
        else (0 if passed else 1)
    )

    sys.exit(exit_code)
