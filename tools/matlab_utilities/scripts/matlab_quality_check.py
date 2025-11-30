#!/usr/bin/env python3
"""
MATLAB Quality Check Script

This script runs comprehensive quality checks on MATLAB code following the project's
.cursorrules.md requirements. It can be run from the command line and integrates
with the project's quality control system.

Usage:
    python scripts/matlab_quality_check.py [--strict] [--output-format json|text]
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MATLABQualityChecker:
    """Comprehensive MATLAB code quality checker."""

    def __init__(self, project_root: Path):
        """Initialize the MATLAB quality checker.

        Args:
            project_root: Path to the project root directory
        """
        self.project_root = project_root
        self.matlab_dir = project_root / "matlab"
        self.results = {
            "timestamp": None,
            "total_files": 0,
            "issues": [],
            "passed": True,
            "summary": "",
            "checks": {},
        }

    def check_matlab_files_exist(self) -> bool:
        """Check if MATLAB files exist in the project.

        Returns:
            True if MATLAB files are found, False otherwise
        """
        if not self.matlab_dir.exists():
            logger.info(
                f"MATLAB directory not found: {self.matlab_dir} (skipping MATLAB checks)",
            )
            return False

        m_files = list(self.matlab_dir.rglob("*.m"))
        self.results["total_files"] = len(m_files)

        if len(m_files) == 0:
            logger.info("No MATLAB files found (skipping MATLAB checks)")
            return False

        logger.info(f"Found {len(m_files)} MATLAB files")
        return True

    def run_matlab_quality_checks(self) -> dict[str, Any]:
        """Run MATLAB quality checks using the MATLAB script.

        Returns:
            Dictionary containing quality check results
        """
        try:
            # Check if we can run MATLAB from command line
            matlab_script = self.matlab_dir / "matlab_quality_config.m"
            if not matlab_script.exists():
                logger.error(f"MATLAB quality config script not found: {matlab_script}")
                return {"error": "MATLAB quality config script not found"}

            # Try to run MATLAB quality checks
            # Note: This requires MATLAB to be installed and accessible from command line
            try:
                # First, try to run the MATLAB script directly if possible
                result = self._run_matlab_script(matlab_script)
                return result
            except Exception as e:
                logger.warning(f"Could not run MATLAB script directly: {e}")
                # Fall back to static analysis
                return self._static_matlab_analysis()

        except Exception as e:
            logger.error(f"Error running MATLAB quality checks: {e}")
            return {"error": str(e)}

    def _run_matlab_script(self, script_path: Path) -> dict[str, Any]:
        """Attempt to run MATLAB script from command line.

        Args:
            script_path: Path to the MATLAB script

        Returns:
            Dictionary containing script results
        """
        try:
            # Try different ways to run MATLAB
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
                    logger.info(f"Trying command: {' '.join(cmd)}")
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        cwd=self.matlab_dir,
                        timeout=300, check=False,  # 5 minute timeout
                    )

                    if result.returncode == 0:
                        logger.info("MATLAB quality checks completed successfully")
                        return {
                            "success": True,
                            "output": result.stdout,
                            "method": "matlab_script",
                        }
                    else:
                        logger.warning(
                            f"Command failed with return code {result.returncode}",
                        )
                        logger.debug(f"stderr: {result.stderr}")

                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue

            # If all commands fail, fall back to static analysis
            logger.info("All MATLAB commands failed, falling back to static analysis")
            return self._static_matlab_analysis()

        except Exception as e:
            logger.error(f"Error running MATLAB script: {e}")
            return {"error": str(e)}

    def _static_matlab_analysis(self) -> dict[str, Any]:
        """Perform static analysis of MATLAB files without running MATLAB.

        Returns:
            Dictionary containing static analysis results
        """
        logger.info("Performing static MATLAB file analysis")

        issues = []
        total_files = 0

        # Analyze each MATLAB file
        for m_file in self.matlab_dir.rglob("*.m"):
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
        """Analyze a single MATLAB file for quality issues.

        Args:
            file_path: Path to the MATLAB file

        Returns:
            List of quality issues found
        """
        issues = []

        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
                lines = content.split("\n")

            import re

            # Track if we're in a function and nesting level
            in_function = False
            nesting_level = 0

            # Check for basic quality issues
            for i, line in enumerate(lines, 1):
                line_stripped = line.strip()
                line_original = line  # Keep original for indentation checks

                # Skip empty lines
                if not line_stripped:
                    continue

                # Skip comment-only lines for most checks (but check comments for banned patterns)
                is_comment = line_stripped.startswith("%")

                # Track function scope by monitoring nesting level
                if not is_comment:
                    # Check for keywords that increase nesting
                    # Note: arguments, properties, methods, events also have 'end'
                    if re.match(
                        r"\b(function|if|for|while|switch|try|parfor|classdef|arguments|properties|methods|events)\b",
                        line_stripped,
                    ):
                        if line_stripped.startswith("function"):
                            in_function = True
                        nesting_level += 1

                    # Check for 'end' keyword that decreases nesting
                    if re.match(r"\bend\b", line_stripped):
                        nesting_level -= 1
                        if nesting_level <= 0:
                            in_function = False
                            nesting_level = 0  # Prevent negative nesting

                # Check for function definition (for docstring and arguments validation)
                if line_stripped.startswith("function") and not is_comment:
                    # Check if next non-empty line has docstring
                    has_docstring = False
                    for j in range(i, min(i + 5, len(lines))):
                        next_line = lines[j].strip()
                        if next_line and not next_line.startswith("%"):
                            break
                        if next_line.startswith("%") and len(next_line) > 3:
                            has_docstring = True
                            break

                    if not has_docstring:
                        issues.append(
                            f"{file_path.name} (line {i}): Missing function docstring",
                        )

                    # Check for arguments validation block
                    has_arguments = False
                    for j in range(i, min(i + 15, len(lines))):
                        if re.search(r"\barguments\b", lines[j]):
                            has_arguments = True
                            break

                    if not has_arguments:
                        issues.append(
                            f"{file_path.name} (line {i}): Missing arguments validation block",
                        )

                # Check for banned patterns (in comments and code)
                banned_patterns = [
                    (r"\bTODO\b", "TODO placeholder found"),
                    (r"\bFIXME\b", "FIXME placeholder found"),
                    (r"\bHACK\b", "HACK comment found"),
                    (r"\bXXX\b", "XXX comment found"),
                    (r"<[A-Z_][A-Z0-9_]*>", "Angle bracket placeholder found"),
                    (r"\{\{.*?\}\}", "Template placeholder found"),
                ]

                for pattern, message in banned_patterns:
                    if re.search(pattern, line_stripped):
                        issues.append(f"{file_path.name} (line {i}): {message}")

                # Skip further checks for comment lines
                if is_comment:
                    continue

                # Check for common MATLAB anti-patterns
                if re.search(r"\beval\s*\(", line_stripped):
                    issues.append(
                        f"{file_path.name} (line {i}): Avoid using eval() - potential security risk and performance issue",
                    )

                if re.search(r"\bassignin\s*\(", line_stripped):
                    issues.append(
                        f"{file_path.name} (line {i}): Avoid using assignin() - violates encapsulation",
                    )

                if re.search(r"\bevalin\s*\(", line_stripped):
                    issues.append(
                        f"{file_path.name} (line {i}): Avoid using evalin() - violates encapsulation",
                    )

                # Check for global variables (often code smell)
                if re.search(r"\bglobal\s+\w+", line_stripped):
                    issues.append(
                        f"{file_path.name} (line {i}): Global variable usage - consider passing as argument",
                    )

                # Check for load without output (loads into workspace)
                if (
                    re.search(r"^\s*load\s+\w+", line_stripped)
                    and "=" not in line_stripped
                ):
                    issues.append(
                        f"{file_path.name} (line {i}): load without output variable - use 'data = load(...)' instead",
                    )

                # Check for magic numbers (but allow common values and known constants)
                # Exclude scientific notation, array indices, and common values
                magic_number_pattern = r"(?<![.\w])\d*\.\d+(?![.\w])"
                magic_numbers = re.findall(magic_number_pattern, line_stripped)

                # Known acceptable values
                acceptable_numbers = {
                    "0.0",
                    "0.5",
                    "1.0",
                    "2.0",
                    "3.0",
                    "4.0",
                    "5.0",
                    "10.0",
                    "100.0",
                    "1000.0",
                    "0.1",
                    "0.01",
                    "0.001",
                    "0.0001",  # Common tolerances
                }

                # Known physics constants (should be defined but at least flag with context)
                known_constants = {
                    "3.14159": "pi constant",
                    "3.1416": "pi constant",
                    "3.14": "pi constant",
                    "1.5708": "pi/2 constant",
                    "1.57": "pi/2 constant",
                    "0.7854": "pi/4 constant",
                    "0.785": "pi/4 constant",
                    "9.81": "gravitational acceleration",
                    "9.8": "gravitational acceleration",
                    "9.807": "gravitational acceleration",
                }

                for num in magic_numbers:
                    # Check if it's a known constant
                    if num in known_constants:
                        issues.append(
                            f"{file_path.name} (line {i}): Magic number {num} ({known_constants[num]}) - define as named constant",
                        )
                    elif num not in acceptable_numbers:
                        # Check if the number appears before a comment on same line
                        comment_idx = line_original.find("%")
                        num_idx = line_original.find(num)
                        if comment_idx == -1 or (
                            num_idx != -1 and num_idx < comment_idx
                        ):
                            issues.append(
                                f"{file_path.name} (line {i}): Magic number {num} should be defined as constant with units and source",
                            )

                # Check for clear/clc/close all in functions (bad practice)
                if in_function:
                    if re.search(r"\bclear\b(?!\s+\w+)", line_stripped):
                        issues.append(
                            f"{file_path.name} (line {i}): Avoid 'clear' in functions - can clear function variables",
                        )
                    if re.search(r"\bclc\b", line_stripped):
                        issues.append(
                            f"{file_path.name} (line {i}): Avoid 'clc' in functions - affects user's workspace",
                        )
                    if re.search(r"\bclose\s+all\b", line_stripped):
                        issues.append(
                            f"{file_path.name} (line {i}): Avoid 'close all' in functions - closes user's figures",
                        )

                # Check for exist() usage (often code smell, prefer try/catch or validation)
                if re.search(r"\bexist\s*\(", line_stripped):
                    issues.append(
                        f"{file_path.name} (line {i}): Consider using validation or try/catch instead of exist()",
                    )

                # Check for addpath in functions (should be in startup.m or managed externally)
                if in_function and re.search(r"\baddpath\s*\(", line_stripped):
                    issues.append(
                        f"{file_path.name} (line {i}): Avoid addpath in functions - manage paths externally",
                    )

        except Exception as e:
            issues.append(f"{file_path.name}: Could not analyze file - {e!s}")

        return issues

    def run_all_checks(self) -> dict[str, Any]:
        """Run all MATLAB quality checks.

        Returns:
            Dictionary containing all quality check results
        """
        logger.info("Starting MATLAB quality checks")

        # Check if MATLAB files exist
        if not self.check_matlab_files_exist():
            self.results["passed"] = True
            self.results["summary"] = "[SKIP] No MATLAB files to check - passed"
            return self.results

        # Run MATLAB quality checks
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
                    f"[PASS] MATLAB quality checks PASSED ({self.results['total_files']} files checked)"
                )
            else:
                self.results["passed"] = False
                self.results["summary"] = (
                    f"[FAIL] MATLAB quality checks FAILED ({self.results['total_files']} files checked)"
                )

        return self.results


def main():
    """Main entry point for the MATLAB quality check script."""
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

    # Get project root
    project_root = Path(args.project_root).resolve()
    if not project_root.exists():
        logger.error(f"Project root does not exist: {project_root}")
        sys.exit(1)

    # Initialize and run quality checks
    checker = MATLABQualityChecker(project_root)
    results = checker.run_all_checks()

    # Output results
    if args.output_format == "json":
        print(json.dumps(results, indent=2, default=str))
    else:
        print("\n" + "=" * 60)
        print("MATLAB QUALITY CHECK RESULTS")
        print("=" * 60)
        print(f"Timestamp: {results.get('timestamp', 'N/A')}")
        print(f"Total Files: {results.get('total_files', 0)}")
        print(f"Status: {'PASSED' if results.get('passed', False) else 'FAILED'}")
        print(f"Summary: {results.get('summary', 'N/A')}")

        if results.get("issues"):
            print(f"\nIssues Found ({len(results['issues'])}):")
            for i, issue in enumerate(results["issues"], 1):
                print(f"  {i}. {issue}")

        print("\n" + "=" * 60)

    # Exit with appropriate code
    sys.exit(0 if results.get("passed", False) else 1)


if __name__ == "__main__":
    main()
