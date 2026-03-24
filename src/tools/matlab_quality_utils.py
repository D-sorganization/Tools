"""Shared utilities for MATLAB quality checks."""

import argparse
import json
import logging
import re
import subprocess
import sys
from datetime import datetime, timezone  # noqa: UP017
from pathlib import Path
from typing import Any

from contracts import require

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

    def __init__(self, project_root: Path) -> None:
        """Initialize the MATLAB quality checker."""
        assert project_root is not None, "project_root must be provided"
        require(isinstance(project_root, Path), "project_root must be a Path")
        require(
            project_root.is_absolute(),
            f"project_root must be absolute, got: {project_root}",
        )
        self.project_root = project_root
        self.matlab_dir = project_root / "matlab"
        self.results: dict[str, Any] = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),  # noqa: UP017
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
            except (OSError, RuntimeError, ValueError) as e:
                logger.warning("Could not run MATLAB script directly: %s", e)
                return self._static_matlab_analysis()

        except (PermissionError, OSError) as e:
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

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
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
        assert file_path is not None, "file_path must be provided"
        issues: list[str] = []

        try:
            with file_path.open(encoding="utf-8", errors="ignore") as f:
                lines = f.read().splitlines()

            in_function = False
            nesting_level = 0

            for i, line in enumerate(lines, 1):
                line_stripped = line.strip()
                if not line_stripped:
                    continue

                is_comment = line_stripped.startswith("%")

                if not is_comment:
                    in_function, nesting_level = self._track_nesting(
                        line_stripped,
                        in_function,
                        nesting_level,
                    )

                if line_stripped.startswith("function") and not is_comment:
                    self._check_function_definition(
                        file_path,
                        lines,
                        i,
                        issues,
                    )

                self._check_banned_patterns(
                    file_path,
                    line_stripped,
                    i,
                    issues,
                )

                if is_comment:
                    continue

                self._check_anti_patterns(
                    file_path,
                    line_stripped,
                    i,
                    issues,
                )
                self._check_magic_numbers(
                    file_path,
                    line,
                    line_stripped,
                    i,
                    issues,
                )
                self._check_workspace_pollution(
                    file_path,
                    line_stripped,
                    i,
                    in_function,
                    issues,
                )

        except (PermissionError, OSError) as e:
            issues.append(
                f"{file_path.name}: Could not analyze file - {e!s}",
            )

        return issues

    @staticmethod
    def _track_nesting(
        line_stripped: str,
        in_function: bool,
        nesting_level: int,
    ) -> tuple[bool, int]:
        """Update nesting state based on current line."""
        assert line_stripped is not None, "line_stripped must be provided"
        if re.match(
            r"\b(function|if|for|while|switch|try|parfor|"
            r"classdef|arguments|properties|methods|events)\b",
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

        return in_function, nesting_level

    @staticmethod
    def _check_function_definition(
        file_path: Path,
        lines: list[str],
        line_num: int,
        issues: list[str],
    ) -> None:
        """Check for function docstring and arguments block."""
        assert file_path is not None, "file_path must be provided"
        require(isinstance(file_path, Path), "file_path must be a Path")
        require(isinstance(lines, list), "lines must be a list")
        require(
            isinstance(line_num, int) and line_num >= 1,
            "line_num must be a positive integer",
        )
        require(isinstance(issues, list), "issues must be a list")
        has_docstring = False
        min_docstring_length = 3
        for j in range(line_num, min(line_num + 5, len(lines))):
            next_line = lines[j].strip()
            if next_line and not next_line.startswith("%"):
                break
            if next_line.startswith("%") and len(next_line) > min_docstring_length:
                has_docstring = True
                break

        if not has_docstring:
            issues.append(
                f"{file_path.name} (line {line_num}): Missing function docstring",
            )

        has_arguments = False
        for j in range(line_num, min(line_num + 25, len(lines))):
            line_check = lines[j].strip()
            if line_check.startswith("%"):
                continue
            if re.search(r"\barguments\b", line_check):
                has_arguments = True
                break

        if not has_arguments:
            issues.append(
                f"{file_path.name} (line {line_num}): "
                "Missing arguments validation block",
            )

    @staticmethod
    def _check_banned_patterns(
        file_path: Path,
        line_stripped: str,
        line_num: int,
        issues: list[str],
    ) -> None:
        """Check for DEFERRED, REVIEW, HACK, XXX, and placeholders."""
        assert file_path is not None, "file_path must be provided"
        require(isinstance(file_path, Path), "file_path must be a Path")
        require(isinstance(line_stripped, str), "line_stripped must be a string")
        require(isinstance(issues, list), "issues must be a list")
        banned = [
            (r"\bDEFERRED\b", "DEFERRED placeholder found"),
            (r"\bREVIEW\b", "REVIEW placeholder found"),
            (r"\bHACK\b", "HACK comment found"),
            (r"\bXXX\b", "XXX comment found"),
            (
                r"<[A-Z_][A-Z0-9_]*>",
                "Angle bracket placeholder found",
            ),
            (r"\{\{.*?\}\}", "Template placeholder found"),
        ]
        for pattern, message in banned:
            if re.search(pattern, line_stripped):
                issues.append(
                    f"{file_path.name} (line {line_num}): {message}",
                )

    def _check_anti_patterns(
        self,
        file_path: Path,
        line_stripped: str,
        line_num: int,
        issues: list[str],
    ) -> None:
        """Check for eval, assignin, evalin, global, load."""
        assert file_path is not None, "file_path must be provided"
        anti_patterns = [
            (
                r"\beval\s*\(",
                "Avoid eval() - security risk and perf issue",
            ),
            (
                r"\bassignin\s*\(",
                "Avoid assignin() - violates encapsulation",
            ),
            (
                r"\bevalin\s*\(",
                "Avoid evalin() - violates encapsulation",
            ),
            (
                r"\bglobal\s+\w+",
                "Global variable - consider passing as arg",
            ),
            (
                r"\bexist\s*\(",
                "Consider try/catch instead of exist()",
            ),
        ]
        for pattern, message in anti_patterns:
            if re.search(pattern, line_stripped):
                issues.append(
                    f"{file_path.name} (line {line_num}): {message}",
                )

        if self.LOAD_PATTERN.search(
            line_stripped,
        ) and not self.ASSIGNMENT_PATTERN.search(line_stripped):
            issues.append(
                f"{file_path.name} (line {line_num}): "
                "load without output variable - "
                "use 'data = load(...)' instead",
            )

    _ACCEPTABLE_NUMBERS = frozenset(
        {
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
    )

    _KNOWN_CONSTANTS: dict[str, str] = {
        "3.14159": "pi [dimensionless]",
        "3.1416": "pi [dimensionless]",
        "3.14": "pi [dimensionless]",
        "1.5708": "pi/2 [dimensionless]",
        "1.57": "pi/2 [dimensionless]",
        "0.7854": "pi/4 [dimensionless]",
        "0.785": "pi/4 [dimensionless]",
        "9.81": "gravity [m/s²]",
        "9.8": "gravity [m/s²]",
        "9.807": "gravity [m/s²]",
    }

    def _check_magic_numbers(
        self,
        file_path: Path,
        line_original: str,
        line_stripped: str,
        line_num: int,
        issues: list[str],
    ) -> None:
        """Detect magic numbers that should be named constants."""
        assert file_path is not None, "file_path must be provided"
        pattern = r"(?<![.\w])(?:\d+\.\d+|\d+)(?![.\w])"
        magic_numbers = re.findall(pattern, line_stripped)

        for num in magic_numbers:
            if num in self._KNOWN_CONSTANTS:
                issues.append(
                    f"{file_path.name} (line {line_num}): "
                    f"Magic number {num} "
                    f"({self._KNOWN_CONSTANTS[num]}) "
                    "- define as named constant",
                )
            elif num not in self._ACCEPTABLE_NUMBERS:
                comment_idx = line_original.find("%")
                num_idx = line_original.find(num)
                if comment_idx == -1 or (num_idx != -1 and num_idx < comment_idx):
                    issues.append(
                        f"{file_path.name} (line {line_num}): "
                        f"Magic number {num} should be "
                        "defined as constant with units",
                    )

    @staticmethod
    def _check_workspace_pollution(
        file_path: Path,
        line_stripped: str,
        line_num: int,
        in_function: bool,
        issues: list[str],
    ) -> None:
        """Check for clear all, clc, close all, addpath in functions."""
        assert file_path is not None, "file_path must be provided"
        require(isinstance(file_path, Path), "file_path must be a Path")
        require(isinstance(line_stripped, str), "line_stripped must be a string")
        require(isinstance(issues, list), "issues must be a list")
        if not in_function:
            return

        if re.search(
            r"\bclear\s+(all|global)\b",
            line_stripped,
            re.IGNORECASE,
        ):
            issues.append(
                f"{file_path.name} (line {line_num}): "
                "Avoid 'clear all/global' in functions",
            )
        elif re.search(r"\bclear\b(?!\s+\w+)", line_stripped):
            issues.append(
                f"{file_path.name} (line {line_num}): Avoid 'clear' in functions",
            )
        if re.search(r"\bclc\b", line_stripped):
            issues.append(
                f"{file_path.name} (line {line_num}): Avoid 'clc' in functions",
            )
        if re.search(r"\bclose\s+all\b", line_stripped):
            issues.append(
                f"{file_path.name} (line {line_num}): Avoid 'close all' in functions",
            )
        if re.search(r"\baddpath\s*\(", line_stripped):
            issues.append(
                f"{file_path.name} (line {line_num}): Avoid addpath in functions",
            )

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
    require(project_root.exists(), f"Project root does not exist: {project_root}")

    checker = MATLABQualityChecker(project_root)
    results = checker.run_all_checks()

    if args.output_format == "json":
        logger.info(json.dumps(results, indent=2, default=str))
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
