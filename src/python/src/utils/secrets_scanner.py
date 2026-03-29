"""Secrets scanner utility for detecting hardcoded secrets in source code.

Scans Python source files for patterns that may indicate hardcoded secrets,
API keys, passwords, or tokens. Designed for pre-commit / CI integration.

Usage:
    from utils.secrets_scanner import scan_file, scan_directory

    issues = scan_directory("src/")
    for issue in issues:
        print(f"{issue['file']}:{issue['line']}: {issue['pattern']}")
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Patterns that indicate potential hardcoded secrets
_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("AWS Access Key", re.compile(r"AKIA[A-Z0-9]{16}")),
    ("GitHub Token", re.compile(r"ghp_[A-Za-z0-9]{36}")),
    ("OpenAI API Key", re.compile(r"sk-[A-Za-z0-9]{20,}")),
    ("Slack Token", re.compile(r"xox[bpsa]-[A-Za-z0-9-]+")),
    ("Generic API Key Assignment", re.compile(
        r"""(?:api[_-]?key|secret[_-]?key|auth[_-]?token|password|private[_-]?key)\s*=\s*["'][A-Za-z0-9+/=@#$%^&*!]{8,}["']""",
        re.IGNORECASE,
    )),
    ("Bearer Token", re.compile(r"""["\']Bearer\s+[A-Za-z0-9._-]{20,}["\']""")),
    ("Base64 Private Key", re.compile(r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----")),
]

# Lines matching these patterns are false positives (examples, docs, tests)
_IGNORE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*#"),          # Comments
    re.compile(r"^\s*>>>"),        # Doctest examples
    re.compile(r"example", re.I), # Example text
    re.compile(r"placeholder", re.I),
    re.compile(r"xxxx", re.I),    # Redacted values
]


@dataclass
class SecretFinding:
    """A potential hardcoded secret found in source code."""

    file: str
    line: int
    pattern: str
    snippet: str
    is_likely_false_positive: bool = False


def scan_line(line: str, line_num: int, filepath: str) -> list[SecretFinding]:
    """Scan a single line for potential secrets.

    Args:
        line: The source line to scan.
        line_num: 1-based line number.
        filepath: Path to the file being scanned.

    Returns:
        List of findings for this line.
    """
    findings: list[SecretFinding] = []
    for name, pattern in _SECRET_PATTERNS:
        if pattern.search(line):
            is_fp = any(ip.search(line) for ip in _IGNORE_PATTERNS)
            findings.append(SecretFinding(
                file=filepath,
                line=line_num,
                pattern=name,
                snippet=line.strip()[:120],
                is_likely_false_positive=is_fp,
            ))
    return findings


def scan_file(filepath: str | Path) -> list[SecretFinding]:
    """Scan a file for potential hardcoded secrets.

    Args:
        filepath: Path to the file to scan.

    Returns:
        List of findings.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return []
    if not filepath.suffix == ".py":
        return []

    findings: list[SecretFinding] = []
    try:
        text = filepath.read_text(encoding="utf-8", errors="ignore")
        for i, line in enumerate(text.splitlines(), 1):
            findings.extend(scan_line(line, i, str(filepath)))
    except OSError:
        pass
    return findings


def scan_directory(
    directory: str | Path,
    exclude_dirs: set[str] | None = None,
) -> list[SecretFinding]:
    """Recursively scan a directory for potential hardcoded secrets.

    Args:
        directory: Root directory to scan.
        exclude_dirs: Directory names to skip (e.g., {'__pycache__', '.git'}).

    Returns:
        List of findings across all files.
    """
    if exclude_dirs is None:
        exclude_dirs = {"__pycache__", ".git", "node_modules", ".venv", "venv"}

    directory = Path(directory)
    findings: list[SecretFinding] = []

    for py_file in directory.rglob("*.py"):
        if any(part in exclude_dirs for part in py_file.parts):
            continue
        findings.extend(scan_file(py_file))

    return findings


def report_findings(findings: list[SecretFinding], show_false_positives: bool = False) -> str:
    """Format findings into a human-readable report.

    Args:
        findings: List of secret findings.
        show_false_positives: If True, include likely false positives.

    Returns:
        Formatted report string.
    """
    filtered = findings if show_false_positives else [
        f for f in findings if not f.is_likely_false_positive
    ]

    if not filtered:
        return "No hardcoded secrets detected."

    lines = [f"Found {len(filtered)} potential secret(s):\n"]
    for f in filtered:
        fp_marker = " [likely false positive]" if f.is_likely_false_positive else ""
        lines.append(f"  {f.file}:{f.line}: {f.pattern}{fp_marker}")
        lines.append(f"    {f.snippet}")
    return "\n".join(lines)
