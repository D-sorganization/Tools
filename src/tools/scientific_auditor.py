"""Scientific auditor — detect potential computation risks in Python code.

Scans Python source files for common scientific computing pitfalls like
division by variables (singularity risk) and trig functions called with
numeric constants (unit ambiguity).
"""

import ast
import json
import logging
import sys
from pathlib import Path

from src.shared.python.contracts import require

logger = logging.getLogger(__name__)


class ScienceAuditor(ast.NodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.risks: list[dict[str, object]] = []

    def visit_BinOp(self, node: ast.BinOp) -> None:  # noqa: N802
        # 1. Division Safety
        if isinstance(node.op, ast.Div) and not (
            isinstance(node.right, ast.Constant) and node.right.value != 0
        ):
            self.risks.append(
                {
                    "line": node.lineno,
                    "type": "Singularity Risk",
                    "msg": "Division by variable detected. Check denominator.",
                },
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        # 2. Trig Safety
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in ["sin", "cos", "tan"]
            and any(
                isinstance(arg, ast.Constant) and isinstance(arg.value, int | float)
                for arg in node.args
            )
        ):
            self.risks.append(
                {
                    "line": node.lineno,
                    "type": "Unit Ambiguity",
                    "msg": (
                        "Trig function called with a numeric constant. "
                        "Check if argument is in radians "
                        "(Python math module expects radians)."
                    ),
                },
            )
        self.generic_visit(node)


def audit_file(file_path: Path | str) -> list[dict[str, object]]:
    """Audit a single Python file for scientific computing risks."""
    path = Path(file_path)
    require(path.exists(), f"File does not exist: {path}")
    require(path.is_file(), f"Path is not a file: {path}")
    require(path.suffix == ".py", f"File must be a Python script: {path}")

    auditor = ScienceAuditor()
    try:
        with path.open(encoding="utf-8") as source:
            auditor.visit(ast.parse(source.read()))
    except (PermissionError, OSError) as e:
        logger.error("Permission or OS error analyzing %s: %s", path, e)
    except SyntaxError as e:
        logger.error("Syntax error parsing %s: %s", path, e)
    return auditor.risks


def audit_directory(dir_path: Path | str) -> list[dict[str, object]]:
    """Audit all Python files recursively in a directory."""
    path = Path(dir_path)
    require(path.exists(), f"Directory does not exist: {path}")
    require(path.is_dir(), f"Path is not a directory: {path}")

    all_risks: list[dict[str, object]] = []
    for py_file in path.rglob("*.py"):
        if "test" in py_file.name:
            continue
        file_risks = audit_file(py_file)
        # Append file name to risks for clarity in directory audit
        for risk in file_risks:
            risk["file"] = str(py_file)
        all_risks.extend(file_risks)

    return all_risks


def main() -> None:
    # Basic CLI wrapper
    target_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path()
    risks = audit_directory(target_dir)

    if risks:
        sys.stdout.write(json.dumps(risks, indent=2) + "\n")
        sys.exit(1)
    else:
        sys.stdout.write("[]\n")


if __name__ == "__main__":
    main()
