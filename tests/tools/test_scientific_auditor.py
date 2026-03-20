"""Comprehensive TDD suite for scientific_auditor.py.

Covers ScienceAuditor AST visitor branches (division, trig), audit_file,
audit_directory, contract violations, graceful error handling.
"""

import ast
from pathlib import Path

import pytest

from src.shared.python.contracts import PreconditionError
from tools.scientific_auditor import ScienceAuditor, audit_directory, audit_file

# ─── ScienceAuditor.visit_BinOp ────────────────────────────────


def test_singularity_risk():
    auditor = ScienceAuditor()
    tree = ast.parse("x = 10 / y")
    auditor.visit(tree)
    assert len(auditor.risks) == 1
    assert auditor.risks[0]["type"] == "Singularity Risk"


def test_safe_division():
    auditor = ScienceAuditor()
    tree = ast.parse("x = 10 / 2")
    auditor.visit(tree)
    assert len(auditor.risks) == 0


def test_division_by_zero_is_flagged():
    """0 is a constant with value==0, so division by it is still flagged."""
    auditor = ScienceAuditor()
    tree = ast.parse("x = a / 0")
    auditor.visit(tree)
    assert any(r["type"] == "Singularity Risk" for r in auditor.risks)


def test_division_by_nonzero_float_not_flagged():
    auditor = ScienceAuditor()
    tree = ast.parse("x = a / 2.5")
    auditor.visit(tree)
    assert auditor.risks == []


# ─── ScienceAuditor.visit_Call ─────────────────────────────────


def test_trig_ambiguity():
    auditor = ScienceAuditor()
    tree = ast.parse("import math\ny = math.sin(90)")
    auditor.visit(tree)
    assert len(auditor.risks) == 1
    assert auditor.risks[0]["type"] == "Unit Ambiguity"


def test_safe_trig():
    auditor = ScienceAuditor()
    tree = ast.parse("import math\ny = math.sin(x)")
    auditor.visit(tree)
    assert len(auditor.risks) == 0


def test_cos_flagged():
    auditor = ScienceAuditor()
    tree = ast.parse("import math\ny = math.cos(3.14)")
    auditor.visit(tree)
    assert any(r["type"] == "Unit Ambiguity" for r in auditor.risks)


def test_tan_flagged():
    auditor = ScienceAuditor()
    tree = ast.parse("import math\ny = math.tan(0.785)")
    auditor.visit(tree)
    assert any(r["type"] == "Unit Ambiguity" for r in auditor.risks)


def test_multiple_risks_detected():
    auditor = ScienceAuditor()
    tree = ast.parse("import math\nx = a / b\ny = math.sin(1.57)")
    auditor.visit(tree)
    types = {r["type"] for r in auditor.risks}
    assert "Singularity Risk" in types
    assert "Unit Ambiguity" in types


# ─── audit_file ────────────────────────────────────────────────


def test_audit_file(tmp_path):
    f = tmp_path / "script.py"
    f.write_text("x = 10 / y\ny = math.sin(90)", encoding="utf-8")
    risks = audit_file(f)
    assert len(risks) == 2
    types = [r["type"] for r in risks]
    assert "Singularity Risk" in types
    assert "Unit Ambiguity" in types


def test_audit_file_clean(tmp_path):
    f = tmp_path / "clean.py"
    f.write_text('def add(x, y):\n    """Sums x and y."""\n    return x + y\n')
    assert audit_file(f) == []


def test_audit_file_syntax_error_graceful(tmp_path):
    f = tmp_path / "broken.py"
    f.write_text("def :")
    assert audit_file(f) == []


def test_audit_file_contract_missing_file(tmp_path):
    with pytest.raises(PreconditionError):
        audit_file(tmp_path / "ghost.py")


def test_audit_file_contract_not_python(tmp_path):
    f = tmp_path / "data.txt"
    f.write_text("not python")
    with pytest.raises(PreconditionError):
        audit_file(f)


def test_audit_file_contract_directory_passed(tmp_path):
    with pytest.raises(PreconditionError):
        audit_file(tmp_path)


# ─── audit_directory ───────────────────────────────────────────


def test_audit_directory_no_test_files(tmp_path):
    (tmp_path / "safe_script.py").write_text("x = 10 / 2", encoding="utf-8")
    (tmp_path / "unsafe_script.py").write_text("y = math.sin(90)", encoding="utf-8")
    risks = audit_directory(tmp_path)
    assert len(risks) == 1
    assert risks[0]["type"] == "Unit Ambiguity"


def test_audit_directory_skips_test_files(tmp_path):
    (tmp_path / "test_utils.py").write_text("y = a / b\n")
    risks = audit_directory(tmp_path)
    assert risks == []


def test_audit_directory_attaches_file_key(tmp_path):
    (tmp_path / "model.py").write_text("x = a / b\n")
    risks = audit_directory(tmp_path)
    assert "file" in risks[0]


def test_audit_directory_multiple_files(tmp_path):
    (tmp_path / "a.py").write_text("x = a / b\n")
    (tmp_path / "b.py").write_text("x = c / d\n")
    risks = audit_directory(tmp_path)
    assert len(risks) == 2


def test_audit_directory_contract_missing():
    with pytest.raises(PreconditionError):
        audit_directory(Path("/nonexistent/xyz"))


def test_audit_directory_contract_file_passed(tmp_path):
    f = tmp_path / "file.py"
    f.write_text("x = 1")
    with pytest.raises(PreconditionError):
        audit_directory(f)
