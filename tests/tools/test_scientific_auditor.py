"""Tests for scientific_auditor, enforcing TDD."""

import ast

from src.tools.scientific_auditor import ScienceAuditor, audit_directory, audit_file


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


def test_audit_file(tmp_path):
    f = tmp_path / "script.py"
    f.write_text("x = 10 / y\ny = math.sin(90)", encoding="utf-8")

    risks = audit_file(f)
    assert len(risks) == 2
    types = [r["type"] for r in risks]
    assert "Singularity Risk" in types
    assert "Unit Ambiguity" in types


def test_audit_directory_no_test_files(tmp_path):
    f1 = tmp_path / "safe_script.py"
    f1.write_text("x = 10 / 2", encoding="utf-8")

    f2 = tmp_path / "unsafe_script.py"
    f2.write_text("y = math.sin(90)", encoding="utf-8")

    risks = audit_directory(tmp_path)
    assert len(risks) == 1
    assert risks[0]["type"] == "Unit Ambiguity"
