from __future__ import annotations

from pathlib import Path

import scripts.check_minimum_test_contract as contract


def test_has_tests_accepts_package_named_tests_directory(
    monkeypatch, tmp_path: Path
) -> None:
    (tmp_path / "src" / "plant_simulator").mkdir(parents=True)
    tests_dir = tmp_path / "tests" / "plant_simulator"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_dataset.py").write_text("def test_dataset():\n    pass\n")

    monkeypatch.setattr(contract, "ROOT", tmp_path)

    assert contract.has_tests("src/plant_simulator")
