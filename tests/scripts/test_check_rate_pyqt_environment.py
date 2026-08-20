"""Contracts for the isolated trusted PyQt runtime smoke check."""

from pathlib import Path

import pytest

from scripts import check_rate_pyqt_environment as environment_check

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _write_constraints(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_reads_exact_required_binary_stack_versions(tmp_path: Path) -> None:
    constraints = _write_constraints(
        tmp_path / "constraints.txt",
        "numpy==2.3.5\nscipy==1.17.1\nPyQt6==6.11.0\npytest==9.1.1\n",
    )

    assert environment_check.read_expected_versions(constraints) == {
        "numpy": "2.3.5",
        "scipy": "1.17.1",
        "pyqt6": "6.11.0",
    }


def test_rate_pyqt_binary_stack_matches_repository_lock() -> None:
    assert environment_check.read_expected_versions(
        REPOSITORY_ROOT / "requirements-rate-pyqt.txt"
    ) == environment_check.read_expected_versions(
        REPOSITORY_ROOT / "requirements-lock.txt"
    )


@pytest.mark.parametrize(
    "body, message",
    [
        ("numpy>=2\nscipy==1.17.1\nPyQt6==6.11.0\n", "exactly pinned"),
        ("numpy==2.3.5\nPyQt6==6.11.0\n", "missing required"),
        (
            "numpy==2.3.5\nNUMPY==2.3.5\nscipy==1.17.1\nPyQt6==6.11.0\n",
            "duplicate",
        ),
    ],
)
def test_rejects_ambiguous_binary_stack_constraints(
    tmp_path: Path, body: str, message: str
) -> None:
    constraints = _write_constraints(tmp_path / "constraints.txt", body)

    with pytest.raises(ValueError, match=message):
        environment_check.read_expected_versions(constraints)


def test_runtime_check_fails_before_import_when_version_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    constraints = _write_constraints(
        tmp_path / "constraints.txt",
        "numpy==2.3.5\nscipy==1.17.1\nPyQt6==6.11.0\n",
    )
    imported = False

    def fake_version(distribution: str) -> str:
        return (
            "2.5.2"
            if distribution == "numpy"
            else {
                "scipy": "1.17.1",
                "PyQt6": "6.11.0",
            }[distribution]
        )

    def fake_import_runtime() -> None:
        nonlocal imported
        imported = True

    monkeypatch.setattr(environment_check, "metadata_version", fake_version)
    monkeypatch.setattr(environment_check, "_import_runtime", fake_import_runtime)

    with pytest.raises(RuntimeError, match="numpy 2.5.2 != constrained 2.3.5"):
        environment_check.verify_runtime(constraints)
    assert imported is False


def test_runtime_check_imports_after_all_versions_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    constraints = _write_constraints(
        tmp_path / "constraints.txt",
        "numpy==2.3.5\nscipy==1.17.1\nPyQt6==6.11.0\n",
    )
    expected = {"numpy": "2.3.5", "scipy": "1.17.1", "PyQt6": "6.11.0"}
    imported = False

    def fake_import_runtime() -> None:
        nonlocal imported
        imported = True

    monkeypatch.setattr(environment_check, "metadata_version", expected.__getitem__)
    monkeypatch.setattr(environment_check, "_import_runtime", fake_import_runtime)

    assert environment_check.verify_runtime(constraints) == {
        "numpy": "2.3.5",
        "scipy": "1.17.1",
        "pyqt6": "6.11.0",
    }
    assert imported is True
