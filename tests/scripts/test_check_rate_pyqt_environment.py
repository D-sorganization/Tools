"""Contracts for the isolated trusted PyQt runtime smoke check."""

import json
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
        "numpy==2.3.5\nscipy==1.17.1\nPyQt6==6.11.0\nmatplotlib==3.11.1\npytest==9.1.1\n",
    )

    assert environment_check.read_expected_versions(constraints) == {
        "numpy": "2.3.5",
        "scipy": "1.17.1",
        "pyqt6": "6.11.0",
        "matplotlib": "3.11.1",
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
        "numpy==2.3.5\nscipy==1.17.1\nPyQt6==6.11.0\nmatplotlib==3.11.1\n",
    )
    imported = False

    def fake_version(distribution: str) -> str:
        return (
            "2.5.2"
            if distribution == "numpy"
            else {
                "scipy": "1.17.1",
                "PyQt6": "6.11.0",
                "matplotlib": "3.11.1",
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
        "numpy==2.3.5\nscipy==1.17.1\nPyQt6==6.11.0\nmatplotlib==3.11.1\n",
    )
    expected = {
        "numpy": "2.3.5",
        "scipy": "1.17.1",
        "PyQt6": "6.11.0",
        "matplotlib": "3.11.1",
    }
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
        "matplotlib": "3.11.1",
    }
    assert imported is True


def _font_expectations(path: Path, values: dict[str, str | list[str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=1), encoding="utf-8")
    return path


def test_font_stack_change_fails_with_named_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#4844: a host font upgrade must fail as a named environment change.

    The whole-window glyph drift on every PyQt tab was caused by
    libfreetype6 2.13.2 -> 2.14.2 and libfontconfig1 2.15.0 -> 2.17.1 on the
    trusted runner host; the check must name the changed package instead of
    letting it surface as opaque pixel drift.
    """

    expectations = _font_expectations(
        tmp_path / "font_stack.json",
        {
            "libfreetype6": "2.13.2+dfsg-1ubuntu0.1",
            "libfontconfig1": "2.15.0-1.1ubuntu2",
        },
    )
    monkeypatch.setattr(
        environment_check,
        "probe_font_stack",
        lambda: {
            "matplotlib_freetype": "2.13.2",
            "libfreetype6": "2.14.2+dfsg-1ubuntu0.1",
            "libfontconfig1": "2.15.0-1.1ubuntu2",
        },
    )

    with pytest.raises(RuntimeError, match="libfreetype6"):
        environment_check.verify_font_stack(expectations)


def test_font_stack_match_returns_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expectations = _font_expectations(
        tmp_path / "font_stack.json", {"libfreetype6": "2.14.2+dfsg-1ubuntu0.1"}
    )
    probe = {
        "matplotlib_freetype": "2.13.2",
        "libfreetype6": "2.14.2+dfsg-1ubuntu0.1",
        "libfontconfig1": "2.17.1-3ubuntu1",
    }
    monkeypatch.setattr(environment_check, "probe_font_stack", lambda: probe)

    assert environment_check.verify_font_stack(expectations) == probe


def test_font_stack_match_with_allowed_version_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expectations = _font_expectations(
        tmp_path / "font_stack.json",
        {
            "libfontconfig1": ["2.15.0-1.1ubuntu2", "2.17.1-3ubuntu1"],
            "libfreetype6": ["2.13.2+dfsg-1ubuntu0.1", "2.14.2+dfsg-1ubuntu0.1"],
        },
    )
    probe_control_tower = {
        "matplotlib_freetype": "2.13.2",
        "libfontconfig1": "2.15.0-1.1ubuntu2",
        "libfreetype6": "2.13.2+dfsg-1ubuntu0.1",
    }
    monkeypatch.setattr(
        environment_check, "probe_font_stack", lambda: probe_control_tower
    )
    assert environment_check.verify_font_stack(expectations) == probe_control_tower

    probe_oglaptop = {
        "matplotlib_freetype": "2.13.2",
        "libfontconfig1": "2.17.1-3ubuntu1",
        "libfreetype6": "2.14.2+dfsg-1ubuntu0.1",
    }
    monkeypatch.setattr(environment_check, "probe_font_stack", lambda: probe_oglaptop)
    assert environment_check.verify_font_stack(expectations) == probe_oglaptop

    probe_unsupported = {
        "matplotlib_freetype": "2.13.2",
        "libfontconfig1": "2.18.0-1",
        "libfreetype6": "2.14.2+dfsg-1ubuntu0.1",
    }
    monkeypatch.setattr(
        environment_check, "probe_font_stack", lambda: probe_unsupported
    )
    with pytest.raises(RuntimeError, match="libfontconfig1 2.18.0-1 not in"):
        environment_check.verify_font_stack(expectations)


def test_font_stack_probe_reports_host_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCompleted:
        returncode = 0
        stdout = "2.14.2+dfsg-1ubuntu0.1\n"
        stderr = ""

    monkeypatch.setattr(
        environment_check.subprocess,
        "run",
        lambda command, **kwargs: FakeCompleted(),
    )
    monkeypatch.setattr(
        environment_check,
        "_MATPLOTLIB_FREETYPE_VERSION",
        lambda: "2.13.2",
    )

    probe = environment_check.probe_font_stack()

    assert probe["matplotlib_freetype"] == "2.13.2"
    assert probe["libfreetype6"] == "2.14.2+dfsg-1ubuntu0.1"
    assert probe["libfontconfig1"] == "2.14.2+dfsg-1ubuntu0.1"
