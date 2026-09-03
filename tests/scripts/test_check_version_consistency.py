"""VERSION / pyproject / package.json / helm chart must agree (Tools #4910)."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

import check_version_consistency as cvc  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def _seed(root: Path, *, version: str, package_version: str | None = None) -> None:
    (root / "VERSION").write_text(version + "\n")
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "x"\nversion = "{version}"\n'
    )
    (root / "package.json").write_text(
        '{\n  "name": "ws",\n  "version": "%s",\n  "private": true\n}\n'
        % (package_version or version)
    )
    chart = root / "helm" / "tools"
    chart.mkdir(parents=True)
    (chart / "Chart.yaml").write_text(
        f'apiVersion: v2\nname: tools\nappVersion: "{version}"\n'
    )


def test_consistent_tree_passes(tmp_path: Path) -> None:
    _seed(tmp_path, version="1.15.0")
    assert cvc.mismatches(cvc.read_versions(tmp_path)) == []
    assert cvc.main(["--root", str(tmp_path)]) == 0


def test_mismatch_is_named_and_fails(tmp_path: Path, capsys) -> None:
    _seed(tmp_path, version="1.15.0", package_version="0.1.0")
    bad = cvc.mismatches(cvc.read_versions(tmp_path))
    assert bad == ["package.json (0.1.0 != 1.15.0)"]
    assert cvc.main(["--root", str(tmp_path)]) == 1
    assert "package.json" in capsys.readouterr().err


def test_set_rewrites_every_source(tmp_path: Path) -> None:
    _seed(tmp_path, version="1.15.0", package_version="0.1.0")
    touched = cvc.set_version(tmp_path, "1.16.0")
    assert touched == [
        "VERSION",
        "pyproject.toml",
        "package.json",
        "helm/tools/Chart.yaml",
    ]
    versions = cvc.read_versions(tmp_path)
    assert set(versions.values()) == {"1.16.0"}
    assert cvc.main(["--root", str(tmp_path)]) == 0
    # Other keys untouched.
    assert '"private": true' in (tmp_path / "package.json").read_text()


def test_set_rejects_non_semver(tmp_path: Path) -> None:
    _seed(tmp_path, version="1.15.0")
    assert cvc.main(["--root", str(tmp_path), "--set", "v1.16"]) == 2


def test_no_chart_is_not_an_error(tmp_path: Path) -> None:
    (tmp_path / "VERSION").write_text("2.0.0\n")
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "2.0.0"\n')
    assert cvc.mismatches(cvc.read_versions(tmp_path)) == []


def test_this_repository_is_consistent() -> None:
    versions = cvc.read_versions(REPO_ROOT)
    assert cvc.mismatches(versions) == [], versions
