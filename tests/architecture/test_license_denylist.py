"""Tests for the ADR-008 dependency and license deny-list check (#5417)."""

from __future__ import annotations

import importlib.metadata
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from scripts import check_license_denylist as cld

REPO_ROOT = Path(__file__).resolve().parents[2]

DENYLIST = {
    "denied_packages": ["freemocap", "skellycam"],
    "denied_license_ids": ["AGPL-3.0", "AGPL-3.0-only", "AGPL-3.0-or-later"],
    "rationale": "docs/adr/ADR-008-markerless-mocap-authority-and-licensing.md",
}

PYPROJECT = """\
[project]
name = "demo"
version = "0.1"
dependencies = ["numpy>=1.0", "foo"]

[project.optional-dependencies]
dev = ["pytest"]
"""


def make_project(
    root: Path,
    *,
    pyproject: str = PYPROJECT,
    requirements: str = "numpy==1.26.0  # pinned\n-r other.txt\n",
    lock: str = "numpy==1.26.0\n",
) -> Path:
    (root / "config").mkdir(exist_ok=True)
    (root / "config" / "license_denylist.json").write_text(
        json.dumps(DENYLIST), encoding="utf-8"
    )
    (root / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    (root / "requirements.txt").write_text(requirements, encoding="utf-8")
    (root / "requirements-lock.txt").write_text(lock, encoding="utf-8")
    (root / "requirements-rate-pyqt.txt").write_text("PyQt6>=6\n", encoding="utf-8")
    return root


class FakeDist:
    def __init__(
        self,
        name: str,
        requires: list[str] | None = None,
        license_expression: str | None = None,
        license_text: str | None = None,
        classifiers: list[str] | None = None,
    ) -> None:
        self.requires = requires
        fields: dict[str, list[str]] = {"Name": [name]}
        if license_expression:
            fields["License-Expression"] = [license_expression]
        if license_text:
            fields["License"] = [license_text]
        if classifiers:
            fields["Classifier"] = classifiers
        self._fields = fields

    @property
    def metadata(self) -> FakeDist:
        return self

    def get(self, key: str, default: Any = None) -> Any:
        values = self._fields.get(key)
        return values[0] if values else default

    def get_all(self, key: str, failobj: Any = None) -> Any:
        return self._fields.get(key, failobj)


def install(monkeypatch: pytest.MonkeyPatch, dists: dict[str, FakeDist]) -> None:
    def fake_distribution(name: str) -> FakeDist:
        try:
            return dists[name.lower()]
        except KeyError:
            raise importlib.metadata.PackageNotFoundError(name) from None

    monkeypatch.setattr(importlib.metadata, "distribution", fake_distribution)


def test_clean_project_passes(tmp_path: Path) -> None:
    make_project(tmp_path)
    assert cld.main(["--root", str(tmp_path)]) == 0


def test_denied_package_in_optional_extra_fails(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    make_project(
        tmp_path,
        pyproject=PYPROJECT.replace('["pytest"]', '["pytest", "freemocap"]'),
    )
    with caplog.at_level(logging.ERROR):
        assert cld.main(["--root", str(tmp_path)]) == 1
    assert "freemocap ->" in caplog.text


def test_normalization_catches_skellycam_in_lock(tmp_path: Path) -> None:
    make_project(tmp_path, lock="numpy==1.26.0\nSkellyCam==1.0\n")
    violations = cld.check_static(tmp_path, cld.load_denylist(tmp_path))
    assert [v.package for v in violations] == ["skellycam"]
    assert cld.main(["--root", str(tmp_path)]) == 1


def test_pep503_separators_normalize() -> None:
    assert cld.normalize_name("Skelly_Cam") == "skelly-cam"
    assert cld.normalize_name("FreeMoCap") == "freemocap"


def test_installed_transitive_agpl_expression_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    make_project(tmp_path)
    install(
        monkeypatch,
        {
            "numpy": FakeDist("numpy", requires=["bar>=1"]),
            "foo": FakeDist("foo", license_expression="MIT"),
            "bar": FakeDist("bar", license_expression="AGPL-3.0-or-later"),
        },
    )
    with caplog.at_level(logging.ERROR):
        assert cld.main(["--root", str(tmp_path), "--installed"]) == 1
    assert "bar ->" in caplog.text


def test_installed_agpl_classifier_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_project(tmp_path)
    classifier = (
        "License :: OSI Approved :: GNU Affero General Public License v3 "
        "or later (AGPLv3+)"
    )
    install(
        monkeypatch,
        {
            "numpy": FakeDist("numpy", classifiers=[classifier]),
            "foo": FakeDist("foo"),
        },
    )
    assert cld.main(["--root", str(tmp_path), "--installed"]) == 1


def test_installed_denied_package_name_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_project(tmp_path)
    install(
        monkeypatch,
        {
            "numpy": FakeDist("numpy", requires=["FreeMoCap"]),
            "foo": FakeDist("foo"),
            "freemocap": FakeDist("freemocap", license_expression="MIT"),
        },
    )
    assert cld.main(["--root", str(tmp_path), "--installed"]) == 1


def test_installed_mit_and_lgpl_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_project(tmp_path)
    install(
        monkeypatch,
        {
            "numpy": FakeDist("numpy", license_expression="MIT"),
            "foo": FakeDist("foo", license_text="LGPL-3.0"),
        },
    )
    assert cld.main(["--root", str(tmp_path), "--installed"]) == 0


def test_not_installed_warns_without_failing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    make_project(tmp_path)
    install(monkeypatch, {"numpy": FakeDist("numpy", license_expression="MIT")})
    with caplog.at_level(logging.WARNING):
        assert cld.main(["--root", str(tmp_path), "--installed"]) == 0
    assert "foo -> not installed" in caplog.text


def test_missing_config_fails_closed(tmp_path: Path) -> None:
    make_project(tmp_path)
    (tmp_path / "config" / "license_denylist.json").unlink()
    assert cld.main(["--root", str(tmp_path)]) == 1


def test_real_repository_static_check_passes() -> None:
    denylist = cld.load_denylist(REPO_ROOT)
    assert cld.check_static(REPO_ROOT, denylist) == []
