"""Tests for model_generation.library.repository."""

from __future__ import annotations

import importlib.util
import shutil
import sys
import zipfile
from pathlib import Path


class TestGitHubRepositoryArchiveExtraction:
    """Regression tests for GitHub archive extraction."""

    def test_download_archive_rejects_zip_slip(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        repository_path = (
            Path(__file__).resolve().parents[4]
            / "src/shared/python/model_generation/library/repository.py"
        )
        spec = importlib.util.spec_from_file_location(
            "repository_under_test",
            repository_path,
        )
        assert spec is not None
        assert spec.loader is not None
        repository_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = repository_module
        spec.loader.exec_module(repository_module)

        archive_path = tmp_path / "malicious.zip"
        with zipfile.ZipFile(archive_path, "w") as zf:
            zf.writestr("repo-main/robot.urdf", "<robot name='safe' />")
            zf.writestr("../escape.txt", "owned")

        def fake_urlretrieve(url: str, filename: str):
            shutil.copy2(archive_path, filename)
            return filename, None

        monkeypatch.setattr(
            repository_module.urllib.request, "urlretrieve", fake_urlretrieve
        )

        repo = repository_module.GitHubRepository(
            owner="owner", repo="repo", branch="main"
        )
        destination = tmp_path / "extract"

        assert repo.download_archive(destination) is False
        assert not any(destination.rglob("*"))
        assert not (tmp_path / "escape.txt").exists()
