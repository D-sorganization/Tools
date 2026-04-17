"""Tests for model_generation.library.repository."""

from __future__ import annotations

import importlib.util
import io
import json
import logging
import shutil
import sys
import zipfile
from pathlib import Path

import pytest


class TestGitHubRepositoryArchiveExtraction:
    """Regression tests for GitHub archive extraction."""

    def _load_repository_module(self):
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
        return repository_module

    def test_download_archive_rejects_zip_slip(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        repository_module = self._load_repository_module()

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

    def test_download_archive_rejects_non_https_urls(self, tmp_path: Path) -> None:
        repository_module = self._load_repository_module()

        with pytest.raises(ValueError, match="absolute HTTPS"):
            repository_module._urlretrieve_https("file:///tmp/model.zip", tmp_path)


class TestRepositoryDownload:
    """Regression tests for repository download error handling."""

    def _load_repository_module(self):
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
        return repository_module

    def test_local_repository_download_model_missing_file(self, tmp_path: Path) -> None:
        repository_module = self._load_repository_module()

        repo = repository_module.LocalRepository(path=tmp_path, name="local")
        with pytest.raises(FileNotFoundError):
            repo.download_model("does-not-exist.urdf", tmp_path / "out")

    def test_local_repository_download_model_copies_model(self, tmp_path: Path) -> None:
        repository_module = self._load_repository_module()
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        model_path = source_dir / "robot.urdf"
        model_path.write_text("<robot name='robot'></robot>")
        (source_dir / "meshes").mkdir()
        (source_dir / "meshes" / "mesh.obj").write_text("v 0 0 0")

        repo = repository_module.LocalRepository(path=source_dir, name="local")
        out_dir = tmp_path / "out"
        downloaded = repo.download_model("robot.urdf", out_dir)

        assert downloaded == out_dir / "robot.urdf"
        assert downloaded.exists()
        assert (out_dir / "meshes" / "mesh.obj").exists()

    def test_github_repository_download_model_keeps_partial_mesh_failures(
        self, tmp_path: Path, monkeypatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        repository_module = self._load_repository_module()

        mesh_listing = [
            {
                "type": "file",
                "name": "good.obj",
                "path": "robots/meshes/good.obj",
                "download_url": "https://example.com/good.obj",
            },
            {
                "type": "file",
                "name": "bad.obj",
                "path": "robots/meshes/bad.obj",
                "download_url": "https://example.com/bad.obj",
            },
        ]
        mesh_list_payload = json.dumps(mesh_listing)

        class FakeResponse:
            def __init__(self, payload: str):
                self._body = io.BytesIO(payload.encode("utf-8"))

            def read(self) -> bytes:
                return self._body.read()

            def __enter__(self) -> FakeResponse:
                return self

            def __exit__(self, *_: object) -> bool:
                return False

        def fake_urlopen(request, timeout: float) -> FakeResponse:
            return FakeResponse(mesh_list_payload)

        def fake_urlretrieve(url: str, filename: str | Path) -> tuple[str, None]:
            if str(url).endswith(".urdf"):
                Path(filename).write_text("<robot name='robot'></robot>")
            elif "bad.obj" in str(url):
                raise OSError("boom")
            else:
                Path(filename).write_text("mesh")
            return str(filename), None

        monkeypatch.setattr(repository_module, "_urlopen_https", fake_urlopen)
        monkeypatch.setattr(repository_module, "_urlretrieve_https", fake_urlretrieve)

        repo = repository_module.GitHubRepository(
            owner="owner", repo="repo", branch="main"
        )
        out_dir = tmp_path / "out"
        caplog.set_level(logging.ERROR, logger=repository_module.__name__)

        out = repo.download_model("robots/model.urdf", out_dir)

        assert out == out_dir / "model.urdf"
        assert (out_dir / "meshes" / "good.obj").exists()
        assert not (out_dir / "meshes" / "bad.obj").exists()
        assert any(
            record.exc_info and "Failed to download mesh" in record.message
            for record in caplog.records
        )

    def test_download_archive_deletes_temp_file(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        repository_module = self._load_repository_module()

        archive_path = tmp_path / "archive.zip"
        with zipfile.ZipFile(archive_path, "w") as zf:
            zf.writestr("repo-main/robot.urdf", "<robot name='safe' />")

        seen_tmp_files: list[Path] = []

        def fake_urlretrieve(url: str, filename: Path) -> tuple[str, None]:
            seen_tmp_files.append(Path(filename))
            Path(filename).write_bytes(archive_path.read_bytes())
            return str(filename), None

        repo = repository_module.GitHubRepository(
            owner="owner", repo="repo", branch="main"
        )
        destination = tmp_path / "extract"

        monkeypatch.setattr(repository_module, "_urlretrieve_https", fake_urlretrieve)

        assert repo.download_archive(destination) is True

        assert not any(p.exists() for p in seen_tmp_files)
