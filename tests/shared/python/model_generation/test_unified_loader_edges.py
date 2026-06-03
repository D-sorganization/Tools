from __future__ import annotations

import json
from pathlib import Path

import pytest
from model_generation.converters.urdf_parser import ParsedModel
from model_generation.library.unified_loader import (
    LoadResult,
    ModelFormat,
    UnifiedModelLoader,
    UserPreferences,
    detect_format,
)


def test_load_result_name_prefers_model_then_source_path() -> None:
    assert LoadResult(model=ParsedModel(name="parsed")).name == "parsed"
    assert LoadResult(source_path=Path("local/model.urdf")).name == "model"
    assert LoadResult().name == "unknown"


def test_user_preferences_rejects_missing_recent_model_id() -> None:
    with pytest.raises(ValueError, match="model_id must be provided"):
        UserPreferences().add_recent(None)  # type: ignore[arg-type]


def test_detect_format_returns_xml_default_when_content_cannot_be_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "model.xml"
    source.write_text("<robot name='x'/>")
    original_read_text = Path.read_text

    def guarded_read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path == source:
            raise OSError("permission denied")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)

    assert detect_format(source) == ModelFormat.MJCF


def test_loader_uses_defaults_for_corrupt_preferences(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    (tmp_path / UnifiedModelLoader._PREFS_FILENAME).write_text("{bad json")

    loader = UnifiedModelLoader(prefs_dir=tmp_path)

    assert loader.preferences.default_model_id == "mujoco_humanoid"
    assert "Failed to load preferences" in caplog.text


def test_save_preferences_logs_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    loader = UnifiedModelLoader(prefs_dir=tmp_path)

    def raise_os_error(*_args: object, **_kwargs: object) -> int:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", raise_os_error)

    loader.save_preferences()

    assert "Failed to save preferences" in caplog.text


def test_set_default_model_rejects_missing_model_id(tmp_path: Path) -> None:
    loader = UnifiedModelLoader(prefs_dir=tmp_path)

    with pytest.raises(ValueError, match="model_id must be provided"):
        loader.set_default_model(None)  # type: ignore[arg-type]


def test_manifest_cache_missing_and_corrupt_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    missing_dir = tmp_path / "missing"
    loader = UnifiedModelLoader(prefs_dir=tmp_path / "prefs")
    monkeypatch.setattr(loader, "_get_bundled_dir", lambda: missing_dir)

    assert loader.list_bundled_models() == []

    corrupt_dir = tmp_path / "corrupt"
    corrupt_dir.mkdir()
    (corrupt_dir / "manifest.json").write_text("{bad json")
    loader._bundled_manifest = None
    monkeypatch.setattr(loader, "_get_bundled_dir", lambda: corrupt_dir)

    assert loader.list_bundled_models() == []
    assert "Failed to load manifest" in caplog.text


def test_get_bundled_model_info_rejects_missing_model_id(tmp_path: Path) -> None:
    loader = UnifiedModelLoader(prefs_dir=tmp_path)

    with pytest.raises(ValueError, match="model_id must be provided"):
        loader.get_bundled_model_info(None)  # type: ignore[arg-type]


def test_load_bundled_reports_missing_model_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundled_dir = tmp_path / "bundled"
    bundled_dir.mkdir()
    (bundled_dir / "manifest.json").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "missing",
                        "name": "Missing",
                        "format": "urdf",
                        "file": "missing.urdf",
                    }
                ]
            }
        )
    )
    loader = UnifiedModelLoader(prefs_dir=tmp_path / "prefs")
    monkeypatch.setattr(loader, "_get_bundled_dir", lambda: bundled_dir)

    result = loader.load_bundled("missing")

    assert result.success is False
    assert result.source_path == bundled_dir / "missing.urdf"
    assert "Bundled model file missing" in (result.error or "")


def test_load_file_unknown_extension_falls_back_to_mjcf_after_urdf_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "model.data"
    source.write_text("not inspected by stubs")
    loader = UnifiedModelLoader(prefs_dir=tmp_path / "prefs")
    calls: list[str] = []

    def fake_load_urdf(path: Path) -> LoadResult:
        calls.append(f"urdf:{path.name}")
        return LoadResult(source_path=path, source_format=ModelFormat.URDF)

    def fake_load_mjcf(path: Path) -> LoadResult:
        calls.append(f"mjcf:{path.name}")
        return LoadResult(
            model=ParsedModel(name="mjcf"),
            source_path=path,
            source_format=ModelFormat.MJCF,
            success=True,
        )

    monkeypatch.setattr(loader, "_load_urdf", fake_load_urdf)
    monkeypatch.setattr(loader, "_load_mjcf", fake_load_mjcf)

    result = loader.load_file(source)

    assert result.success is True
    assert result.name == "mjcf"
    assert calls == ["urdf:model.data", "mjcf:model.data"]


def test_load_file_unknown_extension_returns_urdf_fallback_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "model.data"
    source.write_text("not inspected by stubs")
    loader = UnifiedModelLoader(prefs_dir=tmp_path / "prefs")

    def fake_load_urdf(path: Path) -> LoadResult:
        return LoadResult(
            model=ParsedModel(name="urdf"),
            source_path=path,
            source_format=ModelFormat.URDF,
            success=True,
        )

    monkeypatch.setattr(loader, "_load_urdf", fake_load_urdf)
    monkeypatch.setattr(
        loader,
        "_load_mjcf",
        lambda _path: pytest.fail("MJCF fallback should not run"),
    )

    assert loader.load_file(source).name == "urdf"


def test_load_file_malformed_mjcf_returns_failed_result(tmp_path: Path) -> None:
    source = tmp_path / "broken.xml"
    source.write_text("<mujoco model='x'><worldbody></mujoco>")
    loader = UnifiedModelLoader(prefs_dir=tmp_path / "prefs")

    result = loader.load_file(source)

    assert result.success is False
    assert result.source_format is ModelFormat.MJCF
    assert result.source_path == source
    assert result.error


def test_load_default_returns_failed_default_when_fallback_is_already_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = UnifiedModelLoader(prefs_dir=tmp_path)
    calls: list[str] = []

    def fake_load_bundled(model_id: str) -> LoadResult:
        calls.append(model_id)
        return LoadResult(error=f"missing {model_id}")

    monkeypatch.setattr(loader, "load_bundled", fake_load_bundled)

    result = loader.load_default()

    assert result.success is False
    assert result.error == "missing mujoco_humanoid"
    assert calls == ["mujoco_humanoid"]


def test_convert_methods_accept_inline_xml_strings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = UnifiedModelLoader(prefs_dir=tmp_path)
    seen: list[tuple[str, str]] = []

    def fake_mjcf_to_urdf(source: str) -> str:
        seen.append(("urdf", source))
        return "<robot name='converted'/>"

    def fake_urdf_to_mjcf(source: str) -> str:
        seen.append(("mjcf", source))
        return "<mujoco model='converted'/>"

    monkeypatch.setattr(loader._mjcf_converter, "mjcf_to_urdf", fake_mjcf_to_urdf)
    monkeypatch.setattr(loader._mjcf_converter, "urdf_to_mjcf", fake_urdf_to_mjcf)

    assert loader.convert_to_urdf("<mujoco model='inline'/>") == (
        "<robot name='converted'/>"
    )
    assert loader.convert_to_mjcf("<robot name='inline'/>") == (
        "<mujoco model='converted'/>"
    )
    assert seen == [
        ("urdf", "<mujoco model='inline'/>"),
        ("mjcf", "<robot name='inline'/>"),
    ]
