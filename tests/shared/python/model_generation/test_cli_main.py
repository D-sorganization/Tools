import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import model_generation.library as library_module
import pytest
from model_generation.cli import main as cli_main
from model_generation.cli.main import (
    cmd_inertia,
    cmd_library_add,
    cmd_library_list,
    create_parser,
    main,
)
from model_generation.library import ModelCategory, ModelEntry, RepositorySource


class RecordingLibrary:
    last_instance: "RecordingLibrary | None" = None

    def __init__(self) -> None:
        self.list_calls: list[dict[str, object]] = []
        self.add_calls: list[dict[str, object]] = []
        self.models = [
            ModelEntry(
                id="arm",
                name="Arm",
                category=ModelCategory.ROBOT_ARM,
                source=RepositorySource.LOCAL,
                tags=["demo", "arm"],
            )
        ]
        RecordingLibrary.last_instance = self

    def list_models(self, **kwargs: object) -> list[ModelEntry]:
        self.list_calls.append(kwargs)
        return self.models

    def add_local_model(self, **kwargs: object) -> ModelEntry:
        self.add_calls.append(kwargs)
        return ModelEntry(id="added", name="Added")


@pytest.fixture(autouse=True)
def reset_recording_library() -> None:
    RecordingLibrary.last_instance = None


def test_create_parser_wires_expected_commands() -> None:
    parser = create_parser()

    generate = parser.parse_args(["generate", "bot", "--humanoid"])
    library_list = parser.parse_args(["library", "list", "--json"])
    inertia = parser.parse_args(["inertia", "sphere", "2", "0.5"])

    assert generate.func.__name__ == "cmd_generate"
    assert library_list.func is cmd_library_list
    assert inertia.func is cmd_inertia
    assert main([]) == 0
    assert cli_main is main


def test_library_list_parses_filters_and_outputs_entry_ids(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(library_module, "ModelLibrary", RecordingLibrary)
    args = argparse.Namespace(
        category="robot_arm",
        source="local",
        search="arm",
        json=True,
        verbose=False,
    )

    with caplog.at_level(logging.INFO, logger="model_generation.cli.main"):
        assert cmd_library_list(args) == 0

    library = RecordingLibrary.last_instance
    assert library is not None
    assert library.list_calls == [
        {
            "category": ModelCategory.ROBOT_ARM,
            "source": RepositorySource.LOCAL,
            "search": "arm",
        }
    ]
    payload = json.loads(caplog.records[-1].message)
    assert payload["models"][0]["id"] == "arm"


def test_library_list_rejects_unknown_filter_without_querying_library(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(library_module, "ModelLibrary", RecordingLibrary)
    args = argparse.Namespace(
        category="spaceship",
        source=None,
        search=None,
        json=False,
        verbose=False,
    )

    with caplog.at_level(logging.ERROR, logger="model_generation.cli.main"):
        assert cmd_library_list(args) == 1

    assert "Invalid category: spaceship" in caplog.text
    assert RecordingLibrary.last_instance is not None
    assert RecordingLibrary.last_instance.list_calls == []


def test_library_add_parses_category_and_strips_tags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(library_module, "ModelLibrary", RecordingLibrary)
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("<robot name='robot'><link name='base'/></robot>")
    args = argparse.Namespace(
        input=str(urdf),
        name="Robot",
        category="vehicle",
        tags=" demo, ,wheeled ",
    )

    with caplog.at_level(logging.INFO, logger="model_generation.cli.main"):
        assert cmd_library_add(args) == 0

    library = RecordingLibrary.last_instance
    assert library is not None
    assert library.add_calls == [
        {
            "urdf_path": urdf,
            "name": "Robot",
            "category": ModelCategory.VEHICLE,
            "tags": ["demo", "wheeled"],
        }
    ]
    assert "Added model: added" in caplog.text


@pytest.mark.parametrize(
    ("shape", "dimensions", "message"),
    [
        ("box", [1.0, 2.0], "Box requires 3 dimensions"),
        ("cylinder", [1.0], "Cylinder requires 2 dimensions"),
        ("sphere", [1.0, 2.0], "Sphere requires 1 dimension"),
        ("capsule", [1.0], "Capsule requires 2 dimensions"),
    ],
)
def test_inertia_rejects_wrong_dimension_counts(
    shape: str,
    dimensions: list[float],
    message: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    args = SimpleNamespace(shape=shape, mass=2.0, dimensions=dimensions, json=False)

    with caplog.at_level(logging.ERROR, logger="model_generation.cli.main"):
        assert cmd_inertia(args) == 1

    assert message in caplog.text
