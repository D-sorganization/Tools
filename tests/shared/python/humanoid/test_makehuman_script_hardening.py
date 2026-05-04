from __future__ import annotations

import math
import ast
from pathlib import Path

import pytest

from humanoid_character_builder.generators._makehuman_generator import (
    MakeHumanMeshGenerator,
)


def test_makehuman_script_serializes_quoted_output_path(tmp_path: Path) -> None:
    generator = MakeHumanMeshGenerator(makehuman_path=tmp_path)
    output_dir = tmp_path / "quoted'path"

    script = generator._create_makehuman_script(
        {"macrodetails/Gender": 0.5},
        output_dir,
    )

    compile(script, "<makehuman_script>", "exec")
    export_line = next(line for line in script.splitlines() if "export_path =" in line)
    assert ast.literal_eval(export_line.split("=", 1)[1].strip()) == str(
        (output_dir / "humanoid.obj").resolve()
    )
    assert f'export_path = "{output_dir}/humanoid.obj"' not in script


def test_makehuman_script_rejects_invalid_modifier_key(tmp_path: Path) -> None:
    generator = MakeHumanMeshGenerator(makehuman_path=tmp_path)

    with pytest.raises(ValueError, match="Invalid MakeHuman modifier key"):
        generator._create_makehuman_script({"bad key": 0.5}, tmp_path)


def test_makehuman_script_rejects_nonfinite_modifier_value(tmp_path: Path) -> None:
    generator = MakeHumanMeshGenerator(makehuman_path=tmp_path)

    with pytest.raises(ValueError, match="Invalid MakeHuman modifier value"):
        generator._create_makehuman_script({"macrodetails/Gender": math.nan}, tmp_path)
