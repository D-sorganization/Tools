from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest
from humanoid_character_builder.generators._makehuman_generator import (
    MakeHumanMeshGenerator,
)

_SHELL_METACHAR_PATHS = [
    "out;rm -rf /",
    "out&bad",
    "out|evil",
    "out`cmd`",
    "out$(cmd)",
    "out\x00null",
]


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


@pytest.mark.parametrize("suffix", _SHELL_METACHAR_PATHS)
def test_makehuman_script_shell_metacharacters_are_safely_serialized(
    tmp_path: Path, suffix: str
) -> None:
    """Paths containing shell metacharacters must either be safely serialized
    in the generated script (so the metacharacters cannot cause shell injection
    when the script file is later executed by MakeHuman's Python interpreter)
    or be rejected outright.

    The generated script must remain syntactically valid Python regardless of
    what characters appear in the path, and the embedded path value must
    round-trip correctly via ``ast.literal_eval``.
    """
    generator = MakeHumanMeshGenerator(makehuman_path=tmp_path)
    output_dir = tmp_path / suffix
    modifiers = {"macrodetails/Gender": 0.5}

    try:
        script = generator._create_makehuman_script(modifiers, output_dir)
    except ValueError:
        # Rejection is an acceptable defence — the path never reaches the script.
        return

    # If the path was accepted it must be safely serialized: the generated
    # Python must parse without errors and the exported path must survive a
    # literal_eval round-trip (no injected tokens).
    compile(script, "<makehuman_script>", "exec")
    export_line = next(line for line in script.splitlines() if "export_path =" in line)
    rhs = export_line.split("=", 1)[1].strip()
    recovered = ast.literal_eval(rhs)
    expected = str((output_dir / "humanoid.obj").resolve())
    assert recovered == expected, (
        f"Path with metacharacters was not faithfully round-tripped.\n"
        f"  suffix={suffix!r}\n"
        f"  expected={expected!r}\n"
        f"  got={recovered!r}"
    )


def test_makehuman_validate_output_path_within_base_rejects_traversal(
    tmp_path: Path,
) -> None:
    """A path that escapes the base directory via ``..`` must be rejected."""
    base = tmp_path / "output"
    base.mkdir()
    escaped = base / ".." / ".." / "etc" / "passwd"

    with pytest.raises(ValueError, match="escapes the expected base directory"):
        MakeHumanMeshGenerator._validate_output_path_within_base(escaped, base)


def test_makehuman_validate_output_path_within_base_accepts_child(
    tmp_path: Path,
) -> None:
    """A path that is genuinely inside the base directory must not raise."""
    base = tmp_path / "output"
    child = base / "visual" / "humanoid.obj"
    # Should not raise
    MakeHumanMeshGenerator._validate_output_path_within_base(child, base)


def test_makehuman_script_rejects_output_dir_escaping_base(tmp_path: Path) -> None:
    """_create_makehuman_script must raise when output_dir escapes base_output_dir."""
    generator = MakeHumanMeshGenerator(makehuman_path=tmp_path)
    base = tmp_path / "allowed"
    escaped_output = tmp_path / "not_allowed" / "visual"

    with pytest.raises(ValueError, match="escapes the expected base directory"):
        generator._create_makehuman_script(
            {"macrodetails/Gender": 0.5},
            escaped_output,
            base_output_dir=base,
        )
