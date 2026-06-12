from __future__ import annotations

import ast
import json
import sys
import types
from pathlib import Path

TOOLS_ROOT = Path(__file__).resolve().parents[3]
DATA_PROCESSOR_ROOT = (
    TOOLS_ROOT / "src" / "data_processing" / "data_processor" / "python"
)
sys.path.insert(0, str(DATA_PROCESSOR_ROOT))

from data_processor.core.script_generator import ProcessingPipeline, ScriptGenerator


def test_batch_script_contains_failure_and_atomic_write_guards() -> None:
    script = ScriptGenerator().generate_batch_script(
        ProcessingPipeline(name="Batch Test"),
        input_patterns=["data/*.csv"],
        output_dir="output",
        parallel=True,
    )

    assert "raise SystemExit(main())" in script
    assert "return 1" in script
    assert "os.replace(temp_path, output_path)" in script
    assert "DATA_PROCESSOR_BATCH_MAX_WORKERS" in script


def test_batch_script_serializes_quoted_paths_and_returns_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_dir = tmp_path / "in'put"
    output_dir = tmp_path / "out'put"
    input_dir.mkdir()
    (input_dir / "bad.csv").mkdir()

    script = ScriptGenerator().generate_batch_script(
        ProcessingPipeline(name="Batch Test"),
        input_patterns=[str(input_dir / "*")],
        output_dir=str(output_dir),
        parallel=False,
    )

    monkeypatch.setitem(
        sys.modules,
        "data_processor.core.signal_processing",
        types.ModuleType("signal_processing"),
    )
    monkeypatch.setitem(
        sys.modules,
        "data_processor.vectorized_filter_engine",
        types.ModuleType("vectorized_filter_engine"),
    )
    namespace: dict[str, object] = {}
    exec(compile(script, "<generated_batch>", "exec"), namespace)  # nosec B102

    assert namespace["main"]() == 1
    assert output_dir.exists()


def test_batch_script_path_metacharacters_are_safely_serialized() -> None:
    """Paths with metacharacters must be embedded as JSON string literals.

    Using repr() can produce ambiguous escape sequences for paths that contain
    backslashes, single-quotes, or backslash-n sequences.  json.dumps() always
    produces a valid, unambiguous JSON string whose value round-trips exactly.
    """
    dangerous_path = r"C:\users\n'evil'\data\*.csv"
    script = ScriptGenerator().generate_batch_script(
        ProcessingPipeline(name="Injection Test"),
        input_patterns=[dangerous_path],
        output_dir=r"C:\out\path\with'quote",
        parallel=False,
    )

    # The generated source must be syntactically valid Python
    tree = ast.parse(script)

    # Locate the assignment `input_patterns = ...` inside main() and verify
    # the literal round-trips to the original string.
    found = False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "input_patterns"
        ):
            value = ast.literal_eval(node.value)
            assert value == [dangerous_path], (
                f"Path did not round-trip safely: got {value!r}"
            )
            found = True
            break
    assert found, "input_patterns assignment not found in generated script"

    # Confirm the script does NOT contain a raw repr()-style escape that would
    # silently mangle \\n into a newline or introduce an unmatched quote.
    # json.dumps encodes backslashes as \\ and single-quotes unescaped.
    assert json.dumps([dangerous_path]) in script


def test_batch_script_failure_surfaces_as_nonzero_exit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A file that fails to process must produce exit code 1.

    Verifies that the generated batch script accumulates per-file failures into
    the `failures` list and returns a non-zero exit code rather than silently
    swallowing errors.
    """
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    # Place a directory where a file is expected — pd.read_csv will raise on it.
    bad_entry = input_dir / "bad.csv"
    bad_entry.mkdir()

    script = ScriptGenerator().generate_batch_script(
        ProcessingPipeline(name="Failure Exit Test"),
        input_patterns=[str(input_dir / "*.csv")],
        output_dir=str(output_dir),
        parallel=False,  # sequential path; avoids subprocess for test isolation
    )

    signal_mod = types.ModuleType("signal_processing")
    vfe_mod = types.ModuleType("vectorized_filter_engine")
    monkeypatch.setitem(
        sys.modules, "data_processor.core.signal_processing", signal_mod
    )
    monkeypatch.setitem(sys.modules, "data_processor.vectorized_filter_engine", vfe_mod)

    namespace: dict[str, object] = {}
    exec(compile(script, "<generated_batch_failure>", "exec"), namespace)  # nosec B102

    exit_code = namespace["main"]()
    assert exit_code == 1, f"Expected exit code 1 for failed batch, got {exit_code}"
