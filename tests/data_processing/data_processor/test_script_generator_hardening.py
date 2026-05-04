from __future__ import annotations

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
    exec(compile(script, "<generated_batch>", "exec"), namespace)

    assert namespace["main"]() == 1
    assert output_dir.exists()
