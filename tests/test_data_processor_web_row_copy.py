from pathlib import Path


def test_data_processor_row_copy_uses_own_properties_only() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    hook_path = (
        repo_root
        / "src"
        / "data_processing"
        / "data_processor"
        / "web"
        / "src"
        / "hooks"
        / "useDataProcessor.ts"
    )
    source = hook_path.read_text(encoding="utf-8")

    assert "function copyOwnRowProperties(row: DataRow): DataRow" in source
    assert "Object.keys(row)" in source
    assert "for (const key in row)" not in source
