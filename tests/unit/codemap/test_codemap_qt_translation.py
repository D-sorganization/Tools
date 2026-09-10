"""Qt translation catalogs share .ts with TypeScript but are XML resources."""

from pathlib import Path

import pytest

from shared.python.codemap import api, indexer, parsers


@pytest.mark.parametrize("suffix", [".ts", ".TS"])
def test_valid_qt_translation_is_a_resource_without_code_edges(suffix: str) -> None:
    source = (
        '<?xml version="1.0"?><!DOCTYPE TS><TS version="2.1">'
        "<context><name>UI</name></context></TS>"
    )
    result = parsers.dispatch(f"locale/de{suffix}", source)
    assert result is not None
    assert result.complete
    assert result.language == "qt-translation"
    assert result.symbols == []
    assert result.imports == []


@pytest.mark.parametrize(
    "source",
    [
        '<?xml version="1.0"?><TS>',
        "const value: = ;",
        '<!DOCTYPE TS [<!ENTITY text "untrusted">]><TS>&text;</TS>',
    ],
)
def test_malformed_xml_or_typescript_still_fails(source: str) -> None:
    pytest.importorskip("tree_sitter_typescript")
    result = parsers.dispatch("invalid.ts", source)
    assert result is not None
    assert not result.complete


def test_translation_change_to_code_requires_rebuild(tmp_path: Path) -> None:
    pytest.importorskip("tree_sitter_typescript")
    source = tmp_path / "resource.ts"
    source.write_text('<?xml version="1.0"?><TS version="2.1"/>', encoding="utf-8")
    result = indexer.rebuild(tmp_path)
    assert result.errors == []
    assert api.repo_summary(repo_root=tmp_path).languages == {"qt-translation": 1}
    source.write_text("export function target() { return 1; }", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source content"):
        api.search_code("target", repo_root=tmp_path)
    assert indexer.rebuild(tmp_path).errors == []
    assert api.get_symbol("target", repo_root=tmp_path) is not None
