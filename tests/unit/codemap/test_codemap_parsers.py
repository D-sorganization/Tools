from __future__ import annotations

from pathlib import Path

import pytest
from codemap._ts_common import ParsedSymbol, ParseResult

from codemap import parsers


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("tool.py", "python"),
        ("types.PYI", "python"),
        ("widget.js", "javascript"),
        ("module.MJS", "javascript"),
        ("legacy.cjs", "javascript"),
        ("component.jsx", "javascript"),
        ("library.ts", "typescript"),
        ("view.TSX", "tsx"),
        ("lib.rs", "rust"),
        ("README.md", "markdown"),
        ("guide.MARKDOWN", "markdown"),
    ],
)
def test_language_for_maps_supported_suffixes_case_insensitively(
    path: str,
    expected: str,
) -> None:
    assert parsers.language_for(path) == expected
    assert parsers.language_for(Path(path)) == expected


@pytest.mark.parametrize("path", ["Makefile", "notes.txt", ".gitignore"])
def test_language_for_returns_none_for_unsupported_paths(path: str) -> None:
    assert parsers.language_for(path) is None
    assert parsers.dispatch(path, "ignored") is None


@pytest.mark.parametrize(
    ("path", "expected_language"),
    [
        ("tool.py", "python"),
        ("types.pyi", "python"),
        ("widget.js", "javascript"),
        ("module.mjs", "javascript"),
        ("legacy.cjs", "javascript"),
        ("component.jsx", "javascript"),
        ("library.ts", "typescript"),
        ("view.tsx", "tsx"),
        ("lib.rs", "rust"),
        ("README.md", "markdown"),
        ("guide.markdown", "markdown"),
    ],
)
def test_dispatch_routes_supported_extensions_to_registered_extractors(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    expected_language: str,
) -> None:
    source = b"source bytes"
    observed: list[tuple[str, bytes]] = []

    def extractor(path_arg: str, source_arg: str | bytes) -> ParseResult:
        assert isinstance(source_arg, bytes)
        observed.append((path_arg, source_arg))
        return ParseResult(
            expected_language,
            imports=[f"{expected_language}:import"],
            symbols=[],
        )

    monkeypatch.setitem(parsers._EXTRACTORS, expected_language, extractor)

    result = parsers.dispatch(Path(path), source)

    assert result == ParseResult(expected_language, [f"{expected_language}:import"], [])
    assert observed == [(str(Path(path)), source)]


def test_dispatch_returns_none_when_language_has_no_extractor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(parsers._EXTRACTORS, "python")

    assert parsers.dispatch("tool.py", "print('missing extractor')") is None


def test_public_reexports_and_extractor_registry_are_stable() -> None:
    assert parsers.ParseResult is ParseResult
    assert parsers.ParsedSymbol is ParsedSymbol
    assert parsers._EXTRACTORS["python"] is parsers.extract_python
    assert parsers._EXTRACTORS["javascript"] is parsers.extract_javascript
    assert parsers._EXTRACTORS["typescript"] is parsers.extract_typescript
    assert parsers._EXTRACTORS["tsx"] is parsers.extract_typescript
    assert parsers._EXTRACTORS["rust"] is parsers.extract_rust
    assert parsers._EXTRACTORS["markdown"] is parsers.extract_markdown
