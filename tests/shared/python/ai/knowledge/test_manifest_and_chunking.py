"""Manifest contract and heading chunking for the knowledge-pack engine (#5345)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from shared.python.ai.knowledge import (
    ManifestError,
    PackManifest,
    chunk_document,
    load_manifest,
    manifest_from_dict,
)

from .conftest import ARTICLE, MANIFEST, PAPER


def test_manifest_round_trips_the_fixture(manifest: PackManifest) -> None:
    assert manifest.id == "findings"
    assert [s.authority for s in manifest.sources] == [
        "published",
        "findings",
        "reviews",
    ]
    assert manifest.sources[0].exclude == ("**/*-bibliography.md",)
    assert manifest.status_overrides == {"docs/assessments/retracted.md": "retracted"}


def test_manifest_loads_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "pack.yml"
    path.write_text(
        "id: product\ntitle: Product\nsources:\n"
        "  - repo: Tools\n    authority: product\n    include: ['docs/**/*.md']\n",
        "utf-8",
    )
    loaded = load_manifest(path)
    assert loaded.id == "product"
    assert loaded.chunk_chars > 0
    assert loaded.sources[0].exclude == ()


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (lambda m: m.pop("id"), "id"),
        (lambda m: m.update(id="Bad Id"), "id"),
        (lambda m: m.update(sources=[]), "sources"),
        (lambda m: m["sources"][0].update(authority="rumour"), "authority"),
        (lambda m: m["sources"][0].update(include=[]), "include"),
        (lambda m: m["sources"][0].update(repo="../escape"), "repo"),
        (lambda m: m.update(chunk_chars=10), "chunk_chars"),
        (lambda m: m.update(status_overrides={"x.md": "maybe"}), "status"),
        (lambda m: m.update(surprise=True), "unknown"),
    ],
)
def test_invalid_manifests_are_rejected(mutate, needle: str) -> None:
    raw = copy.deepcopy(MANIFEST)
    mutate(raw)
    with pytest.raises(ManifestError, match=needle):
        manifest_from_dict(raw)


def test_markdown_chunks_by_heading_and_ignores_code_fences() -> None:
    chunks = chunk_document(ARTICLE, suffix=".qmd", max_chars=2000)
    assert [c.title for c in chunks] == ["Energy Transfer", "Timing", "Shaft Flex"]
    assert [c.anchor for c in chunks] == [
        "energy-transfer",
        "energy-transfer/timing",
        "energy-transfer/shaft-flex",
    ]
    assert "not a heading" in chunks[1].text
    assert "title: Energy Transfer" not in "".join(c.text for c in chunks)


def test_front_matter_status_is_reported() -> None:
    chunks = chunk_document(
        "---\nstatus: superseded\n---\n# A\n\nBody.\n", suffix=".md", max_chars=2000
    )
    assert chunks[0].status == "superseded"
    plain = chunk_document("# A\n\nBody.\n", suffix=".md", max_chars=2000)
    assert plain[0].status is None


def test_latex_chunks_by_section() -> None:
    chunks = chunk_document(PAPER, suffix=".tex", max_chars=2000)
    assert [c.title for c in chunks] == ["Methods", "Filtering"]
    assert chunks[1].anchor == "methods/filtering"
    assert "Butterworth" in chunks[1].text


def test_long_sections_split_at_paragraphs_without_losing_text() -> None:
    body = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(10))
    chunks = chunk_document(f"# Long\n\n{body}\n", suffix=".md", max_chars=500)
    assert len(chunks) > 1
    assert all(len(c.text) <= 500 for c in chunks)
    assert all(c.anchor == "long" for c in chunks)
    joined = " ".join(c.text for c in chunks)
    for i in range(10):
        assert f"Paragraph {i} " in joined


def test_text_before_any_heading_is_kept() -> None:
    chunks = chunk_document(
        "Preamble line.\n\n# H\n\nBody.\n", suffix=".md", max_chars=2000
    )
    assert chunks[0].title == ""
    assert "Preamble" in chunks[0].text
