"""Build, search, freshness and CLI contracts of the knowledge-pack engine (#5345)."""

from __future__ import annotations

import io
import sqlite3
import sys
from pathlib import Path

import pytest

from shared.python.ai.knowledge import (
    KnowledgePack,
    PackFormatError,
    PackManifest,
    Passage,
    build_pack,
)
from shared.python.ai.knowledge.cli import main

from .conftest import commit_all


@pytest.fixture
def pack(tmp_path: Path, manifest: PackManifest, corpus: dict[str, Path]) -> Path:
    out = tmp_path / "packs" / "findings.sqlite"
    build_pack(manifest, corpus, out)
    return out


def test_build_reports_what_it_indexed(
    tmp_path: Path, manifest: PackManifest, corpus: dict[str, Path]
) -> None:
    info = build_pack(manifest, corpus, tmp_path / "p.sqlite")
    assert info.pack_id == "findings"
    assert set(info.commits) == {"AffineDrift", "UpstreamDrift"}
    assert all(len(sha) == 40 for sha in info.commits.values())
    assert info.passages >= 7
    assert info.files == 5  # the bibliography is excluded


def test_search_returns_cited_passages(pack: Path) -> None:
    hits = KnowledgePack.open(pack).search("pelvis torso timing")
    assert hits, "expected a hit"
    top = hits[0]
    assert isinstance(top, Passage)
    assert (top.repo, top.source, top.anchor) == (
        "AffineDrift",
        "articles/energy/energy.qmd",
        "energy-transfer/timing",
    )
    assert top.authority == "published"
    assert top.status == "current"
    assert len(top.commit) == 40
    where = "AffineDrift:articles/energy/energy.qmd#energy-transfer/timing"
    assert top.citation == f"{where} @ {top.commit[:8]}"


def test_excluded_files_are_not_indexed(pack: Path) -> None:
    assert KnowledgePack.open(pack).search("Bibliography references") == []


def test_superseded_and_retracted_are_hidden_by_default(pack: Path) -> None:
    kp = KnowledgePack.open(pack)
    sources = {h.source for h in kp.search("bounce dig depth")}
    assert sources == {"docs/assessments/bounce_new.md"}
    everything = {
        h.source: h.status
        for h in kp.search("bounce dig depth", include_superseded=True)
    }
    assert everything["docs/assessments/bounce_old.md"] == "superseded"
    assert (
        everything["docs/assessments/retracted.md"] == "retracted"
    )  # override beats front-matter


def test_search_is_safe_for_arbitrary_query_text(pack: Path) -> None:
    kp = KnowledgePack.open(pack)
    for query in [
        '"unbalanced',
        "NEAR(a b)",
        "a OR",
        "*",
        "",
        "   ",
        "rigid-segment chain?",
    ]:
        kp.search(query)  # must not raise an FTS5 syntax error
    assert kp.search("") == []


def test_k_limits_and_validates(pack: Path) -> None:
    kp = KnowledgePack.open(pack)
    assert len(kp.search("the", k=1, include_superseded=True)) <= 1
    with pytest.raises(ValueError, match="k"):
        kp.search("bounce", k=0)


def test_title_matches_outrank_body_matches(pack: Path) -> None:
    hits = KnowledgePack.open(pack).search("filtering")
    assert hits[0].anchor == "methods/filtering"


def test_info_reads_back_the_build(pack: Path) -> None:
    info = KnowledgePack.open(pack).info()
    assert info.pack_id == "findings"
    assert info.title == "Fixture findings"
    assert info.built_at.endswith("Z")


def test_pack_is_fresh_until_a_source_changes(
    pack: Path, corpus: dict[str, Path]
) -> None:
    kp = KnowledgePack.open(pack)
    assert kp.is_stale(corpus) is False

    # A commit that touches no indexed file does not make the pack stale.
    (corpus["UpstreamDrift"] / "README.md").write_text("unrelated\n", "utf-8")
    commit_all(corpus["UpstreamDrift"], "readme")
    assert kp.is_stale(corpus) is False

    target = corpus["UpstreamDrift"] / "docs" / "assessments" / "bounce_new.md"
    target.write_text(target.read_text("utf-8") + "\nNew evidence.\n", "utf-8")
    assert kp.is_stale(corpus) is True


def test_new_matching_file_makes_the_pack_stale(
    pack: Path, corpus: dict[str, Path]
) -> None:
    extra = corpus["AffineDrift"] / "articles" / "new.md"
    extra.write_text("# New\n\nFresh article.\n", "utf-8")
    assert KnowledgePack.open(pack).is_stale(corpus) is True


def test_missing_root_is_rejected_at_build(
    tmp_path: Path, manifest: PackManifest, corpus: dict[str, Path]
) -> None:
    with pytest.raises(ValueError, match="UpstreamDrift"):
        build_pack(
            manifest, {"AffineDrift": corpus["AffineDrift"]}, tmp_path / "x.sqlite"
        )


def test_unknown_format_version_is_refused(pack: Path) -> None:
    with sqlite3.connect(pack) as conn:
        conn.execute("PRAGMA user_version = 99")
    with pytest.raises(PackFormatError, match="99"):
        KnowledgePack.open(pack)


def test_missing_pack_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        KnowledgePack.open(tmp_path / "nope.sqlite")


def test_rebuild_replaces_atomically(
    pack: Path, manifest: PackManifest, corpus: dict[str, Path]
) -> None:
    first = KnowledgePack.open(pack).info().passages
    build_pack(manifest, corpus, pack)
    assert KnowledgePack.open(pack).info().passages == first
    assert list(pack.parent.glob("*.tmp")) == []


def test_cli_build_search_info(
    tmp_path: Path, corpus: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = tmp_path / "findings.yml"
    manifest_path.write_text(
        "id: findings\ntitle: CLI\nsources:\n"
        "  - repo: UpstreamDrift\n    authority: findings\n"
        "    include: ['docs/**/*.md', 'docs/**/*.tex']\n",
        "utf-8",
    )
    out = tmp_path / "cli.sqlite"
    rc = main(
        [
            "build",
            str(manifest_path),
            "--root",
            f"UpstreamDrift={corpus['UpstreamDrift']}",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert main(["search", str(out), "Butterworth filter", "-k", "2"]) == 0
    printed = capsys.readouterr().out
    assert "UpstreamDrift:docs/research/paper.tex#methods/filtering" in printed
    assert main(["info", str(out)]) == 0
    assert '"pack_id": "findings"' in capsys.readouterr().out
    assert (
        main(["build", str(manifest_path), "--root", "no-equals", "--out", str(out)])
        == 2
    )


def test_cli_output_survives_a_narrow_console_encoding(
    tmp_path: Path, corpus: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    doc = corpus["UpstreamDrift"] / "docs" / "research" / "check.tex"
    doc.write_text("\\section{Checks}\nAll passed \u2714 at 30\u00b0.\n", "utf-8")
    manifest_path = tmp_path / "m.yml"
    manifest_path.write_text(
        "id: m\nsources:\n  - repo: UpstreamDrift\n    authority: findings\n"
        "    include: ['docs/**/*.tex']\n",
        "utf-8",
    )
    out = tmp_path / "m.sqlite"
    root = f"UpstreamDrift={corpus['UpstreamDrift']}"
    assert main(["build", str(manifest_path), "--root", root, "--out", str(out)]) == 0
    narrow = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", narrow)
    assert main(["search", str(out), "checks passed"]) == 0
    narrow.flush()
    assert b"Checks" in narrow.buffer.getvalue()
