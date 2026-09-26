"""Tests for optional MiniLM embeddings and hybrid ranking (Tools #5347, RM#1772 K4)."""

from __future__ import annotations

import logging
import math
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from shared.python.ai.knowledge import (
    KnowledgePack,
    ManifestError,
    build_pack,
    manifest_from_dict,
)
from shared.python.ai.knowledge import (
    pack as pack_module,
)
from shared.python.ai.knowledge.pack import (
    blob_to_embedding,
    cosine_similarity,
    embedding_to_blob,
    get_minilm_embedder,
)


class DeterministicEmbedder:
    """Simple 4-dim deterministic embedder for testing vector math.

    Runs without ML dependencies.
    """

    def __init__(self, weights: dict[str, Sequence[float]] | None = None) -> None:
        self.weights = weights or {
            "pelvis": [1.0, 0.0, 0.0, 0.0],
            "torso": [0.8, 0.2, 0.0, 0.0],
            "bounce": [0.0, 1.0, 0.0, 0.0],
            "filter": [0.0, 0.0, 1.0, 0.0],
            "energy": [0.5, 0.5, 0.0, 0.0],
        }

    def embed(self, text: str) -> list[float]:
        vec = [0.0, 0.0, 0.0, 0.0]
        text_lower = text.lower()
        matched = False
        for word, w_vec in self.weights.items():
            if word in text_lower:
                matched = True
                for i in range(4):
                    vec[i] += w_vec[i]
        if not matched:
            vec = [0.1, 0.1, 0.1, 0.1]
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm > 0 else vec


def test_manifest_embeddings_flag() -> None:
    # Default is False
    m_default = manifest_from_dict(
        {
            "id": "p",
            "title": "P",
            "sources": [{"repo": "R", "authority": "published", "include": ["*.md"]}],
        }
    )
    assert m_default.embeddings is False
    assert m_default.to_dict()["embeddings"] is False

    # Explicit True
    m_true = manifest_from_dict(
        {
            "id": "p",
            "title": "P",
            "embeddings": True,
            "sources": [{"repo": "R", "authority": "published", "include": ["*.md"]}],
        }
    )
    assert m_true.embeddings is True
    assert m_true.to_dict()["embeddings"] is True

    # Invalid type
    with pytest.raises(ManifestError, match="embeddings"):
        manifest_from_dict(
            {
                "id": "p",
                "title": "P",
                "embeddings": "yes",
                "sources": [
                    {"repo": "R", "authority": "published", "include": ["*.md"]}
                ],
            }
        )


def test_cosine_similarity_math() -> None:
    # Identical
    assert pytest.approx(cosine_similarity([1.0, 0.0], [1.0, 0.0])) == 1.0
    # Orthogonal
    assert pytest.approx(cosine_similarity([1.0, 0.0], [0.0, 1.0])) == 0.0
    # Opposite
    assert pytest.approx(cosine_similarity([1.0, 0.0], [-1.0, 0.0])) == -1.0
    # Zero vector
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    # Dimension mismatch
    with pytest.raises(ValueError, match="mismatch"):
        cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


def test_embedding_blob_roundtrip() -> None:
    vec = [0.1234, -0.5678, 1.0, 0.0]
    blob = embedding_to_blob(vec)
    assert isinstance(blob, bytes)
    assert len(blob) == 16  # 4 floats * 4 bytes
    restored = blob_to_embedding(blob)
    for a, b in zip(vec, restored, strict=True):
        assert pytest.approx(a, rel=1e-5) == b


def test_build_pack_with_and_without_embeddings(
    corpus: dict[str, Path], tmp_path: Path
) -> None:
    embedder = DeterministicEmbedder()

    # Pack 1: with embeddings
    m_emb = manifest_from_dict(
        {
            "id": "with_emb",
            "title": "With Embeddings",
            "embeddings": True,
            "sources": [
                {
                    "repo": "AffineDrift",
                    "authority": "published",
                    "include": ["articles/**/*.qmd"],
                }
            ],
        }
    )
    pack1_path = tmp_path / "with_emb.pack"
    build_pack(m_emb, corpus, pack1_path, embedder=embedder)
    pack1 = KnowledgePack.open(pack1_path)
    assert pack1.has_embeddings is True

    # Verify column populated in sqlite
    conn = sqlite3.connect(f"{pack1_path.as_uri()}?mode=ro", uri=True)
    rows = conn.execute("SELECT count(*), count(embedding) FROM passages").fetchone()
    assert rows[0] > 0
    assert rows[0] == rows[1]
    conn.close()

    # Pack 2: without embeddings (default)
    m_no_emb = manifest_from_dict(
        {
            "id": "no_emb",
            "title": "No Embeddings",
            "sources": [
                {
                    "repo": "AffineDrift",
                    "authority": "published",
                    "include": ["articles/**/*.qmd"],
                }
            ],
        }
    )
    pack2_path = tmp_path / "no_emb.pack"
    build_pack(m_no_emb, corpus, pack2_path)
    pack2 = KnowledgePack.open(pack2_path)
    assert pack2.has_embeddings is False

    conn = sqlite3.connect(f"{pack2_path.as_uri()}?mode=ro", uri=True)
    rows = conn.execute("SELECT count(*), count(embedding) FROM passages").fetchone()
    assert rows[0] > 0
    assert rows[1] == 0
    conn.close()


def test_hybrid_search_fuses_rrf(corpus: dict[str, Path], tmp_path: Path) -> None:
    embedder = DeterministicEmbedder()
    manifest = manifest_from_dict(
        {
            "id": "findings",
            "title": "Findings",
            "embeddings": True,
            "sources": [
                {
                    "repo": "AffineDrift",
                    "authority": "published",
                    "include": ["articles/**/*.qmd"],
                    "exclude": ["**/*-bibliography.md"],
                },
                {
                    "repo": "UpstreamDrift",
                    "authority": "findings",
                    "include": ["docs/research/*", "docs/assessments/*"],
                },
            ],
        }
    )
    pack_path = tmp_path / "hybrid.pack"
    build_pack(manifest, corpus, pack_path, embedder=embedder)
    pack = KnowledgePack.open(pack_path)

    # Hybrid search
    hits_hybrid = pack.search("pelvis kinematic speed", k=5, embedder=embedder)
    assert len(hits_hybrid) > 0
    assert "articles/energy/energy.qmd" in hits_hybrid[0].source

    # Pure BM25 search
    hits_bm25 = pack.search("pelvis kinematic speed", k=5, hybrid=False)
    assert len(hits_bm25) > 0

    # Hybrid fused score is > 0
    assert hits_hybrid[0].score > 0


def test_sentence_transformers_optional_import() -> None:
    """Verify sentence-transformers is optional and gracefully handled."""
    import subprocess
    import sys

    try:
        probe = subprocess.run(
            [sys.executable, "-c", "import sentence_transformers"],
            capture_output=True,
            timeout=5,
        )
        if probe.returncode != 0:
            pytest.skip("sentence_transformers cannot be loaded in this environment")
    except (subprocess.TimeoutExpired, OSError):
        pytest.skip("sentence_transformers probe timed out or crashed")

    embedder = get_minilm_embedder()
    assert embedder is not None
    vec = embedder.embed("test sentence")
    assert len(vec) == 384


def test_get_minilm_embedder_logs_warning_and_returns_none_when_unavailable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Neither backend importable: log the reason and return None, not raise."""
    monkeypatch.setitem(sys.modules, "ai_backend", None)
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    with caplog.at_level(logging.WARNING):
        result = get_minilm_embedder()

    assert result is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "expected at least one warning to be logged"
    assert any("embedder" in r.message.lower() for r in warnings)


def test_search_hybrid_falls_back_to_bm25_when_no_embedder_available(
    corpus: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No embedder at search time: fall back to plain BM25 and warn once."""
    embedder = DeterministicEmbedder()
    manifest = manifest_from_dict(
        {
            "id": "fallback",
            "title": "Fallback",
            "embeddings": True,
            "sources": [
                {
                    "repo": "AffineDrift",
                    "authority": "published",
                    "include": ["articles/**/*.qmd"],
                    "exclude": ["**/*-bibliography.md"],
                }
            ],
        }
    )
    pack_path = tmp_path / "fallback.pack"
    build_pack(manifest, corpus, pack_path, embedder=embedder)
    pack = KnowledgePack.open(pack_path)
    assert pack.has_embeddings is True

    monkeypatch.setattr(pack_module, "get_minilm_embedder", lambda: None)

    with caplog.at_level(logging.WARNING):
        hybrid_hits = pack.search("pelvis kinematic speed", k=5)
    bm25_hits = pack.search("pelvis kinematic speed", k=5, hybrid=False)

    assert [h.source for h in hybrid_hits] == [h.source for h in bm25_hits]
    assert [h.anchor for h in hybrid_hits] == [h.anchor for h in bm25_hits]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("bm25" in r.message.lower() for r in warnings)
