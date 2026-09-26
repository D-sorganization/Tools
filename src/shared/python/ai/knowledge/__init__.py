"""Knowledge packs: cited, freshness-checked retrieval over repository documents.

One engine backs the Disciple and Vision Quest staff roles (a findings pack
over UpstreamDrift and AffineDrift) and the per-product Sidekick Wizards (one
pack per product). A pack is a single SQLite FTS5 file ranked by BM25 with
optional dense MiniLM embeddings and hybrid RRF ranking (RM#1772 K4).

The package depends only on the standard library and PyYAML and imports
nothing else from Tools, so Runner_Dashboard can vendor it unchanged
(Tools#5345, Tools#5347, Repository_Management#1772).
"""

from __future__ import annotations

from .chunking import Chunk, chunk_document
from .eval import (
    CaseEvalResult,
    EvalSummary,
    GoldenQACase,
    evaluate_case,
    evaluate_pack,
    load_golden_set,
)
from .manifest import (
    AUTHORITIES,
    HIDDEN_STATUSES,
    STATUSES,
    ManifestError,
    PackManifest,
    SourceSpec,
    load_manifest,
    manifest_from_dict,
)
from .pack import (
    FORMAT_VERSION,
    Embedder,
    KnowledgePack,
    PackFormatError,
    PackInfo,
    Passage,
    blob_to_embedding,
    build_pack,
    cosine_similarity,
    embedding_to_blob,
    get_minilm_embedder,
    reciprocal_rank_fusion,
)

__all__ = [
    "AUTHORITIES",
    "FORMAT_VERSION",
    "HIDDEN_STATUSES",
    "STATUSES",
    "CaseEvalResult",
    "Chunk",
    "Embedder",
    "EvalSummary",
    "GoldenQACase",
    "KnowledgePack",
    "ManifestError",
    "PackFormatError",
    "PackInfo",
    "PackManifest",
    "Passage",
    "SourceSpec",
    "blob_to_embedding",
    "build_pack",
    "chunk_document",
    "cosine_similarity",
    "embedding_to_blob",
    "evaluate_case",
    "evaluate_pack",
    "get_minilm_embedder",
    "load_golden_set",
    "load_manifest",
    "manifest_from_dict",
    "reciprocal_rank_fusion",
]
