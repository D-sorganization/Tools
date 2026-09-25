"""Build and query a knowledge pack: one SQLite FTS5 (BM25) file per manifest.

The pack file is self-describing (manifest, commits, per-file hashes) so a
consumer can check freshness and cite passages without the Tools repository.
Optionally stores dense embeddings for hybrid BM25 + cosine ranking (RM#1772 K4).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import struct
from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from .chunking import chunk_document
from .manifest import (
    AUTHORITIES,
    HIDDEN_STATUSES,
    STATUSES,
    PackManifest,
    manifest_from_dict,
)
from .sources import SourceFile, head_commit, select_files

#: Bumped on any incompatible schema change; readers refuse other versions.
FORMAT_VERSION = 1
_TITLE_WEIGHT = 5.0
_TEXT_WEIGHT = 1.0
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_MAX_QUERY_TOKENS = 32
_DEFAULT_RRF_K = 60

_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE files (
    repo TEXT NOT NULL, path TEXT NOT NULL, sha256 TEXT NOT NULL,
    PRIMARY KEY (repo, path)
);
CREATE TABLE passages (
    id INTEGER PRIMARY KEY, repo TEXT NOT NULL, source TEXT NOT NULL,
    anchor TEXT NOT NULL, title TEXT NOT NULL, text TEXT NOT NULL,
    commit_sha TEXT NOT NULL, content_hash TEXT NOT NULL, status TEXT NOT NULL,
    authority TEXT NOT NULL, authority_rank INTEGER NOT NULL,
    embedding BLOB
);
CREATE VIRTUAL TABLE passages_fts USING fts5(
    title, text, content='', tokenize='porter unicode61'
);
"""


@runtime_checkable
class Embedder(Protocol):
    """Protocol for vector embedders (e.g. MiniLM)."""

    def embed(self, text: str) -> Sequence[float]:
        """Embed a single text string into a float vector."""
        ...


def get_minilm_embedder() -> Embedder:
    """Return a MiniLM embedder using sentence-transformers or ai_backend.

    Raises RuntimeError if neither backend is available.
    """
    try:
        import ai_backend

        if hasattr(ai_backend, "LocalEmbedder"):
            embedder = ai_backend.LocalEmbedder.from_default_cache()

            class _RustEmbedder:
                def embed(self, text: str) -> Sequence[float]:
                    return embedder.embed(text)

            return _RustEmbedder()
    except Exception:
        pass

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")

        class _STEmbedder:
            def embed(self, text: str) -> Sequence[float]:
                vec = model.encode(text, convert_to_numpy=True)
                return [float(x) for x in vec]

        return _STEmbedder()
    except Exception:
        pass

    raise RuntimeError(
        "No MiniLM embedding backend available. Install sentence-transformers "
        "or build ai_backend with local-embeddings."
    )


def embedding_to_blob(vec: Sequence[float]) -> bytes:
    """Pack a sequence of floats into little-endian 32-bit float bytes."""
    return struct.pack(f"<{len(vec)}f", *vec)


def blob_to_embedding(blob: bytes) -> tuple[float, ...]:
    """Unpack little-endian 32-bit float bytes into a tuple of floats."""
    count = len(blob) // 4
    return struct.unpack(f"<{count}f", blob)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    if not a:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def reciprocal_rank_fusion(
    bm25_ranks: Mapping[int, int],
    cosine_ranks: Mapping[int, int],
    rrf_k: int = _DEFAULT_RRF_K,
) -> dict[int, float]:
    """Fuse BM25 ranks and cosine ranks using reciprocal rank fusion."""
    scores: dict[int, float] = {}
    for p_id, r in bm25_ranks.items():
        scores[p_id] = scores.get(p_id, 0.0) + (1.0 / (rrf_k + r))
    for p_id, r in cosine_ranks.items():
        scores[p_id] = scores.get(p_id, 0.0) + (1.0 / (rrf_k + r))
    return scores


class PackFormatError(RuntimeError):
    """The file is not a knowledge pack this engine can read."""


@dataclass(frozen=True)
class Passage:
    """One retrieved chunk with everything needed to cite it."""

    repo: str
    source: str
    anchor: str
    title: str
    text: str
    commit: str
    content_hash: str
    status: str
    authority: str
    score: float

    @property
    def citation(self) -> str:
        """``repo:path#anchor @ <short sha>``."""
        anchor = f"#{self.anchor}" if self.anchor else ""
        commit = f" @ {self.commit[:8]}" if self.commit else ""
        return f"{self.repo}:{self.source}{anchor}{commit}"


@dataclass(frozen=True)
class PackInfo:
    """What a pack was built from."""

    pack_id: str
    title: str
    built_at: str
    commits: Mapping[str, str]
    files: int
    passages: int
    format_version: int = FORMAT_VERSION


def build_pack(
    manifest: PackManifest,
    roots: Mapping[str, Path],
    out: Path,
    embedder: Embedder | Callable[[str], Sequence[float]] | None = None,
) -> PackInfo:
    """Index every file the manifest selects into ``out``, replacing it atomically.

    Preconditions: ``roots`` names an existing directory for every manifest repo.
    Postcondition: ``KnowledgePack.open(out).info()`` equals the returned value.
    """
    if not isinstance(manifest, PackManifest):
        raise TypeError("manifest must be a PackManifest")
    if manifest.embeddings and embedder is None:
        embedder = get_minilm_embedder()

    files = select_files(manifest, roots)
    commits = {repo: head_commit(Path(roots[repo])) for repo in manifest.repos}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.unlink(missing_ok=True)
    try:
        with closing(sqlite3.connect(tmp)) as conn:
            conn.executescript(_SCHEMA)
            count = sum(
                _index_file(conn, manifest, f, commits[f.repo], embedder) for f in files
            )
            info = PackInfo(
                pack_id=manifest.id,
                title=manifest.title,
                built_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                commits=commits,
                files=len(files),
                passages=count,
            )
            _write_meta(conn, manifest, info)
            conn.execute(f"PRAGMA user_version = {FORMAT_VERSION}")
            conn.commit()
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    return info


def _index_file(
    conn: sqlite3.Connection,
    manifest: PackManifest,
    source: SourceFile,
    commit: str,
    embedder: Embedder | Callable[[str], Sequence[float]] | None = None,
) -> int:
    raw = source.absolute.read_bytes()
    conn.execute(
        "INSERT INTO files (repo, path, sha256) VALUES (?, ?, ?)",
        (source.repo, source.path, hashlib.sha256(raw).hexdigest()),
    )
    chunks = chunk_document(
        raw.decode("utf-8", "replace"),
        suffix=Path(source.path).suffix,
        max_chars=manifest.chunk_chars,
    )
    override = manifest.status_overrides.get(source.path)
    rank = AUTHORITIES.index(source.authority)
    for chunk in chunks:
        status = override or (chunk.status if chunk.status in STATUSES else "current")
        emb_blob: bytes | None = None
        if manifest.embeddings and embedder is not None:
            emb_fn = getattr(embedder, "embed", embedder)
            emb_blob = embedding_to_blob(emb_fn(chunk.text))

        cur = conn.execute(
            "INSERT INTO passages (repo, source, anchor, title, text, commit_sha,"
            " content_hash, status, authority, authority_rank, embedding)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                source.repo,
                source.path,
                chunk.anchor,
                chunk.title,
                chunk.text,
                commit,
                hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
                status,
                source.authority,
                rank,
                emb_blob,
            ),
        )
        conn.execute(
            "INSERT INTO passages_fts (rowid, title, text) VALUES (?, ?, ?)",
            (cur.lastrowid, chunk.title, chunk.text),
        )
    return len(chunks)


def _write_meta(
    conn: sqlite3.Connection, manifest: PackManifest, info: PackInfo
) -> None:
    meta = {
        "pack_id": info.pack_id,
        "title": info.title,
        "built_at": info.built_at,
        "commits": json.dumps(dict(info.commits), sort_keys=True),
        "files": str(info.files),
        "passages": str(info.passages),
        "manifest": json.dumps(manifest.to_dict(), sort_keys=True),
    }
    conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta.items())


class KnowledgePack:
    """Read-only handle on a built pack. Construct with :meth:`open`."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path).resolve()

    @property
    def has_embeddings(self) -> bool:
        """True if this pack was built with embeddings enabled."""
        manifest_data = json.loads(self._meta().get("manifest", "{}"))
        return bool(manifest_data.get("embeddings", False))

    @classmethod
    def open(cls, path: Path) -> KnowledgePack:
        """Open ``path``; raises FileNotFoundError or PackFormatError."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"knowledge pack not found: {path}")
        pack = cls(path)
        with closing(pack._connect()) as conn:
            try:
                version = conn.execute("PRAGMA user_version").fetchone()[0]
            except sqlite3.DatabaseError as exc:
                raise PackFormatError(f"{path} is not a knowledge pack: {exc}") from exc
        if version != FORMAT_VERSION:
            raise PackFormatError(
                f"{path} has pack format {version}; this engine reads {FORMAT_VERSION}"
            )
        return pack

    def search(
        self,
        query: str,
        k: int = 8,
        include_superseded: bool = False,
        embedder: Embedder | Callable[[str], Sequence[float]] | None = None,
        hybrid: bool = True,
    ) -> list[Passage]:
        """Top ``k`` passages.

        Uses hybrid BM25 + cosine fused by reciprocal rank when the pack has
        embeddings and hybrid is True; otherwise pure BM25.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be a positive integer")

        if self.has_embeddings and hybrid:
            return self._search_hybrid(
                query, k=k, include_superseded=include_superseded, embedder=embedder
            )
        return self._search_bm25(query, k=k, include_superseded=include_superseded)

    def _search_bm25(
        self, query: str, k: int, include_superseded: bool
    ) -> list[Passage]:
        tokens = _TOKEN_RE.findall(query)[:_MAX_QUERY_TOKENS]
        if not tokens:
            return []
        match = " OR ".join('"' + t.replace('"', "") + '"' for t in tokens)
        hidden = "" if include_superseded else _hidden_clause()
        sql = (
            "SELECT p.repo, p.source, p.anchor, p.title, p.text, p.commit_sha,"
            " p.content_hash, p.status, p.authority,"
            f" bm25(passages_fts, {_TITLE_WEIGHT}, {_TEXT_WEIGHT}) AS score"
            " FROM passages_fts JOIN passages p ON p.id = passages_fts.rowid"
            f" WHERE passages_fts MATCH ?{hidden}"
            " ORDER BY score, p.authority_rank, p.id LIMIT ?"
        )
        with closing(self._connect()) as conn:
            rows = conn.execute(sql, (match, k)).fetchall()
        return [_passage(row) for row in rows]

    def _search_hybrid(
        self,
        query: str,
        k: int,
        include_superseded: bool,
        embedder: Embedder | Callable[[str], Sequence[float]] | None = None,
    ) -> list[Passage]:
        if embedder is None:
            embedder = get_minilm_embedder()
        emb_fn = getattr(embedder, "embed", embedder)
        query_vec = emb_fn(query)

        candidate_pool_size = max(k * 5, 50)
        tokens = _TOKEN_RE.findall(query)[:_MAX_QUERY_TOKENS]
        bm25_ids: list[int] = []
        hidden = "" if include_superseded else _hidden_clause()

        with closing(self._connect()) as conn:
            if tokens:
                match = " OR ".join('"' + t.replace('"', "") + '"' for t in tokens)
                bm25_sql = (
                    "SELECT p.id"
                    f" FROM passages_fts JOIN passages p ON p.id = passages_fts.rowid"
                    f" WHERE passages_fts MATCH ?{hidden}"
                    f" ORDER BY bm25(passages_fts, {_TITLE_WEIGHT}, {_TEXT_WEIGHT}),"
                    f" p.authority_rank, p.id LIMIT ?"
                )
                bm25_ids = [
                    row[0]
                    for row in conn.execute(
                        bm25_sql, (match, candidate_pool_size)
                    ).fetchall()
                ]

            bm25_ranks = {p_id: rank for rank, p_id in enumerate(bm25_ids, start=1)}

            emb_sql = (
                f"SELECT p.id, p.embedding FROM passages p"
                f" WHERE p.embedding IS NOT NULL{hidden}"
            )
            scored_candidates: list[tuple[float, int]] = []
            for p_id, blob in conn.execute(emb_sql).fetchall():
                p_vec = blob_to_embedding(blob)
                sim = cosine_similarity(query_vec, p_vec)
                scored_candidates.append((sim, p_id))

            scored_candidates.sort(key=lambda item: -item[0])
            cosine_top = scored_candidates[:candidate_pool_size]
            cosine_ranks = {
                p_id: rank for rank, (_, p_id) in enumerate(cosine_top, start=1)
            }

            fused_scores = reciprocal_rank_fusion(bm25_ranks, cosine_ranks)
            if not fused_scores:
                return []

            sorted_p_ids = sorted(
                fused_scores.keys(), key=lambda pid: -fused_scores[pid]
            )
            target_ids = sorted_p_ids[: k * 2]

            placeholders = ",".join("?" for _ in target_ids)
            fetch_sql = (
                "SELECT p.repo, p.source, p.anchor, p.title, p.text, p.commit_sha,"
                " p.content_hash, p.status, p.authority, p.authority_rank, p.id"
                f" FROM passages p WHERE p.id IN ({placeholders})"
            )
            rows = conn.execute(fetch_sql, target_ids).fetchall()

        passages: list[tuple[float, int, int, Passage]] = []
        for row in rows:
            p_id = int(row[10])
            auth_rank = int(row[9])
            f_score = fused_scores.get(p_id, 0.0)
            passages.append(
                (
                    f_score,
                    auth_rank,
                    p_id,
                    Passage(
                        repo=str(row[0]),
                        source=str(row[1]),
                        anchor=str(row[2]),
                        title=str(row[3]),
                        text=str(row[4]),
                        commit=str(row[5]),
                        content_hash=str(row[6]),
                        status=str(row[7]),
                        authority=str(row[8]),
                        score=f_score,
                    ),
                )
            )

        passages.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [item[3] for item in passages[:k]]

    def info(self) -> PackInfo:
        meta = self._meta()
        return PackInfo(
            pack_id=meta["pack_id"],
            title=meta["title"],
            built_at=meta["built_at"],
            commits=json.loads(meta["commits"]),
            files=int(meta["files"]),
            passages=int(meta["passages"]),
        )

    def is_stale(self, roots: Mapping[str, Path]) -> bool:
        """True when the selected file set or any selected file's bytes changed."""
        manifest = manifest_from_dict(json.loads(self._meta()["manifest"]))
        current = {(f.repo, f.path): f.sha256() for f in select_files(manifest, roots)}
        with closing(self._connect()) as conn:
            built = {
                (r, p): h
                for r, p, h in conn.execute("SELECT repo, path, sha256 FROM files")
            }
        return current != built

    def _meta(self) -> dict[str, str]:
        with closing(self._connect()) as conn:
            return dict(conn.execute("SELECT key, value FROM meta").fetchall())

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(f"{self._path.as_uri()}?mode=ro", uri=True)


def _passage(row: tuple[object, ...]) -> Passage:
    """Map a search row; FTS5 bm25 is lower-is-better, so the score is negated."""
    repo, source, anchor, title, text, commit, digest, status, authority = (
        str(v) for v in row[:9]
    )
    return Passage(
        repo=repo,
        source=source,
        anchor=anchor,
        title=title,
        text=text,
        commit=commit,
        content_hash=digest,
        status=status,
        authority=authority,
        score=-float(str(row[9])),
    )


def _hidden_clause() -> str:
    quoted = ", ".join(f"'{s}'" for s in sorted(HIDDEN_STATUSES))
    return f" AND p.status NOT IN ({quoted})"
