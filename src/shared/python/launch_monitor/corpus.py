"""Load the private launch-monitor shot corpus as validated canonical frames.

Step **P19** of the ADR-0046 G1 port plan (UpstreamDrift
``docs/adr/0048-launch-monitor-port-plan.md``). This is the plan's third
``needs-decision`` row and the one its three-way taxonomy has no bucket for, so
it is a **merge, not a port** — ADR-0046's Amendment 1 merge bucket, the same
one P18 used.

The two halves being merged
---------------------------
Both stacks export ``load_private_corpus``, both read the *same physical
dataset* — same ``LAUNCH_MONITOR_DATA_ROOT`` environment variable, same
``data/authority/database/shot_corpus_parquet`` partition tree — and neither is
a subset of the other:

``UpstreamDrift src/shared/python/launch_monitor/corpus.py`` (197 lines)
    Canonicalises the source-native imperial columns into the ADR-0031 schema
    (radians, m/s, rad/s, metres), derives a stable shot identity, and pushes a
    ``source_id`` and canonical-metric allowlist down into the Parquet reader.
    It validates **nothing about the corpus itself**.

``rate_of_closure/launch_monitor_private_corpus.py`` (106 lines)
    Validates the manifest schema version, the retained-row desktop cap, the
    row count and the source-partition set, reports a content-addressed
    ``manifest_sha256`` and a privacy-safe display label — then hands back the
    **native** columns untouched, with no selection expressible.

UpstreamDrift#9372's G0.1 gate
``tests/integration/launch_monitor_drift/test_corpus_drift.py`` measured both
sides over one synthetic two-source corpus and pinned the result in 13 gates:
they agree on the env var, the relative path, the resolved directory, the row
and column counts and the fail-closed-without-a-root posture, and they diverge
as **D28** (of 15 columns each only ``club`` and ``smash_factor`` are shared),
**D29** (unit canonicalisation exists only in UpstreamDrift), **D30** (manifest
validation exists only in ``rate_of_closure``) and **D31** (selection pushdown
exists only in UpstreamDrift).

D30 is the governance hole this merge exists to close
-----------------------------------------------------
D30 is not a stylistic difference. Its five parametrised cases are five corpora
that ``rate_of_closure`` **refuses** — a missing manifest, an unsupported
``schema_version``, a ``total_rows`` above the desktop cap, a row-count
mismatch, and a source-set mismatch — and that UpstreamDrift **loads silently**,
returning the same four rows in all five. A caller who wanted canonical units
had to give up every guarantee that the bytes on disk are the corpus the
authority published.

In this module the two are no longer alternatives. **Validation refuses what
the manifest rejects, and the canonicalisation then runs on what survives.**
There is no flag to skip it: an unvalidated corpus is not a canonical corpus.

Merge decisions
---------------
============================  =============================================
Capability                    Canonical decision
============================  =============================================
Env var + relative path       Kept once. ``PRIVATE_DATA_ENV`` and
                              ``CORPUS_RELATIVE_PATH`` are folded in from
                              ``rate_of_closure`` as the named constants both
                              halves now share; ``corpus_dataset_path`` keeps
                              UpstreamDrift's name and signature.
Root resolution               **Union.** ``rate_of_closure`` accepts either
                              the authority repository root or the
                              ``shot_corpus_parquet`` directory itself;
                              UpstreamDrift accepts only the former. Both are
                              accepted. UpstreamDrift's fail-closed message
                              when neither a root nor the environment variable
                              is supplied is kept verbatim.
Manifest validation           **Folded in and made mandatory** — the D30 hole,
                              closed. All five refusals travel with their
                              ``rate_of_closure`` messages.
``MAX_RETAINED_ROWS``         Defined here as ``300_000``. It is *not*
                              imported from ``rate_of_closure`` — this layer
                              never imports that package — and a seam test
                              pins the two constants equal.
Unit canonicalisation         UpstreamDrift's, unchanged (D29). Including its
                              refusal to convert ``apex_native``, whose unit
                              varies by source.
Shot identity                 UpstreamDrift's 20-hex-character digest over
                              ``source_id``/``file``/``row_index``, unchanged.
Selection pushdown            UpstreamDrift's, unchanged (D31) — but see the
                              next two rows, which are what makes it survive
                              contact with the validation gate.
Row-count check basis         ``rate_of_closure`` compares the manifest's
                              ``total_rows`` against ``len(frame)`` after
                              reading the whole corpus with
                              ``pd.read_parquet``. Under a selection that
                              check would be unsatisfiable by construction, so
                              the canonical loader compares against
                              ``dataset.count_rows()`` on the **unfiltered**
                              dataset. Same guarantee, cheaper, and it holds
                              whether or not the caller pruned.
Source-set check basis        Likewise: the observed source set comes from the
                              hive partition directory names rather than from
                              the loaded frame, so pruning cannot weaken it.
Provenance surface            Folded in as ``CanonicalPrivateCorpus`` —
                              ``frame``, ``parquet_path``, ``manifest_sha256``,
                              ``source_count`` and the same privacy-safe
                              ``source_name`` label the desktop UI renders —
                              returned by
                              :func:`load_private_corpus_with_provenance`.
                              :func:`load_private_corpus` keeps
                              UpstreamDrift's ``DataFrame`` return so the ported
                              callers and tests are unchanged.
============================  =============================================

Which pins this moves
---------------------
**None of the G0.1 pins move**, and that is correct rather than a miss. All 13
gates in ``test_corpus_drift.py`` measure UpstreamDrift's ``corpus.py`` against
``rate_of_closure.launch_monitor_private_corpus``, and this PR touches neither.
Retiring either legacy posture is a coordinated cross-repo change for the same
reason G1-D3's and D22/D23's legacy halves are, and is tracked rather than
smuggled.

What *would* move if the gate were re-pointed at this module is exactly one
divergence class, and only in one direction: **D30 inverts.** Its five
parametrised cases assert "``rate_of_closure`` refuses, UpstreamDrift accepts";
against this module the canonical side refuses all five, so the divergence
ceases to exist rather than changing value.
``test_divergence_d30_only_tools_reports_the_manifest_digest`` likewise stops
being a divergence — the canonical loader reports the same digest and the same
``source_name`` string. **D28 narrows but does not close** (the canonical frame
is UpstreamDrift's column set, so it stays almost disjoint from the native
one), and **D29 and D31 are unchanged**, because both are UpstreamDrift
capabilities carried over verbatim.

Reading Parquet requires ``pyarrow``; the import is lazy so this module stays
importable without it.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from shared.python.launch_monitor.importer import _convert

PRIVATE_DATA_ENV = "LAUNCH_MONITOR_DATA_ROOT"
CORPUS_RELATIVE_PATH = Path("data/authority/database/shot_corpus_parquet")
MANIFEST_FILENAME = "_MANIFEST.json"
SUPPORTED_MANIFEST_SCHEMA_VERSION = 1

# The desktop retained-data limit, folded in from ``rate_of_closure``'s
# ``launch_monitor_linked_scatter``. It is redefined rather than imported
# because the canonical layer must not depend on ``rate_of_closure``; the two
# values are pinned equal by a seam test in this module's suite.
MAX_RETAINED_ROWS = 300_000

# Corpus native column -> (canonical metric name, source unit). The corpus
# stores source-native imperial units. ``apex_native`` is excluded: its unit
# varies by source, so it cannot be converted safely.
CORPUS_COLUMN_MAP: dict[str, tuple[str, str]] = {
    "club_speed_mph": ("club_speed", "mph"),
    "ball_speed_mph": ("ball_speed", "mph"),
    "smash_factor": ("smash_factor", "1"),
    "launch_angle_deg": ("launch_angle", "deg"),
    "launch_direction_deg": ("launch_direction", "deg"),
    "spin_rate_rpm": ("spin_rate", "rpm"),
    "back_spin_rpm": ("back_spin", "rpm"),
    "side_spin_rpm": ("side_spin", "rpm"),
    "spin_axis_deg": ("spin_axis", "deg"),
    "attack_angle_deg": ("attack_angle", "deg"),
    "club_path_deg": ("club_path", "deg"),
    "face_angle_deg": ("face_angle", "deg"),
    "carry_yd": ("carry_distance", "yd"),
    "total_yd": ("total_distance", "yd"),
    "descent_angle_deg": ("descent_angle", "deg"),
    "lateral_carry_yd": ("lateral_carry", "yd"),
    "flight_time_s": ("flight_time", "s"),
}

# Identity columns carried straight through when the corpus provides them.
# captured_at is what the Trends analysis binds to; a corpus built before the
# data authority added it simply lacks the column.
OPTIONAL_IDENTITY_COLUMNS: tuple[str, ...] = ("captured_at",)


@dataclass(frozen=True)
class CorpusManifest:
    """The authority's published description of one corpus snapshot."""

    schema_version: int
    sources: dict[str, Any]
    total_rows: int
    manifest_sha256: str

    @property
    def source_count(self) -> int:
        """Return how many source partitions the manifest declares."""
        return len(self.sources)


@dataclass(frozen=True)
class CanonicalPrivateCorpus:
    """A validated canonical-schema frame and its immutable provenance."""

    frame: pd.DataFrame
    parquet_path: Path
    manifest_sha256: str
    source_count: int

    @property
    def source_name(self) -> str:
        """Return a privacy-safe source label for the desktop UI."""
        return (
            f"Private Corpus ({self.source_count} sources; manifest "
            f"{self.manifest_sha256[:12]}...)"
        )


def _authority_root(root: str | Path | None) -> Path:
    """Resolve the authorized checkout root, failing closed without one."""
    resolved = (
        root if root is not None else os.environ.get(PRIVATE_DATA_ENV, "").strip()
    )
    if not resolved:
        raise FileNotFoundError(
            "private launch-monitor authority is unavailable; set "
            "LAUNCH_MONITOR_DATA_ROOT to an authorized, commit-pinned "
            "Launch-Monitor-Flight-Model-Campaign checkout"
        )
    return Path(resolved).expanduser()


def corpus_dataset_path(root: str | Path | None = None) -> Path:
    """Resolve the Parquet corpus path inside the private checkout."""
    return _authority_root(root) / CORPUS_RELATIVE_PATH


def resolve_private_corpus_path(root: str | Path | None = None) -> Path:
    """Resolve a checkout root *or* a corpus directory to the corpus directory.

    Union of both halves' resolution: ``rate_of_closure`` lets a caller select
    either the authority repository root or the ``shot_corpus_parquet``
    directory itself, and UpstreamDrift's convention (root plus the ADR-0031
    relative path) is tried first. The two failure modes stay distinguishable —
    a checkout with no corpus at all reports UpstreamDrift's message, and a
    corpus with no manifest reports ``rate_of_closure``'s.
    """
    base = _authority_root(root)
    candidates = (base / CORPUS_RELATIVE_PATH, base)
    for choice in candidates:
        if (choice / MANIFEST_FILENAME).is_file():
            return choice
    present = [choice for choice in candidates if choice.is_dir()]
    if present:
        raise FileNotFoundError(
            f"Private corpus manifest not found at {present[0] / MANIFEST_FILENAME}. "
            "Select either the authority repository root or its "
            "shot_corpus_parquet directory. The canonical loader refuses an "
            "unvalidated corpus (ADR-0048 D30)."
        )
    raise FileNotFoundError(
        f"shot corpus dataset not found at {candidates[0]}; the checkout "
        "may predate the Parquet corpus - sync it to a newer commit"
    )


def read_corpus_manifest(dataset_dir: Path) -> CorpusManifest:
    """Read and schema-check the corpus manifest, content-addressing its bytes."""
    manifest_bytes = (dataset_dir / MANIFEST_FILENAME).read_bytes()
    payload = json.loads(manifest_bytes)
    if payload.get("schema_version") != SUPPORTED_MANIFEST_SCHEMA_VERSION or not (
        isinstance(payload.get("sources"), dict)
    ):
        raise ValueError("Private corpus manifest schema is unsupported")
    return CorpusManifest(
        schema_version=int(payload["schema_version"]),
        sources=dict(payload["sources"]),
        total_rows=int(payload.get("total_rows", -1)),
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )


def validate_corpus_manifest(
    manifest: CorpusManifest,
    *,
    observed_rows: int,
    observed_sources: set[str],
) -> None:
    """Refuse a corpus whose bytes disagree with the manifest that describes it.

    ``observed_rows`` and ``observed_sources`` must describe the **whole**
    dataset, before any ``sources``/``metrics`` selection is applied — the
    guarantee is about the corpus on disk, not about the caller's slice of it.
    """
    if not 0 <= manifest.total_rows <= MAX_RETAINED_ROWS:
        raise ValueError(
            "Private corpus manifest row count is outside the desktop retained-"
            f"data limit of {MAX_RETAINED_ROWS}"
        )
    if observed_rows != manifest.total_rows:
        raise ValueError(
            f"Private corpus row count mismatch: expected {manifest.total_rows}, "
            f"loaded {observed_rows}"
        )
    if observed_sources != set(manifest.sources):
        raise ValueError("Private corpus source IDs do not match the manifest")


def _partition_source_ids(dataset_dir: Path) -> set[str]:
    """Return the ``source_id`` values the hive partition tree actually holds."""
    return {
        entry.name.split("=", 1)[1]
        for entry in dataset_dir.iterdir()
        if entry.is_dir() and entry.name.startswith("source_id=")
    }


def _selected_column_map(metrics: list[str] | None) -> dict[str, tuple[str, str]]:
    """Narrow the corpus column map to an optional canonical-metric allowlist."""
    if metrics is None:
        return dict(CORPUS_COLUMN_MAP)
    unknown = set(metrics) - {name for name, _ in CORPUS_COLUMN_MAP.values()}
    if unknown:
        raise ValueError(f"Unknown corpus metrics requested: {sorted(unknown)}")
    return {
        column: (name, unit)
        for column, (name, unit) in CORPUS_COLUMN_MAP.items()
        if name in metrics
    }


def _source_filter(
    pyarrow_dataset: Any, available: set[str], sources: list[str] | None
) -> Any:
    """Build the partition filter for a ``source_id`` allowlist, if any."""
    if sources is None:
        return None
    unknown = set(sources) - available
    if unknown:
        raise ValueError(f"Unknown corpus sources requested: {sorted(unknown)}")
    return pyarrow_dataset.field("source_id").isin(sources)


def _canonicalize_metrics(
    frame: pd.DataFrame, selected_map: dict[str, tuple[str, str]]
) -> pd.DataFrame:
    """Convert native corpus columns to canonical units and metric names."""
    from shared.python.launch_monitor.schema import METRICS

    present = {
        column: value
        for column, value in selected_map.items()
        if column in frame.columns
    }
    for column, (name, unit) in present.items():
        frame[column] = _convert(frame[column], unit, METRICS[name].canonical_unit)
    return frame.rename(columns={column: name for column, (name, _) in present.items()})


def _apply_identity(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive ``shot_id`` and rename corpus identity columns to the schema."""
    identity = (
        frame["source_id"].astype(str)
        + "\x1f"
        + frame["file"].astype(str)
        + "\x1f"
        + frame["row_index"].astype(str)
    )
    frame["shot_id"] = identity.map(
        lambda value: hashlib.sha256(value.encode()).hexdigest()[:20]
    )
    return frame.rename(
        columns={
            "source_id": "session_id",
            "monitor": "monitor_vendor",
            "row_index": "source_row",
        }
    ).drop(columns=["file"])


def load_private_corpus_with_provenance(
    root: str | Path | None = None,
    sources: list[str] | None = None,
    metrics: list[str] | None = None,
) -> CanonicalPrivateCorpus:
    """Load corpus shots as canonical frames, refusing an invalid corpus.

    The manifest gate runs **first and always**, against the unfiltered
    dataset, and only then is the caller's selection applied and the surviving
    rows canonicalised into the ADR-0031 schema.

    Args:
        root: Authority checkout root or the corpus directory itself; defaults
            to ``LAUNCH_MONITOR_DATA_ROOT``.
        sources: Optional ``source_id`` allowlist; ``None`` loads everything.
        metrics: Optional canonical metric-name allowlist; pruning is pushed
            down to the Parquet reader.

    Returns:
        A :class:`CanonicalPrivateCorpus` carrying the canonical-schema frame
        and the corpus' content-addressed provenance.

    Raises:
        FileNotFoundError: No authorized root, no corpus, or no manifest.
        ValueError: The manifest schema, row cap, row count or source set does
            not describe the corpus on disk, or an unknown source or metric
            was requested.
        ImportError: ``pyarrow`` is not installed.
    """
    try:
        import pyarrow.dataset as pyarrow_dataset
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "loading the private corpus requires pyarrow; install it with "
            "pip install pyarrow"
        ) from exc

    dataset_dir = resolve_private_corpus_path(root)
    manifest = read_corpus_manifest(dataset_dir)

    dataset = pyarrow_dataset.dataset(
        dataset_dir, format="parquet", partitioning="hive"
    )
    available_sources = _partition_source_ids(dataset_dir)
    validate_corpus_manifest(
        manifest,
        observed_rows=dataset.count_rows(),
        observed_sources=available_sources,
    )

    selected_map = _selected_column_map(metrics)
    # A corpus pinned before a column was introduced simply lacks it; select
    # what the dataset actually has rather than failing the whole read.
    available_columns = set(dataset.schema.names)
    requested = [
        "source_id",
        "monitor",
        "club",
        "file",
        "row_index",
        *OPTIONAL_IDENTITY_COLUMNS,
        *selected_map,
    ]
    table = dataset.to_table(
        columns=[name for name in requested if name in available_columns],
        filter=_source_filter(pyarrow_dataset, available_sources, sources),
    )

    frame = _canonicalize_metrics(table.to_pandas(), selected_map)
    frame = _apply_identity(frame)
    frame["observation_kind"] = "shot"
    return CanonicalPrivateCorpus(
        frame=frame,
        parquet_path=dataset_dir,
        manifest_sha256=manifest.manifest_sha256,
        source_count=manifest.source_count,
    )


def load_private_corpus(
    root: str | Path | None = None,
    sources: list[str] | None = None,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Load validated corpus shots as one canonical-schema DataFrame.

    UpstreamDrift's signature and return type, now behind the manifest gate.
    Callers that need the corpus' content-addressed provenance should use
    :func:`load_private_corpus_with_provenance` instead.

    Returns:
        DataFrame with canonical metric columns, identity columns
        (``shot_id``, ``session_id`` carrying the corpus ``source_id``,
        ``source_row``, ``monitor_vendor``, ``club``), and
        ``observation_kind`` fixed to ``"shot"``.
    """
    return load_private_corpus_with_provenance(
        root, sources=sources, metrics=metrics
    ).frame


__all__ = [
    "CORPUS_COLUMN_MAP",
    "CORPUS_RELATIVE_PATH",
    "MANIFEST_FILENAME",
    "MAX_RETAINED_ROWS",
    "OPTIONAL_IDENTITY_COLUMNS",
    "PRIVATE_DATA_ENV",
    "SUPPORTED_MANIFEST_SCHEMA_VERSION",
    "CanonicalPrivateCorpus",
    "CorpusManifest",
    "corpus_dataset_path",
    "load_private_corpus",
    "load_private_corpus_with_provenance",
    "read_corpus_manifest",
    "resolve_private_corpus_path",
    "validate_corpus_manifest",
]
