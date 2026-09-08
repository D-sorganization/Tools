"""Strict TOOLS-D6 calculation freshness, executable evidence, and reverse-impact contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

FRESHNESS_PATH = PurePosixPath("manuals/tools/calculation-freshness.json")
FRESHNESS_SCHEMA_PATH = PurePosixPath(
    "manuals/tools/schemas/calculation-freshness.schema.json"
)
FRESHNESS_VERSION = "tools-calculation-freshness/1.0.0"
CALCULATION_REGISTRY_PATH = PurePosixPath("manuals/tools/calculation-registry.json")
EXEMPLAR_COVERAGE_PATH = PurePosixPath("manuals/tools/exemplar-coverage.json")
TEXTBOOK_REGISTRY_PATH = PurePosixPath("manuals/tools/textbook-chapters.json")
CHAPTER_CONTRACT_PATH = PurePosixPath("manuals/tools/textbook-chapter-contract.json")
TOOLCHAIN_LOCK_PATH = PurePosixPath("manuals/tools/toolchain-lock.json")
ARTIFACT_MANIFEST_PATH = PurePosixPath("manuals/tools/manifests/artifacts.json")
REFERENCES_PATH = PurePosixPath("manuals/tools/references.bib")

HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
EXEMPTION_ID_PATTERN = re.compile(r"^EXEMPT-[A-Z0-9-]+$")
CALCULATION_ID_PATTERN = re.compile(r"^TOOLS-[A-Z0-9-]+$")
EXAMPLE_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
PLACEHOLDER_TOKENS = frozenset({"fixme", "tbd", "todo", "placeholder", "xxx"})


class CalculationFreshnessError(RuntimeError):
    """Base error for calculation freshness and evidence failures."""


class ReverseImpactError(CalculationFreshnessError):
    """Raised when a transitive dependency or referenced file moved, deleted, or orphaned."""


class ExemptionError(CalculationFreshnessError):
    """Raised when an exemption is invalid, expired, revoked, or unreviewed."""


class DriftError(CalculationFreshnessError):
    """Raised when numerical or source drift is detected without an active exemption."""


@dataclass(frozen=True)
class SourceLink:
    path: PurePosixPath
    symbol: str
    symbol_type: str
    sha256_lf: str


@dataclass(frozen=True)
class SchemaLink:
    path: PurePosixPath
    schema_version: str
    sha256_lf: str


@dataclass(frozen=True)
class TestLink:
    test_id: str
    path: str
    level: str
    sha256_lf: str


@dataclass(frozen=True)
class CitationLink:
    citation_id: str
    locator: str


@dataclass(frozen=True)
class ExecutableExample:
    example_id: str
    category: str
    description: str
    fixture_path: PurePosixPath
    fixture_case_id: str
    inputs: dict[str, Any]
    expected_outcome: dict[str, Any]
    execution_digest: str


@dataclass(frozen=True)
class DocumentedValue:
    value_id: str
    field_name: str
    example_id: str
    value: Any
    unit: str
    tolerance: float | None
    manual_chapter: PurePosixPath
    manual_anchor: str


@dataclass(frozen=True)
class DocumentedTable:
    table_id: str
    title: str
    manual_chapter: PurePosixPath
    manual_anchor: str
    headers: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    digest: str


@dataclass(frozen=True)
class DocumentedFigure:
    figure_id: str
    path: PurePosixPath
    media_type: str
    caption: str
    sha256: str


@dataclass(frozen=True)
class CalculationFreshnessEntry:
    calculation_id: str
    title: str
    status: str
    method_version: str
    generator_script: PurePosixPath
    source_links: tuple[SourceLink, ...]
    schema_links: tuple[SchemaLink, ...]
    test_links: tuple[TestLink, ...]
    citation_links: tuple[CitationLink, ...]
    executable_examples: tuple[ExecutableExample, ...]
    documented_values: tuple[DocumentedValue, ...]
    documented_tables: tuple[DocumentedTable, ...]
    documented_figures: tuple[DocumentedFigure, ...]
    transitive_dependencies: tuple[PurePosixPath, ...]


@dataclass(frozen=True)
class ExpiringExemption:
    exemption_id: str
    calculation_id: str
    drift_type: str
    target_path: PurePosixPath
    rationale: str
    reviewed_by: tuple[str, ...]
    created_at: datetime
    expires_at: datetime
    status: str


@dataclass(frozen=True)
class CalculationFreshnessManifest:
    schema_version: str
    manual_id: str
    release_status: str
    owner_subepic: int
    calculations: tuple[CalculationFreshnessEntry, ...]
    exemptions: tuple[ExpiringExemption, ...]


@dataclass(frozen=True)
class FreshnessVerificationSummary:
    calculation_count: int
    nominal_count: int
    boundary_count: int
    failure_count: int
    documented_value_count: int
    table_count: int
    active_exemption_count: int
    verified_at: datetime


def _ensure_src_path(root: Path | None = None) -> None:
    """Ensure Tools repository src directory is in sys.path."""
    if root is not None:
        src_path = str((root.resolve() / "src").resolve())
    else:
        src_path = str((Path(__file__).resolve().parents[1] / "src").resolve())
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def sha256_lf(data: bytes | str) -> str:
    """Compute SHA-256 after normalizing CRLF to LF."""
    if isinstance(data, str):
        content = data.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    else:
        content = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def sha256_file_lf(path: Path) -> str:
    """Read a text/binary file and return LF-normalized sha256."""
    try:
        data = path.read_bytes()
    except OSError as err:
        raise CalculationFreshnessError(
            f"unable to read file for digest: {path}"
        ) from err
    return sha256_lf(data)


def canonical_json_digest(data: Any) -> str:
    """Return SHA-256 hex digest of sorted canonical JSON representation."""
    encoded = json.dumps(
        data,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_posix_path(val: Any, label: str) -> PurePosixPath:
    if not isinstance(val, str) or not val.strip() or "\\" in val:
        raise CalculationFreshnessError(f"{label} must be a valid posix path: {val!r}")
    path = PurePosixPath(val.strip())
    if path.is_absolute() or ".." in path.parts:
        raise CalculationFreshnessError(f"{label} must be relative and safe: {val!r}")
    return path


def _parse_iso_datetime(val: Any, label: str) -> datetime:
    if not isinstance(val, str) or not val.strip():
        raise CalculationFreshnessError(f"{label} must be an ISO 8601 string: {val!r}")
    normalized = val.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise CalculationFreshnessError(
            f"{label} invalid ISO datetime: {val!r}"
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def load_calculation_freshness(payload: dict[str, Any]) -> CalculationFreshnessManifest:
    """Load and validate the calculation-freshness document with typed invariants."""
    if not isinstance(payload, dict):
        raise CalculationFreshnessError("freshness document must be a dict")

    expected_keys = {
        "schema_version",
        "manual_id",
        "release_status",
        "owner_subepic",
        "calculations",
        "exemptions",
    }
    actual_keys = set(payload)
    if actual_keys != expected_keys:
        raise CalculationFreshnessError(
            f"freshness document keys differ: missing={expected_keys - actual_keys} "
            f"extra={actual_keys - expected_keys}"
        )

    if payload["schema_version"] != FRESHNESS_VERSION:
        raise CalculationFreshnessError(
            f"unsupported freshness schema_version: {payload['schema_version']!r}"
        )
    if payload["manual_id"] != "tools":
        raise CalculationFreshnessError(
            f"manual_id must be 'tools': {payload['manual_id']!r}"
        )
    if payload["release_status"] != "provisional":
        raise CalculationFreshnessError("release_status must remain 'provisional'")
    if payload["owner_subepic"] != 4723:
        raise CalculationFreshnessError("owner_subepic must be 4723")

    calc_entries: list[CalculationFreshnessEntry] = []
    for raw in payload.get("calculations", []):
        if not isinstance(raw, dict):
            raise CalculationFreshnessError("calculation entry must be an object")

        calc_id = raw.get("calculation_id", "")
        if not isinstance(calc_id, str) or not CALCULATION_ID_PATTERN.fullmatch(
            calc_id
        ):
            raise CalculationFreshnessError(f"invalid calculation_id: {calc_id!r}")

        title = raw.get("title", "")
        if not isinstance(title, str) or len(title.strip()) < 3:
            raise CalculationFreshnessError(f"invalid title for {calc_id}")

        status = raw.get("status", "")
        if status not in ("verified-unapproved", "blocked"):
            raise CalculationFreshnessError(
                f"unsupported status {status!r} for {calc_id}"
            )

        method_version = raw.get("method_version", "")
        if not isinstance(method_version, str) or not re.fullmatch(
            r"^[0-9]+\.[0-9]+\.[0-9]+$", method_version
        ):
            raise CalculationFreshnessError(f"invalid method_version for {calc_id}")

        gen_script = _safe_posix_path(
            raw.get("generator_script"), f"{calc_id}.generator_script"
        )

        # source_links
        source_links: list[SourceLink] = []
        for s in raw.get("source_links", []):
            if not isinstance(s, dict):
                raise CalculationFreshnessError(f"invalid source link in {calc_id}")
            sp = _safe_posix_path(s.get("path"), f"{calc_id}.source_links.path")
            sym = str(s.get("symbol", "")).strip()
            stype = str(s.get("symbol_type", "")).strip()
            sha = str(s.get("sha256_lf", "")).strip()
            if (
                not sym
                or stype not in ("function", "class", "module")
                or not HASH_PATTERN.fullmatch(sha)
            ):
                raise CalculationFreshnessError(
                    f"invalid source link fields in {calc_id}"
                )
            source_links.append(SourceLink(sp, sym, stype, sha))

        # schema_links
        schema_links: list[SchemaLink] = []
        for sc in raw.get("schema_links", []):
            if not isinstance(sc, dict):
                raise CalculationFreshnessError(f"invalid schema link in {calc_id}")
            scp = _safe_posix_path(sc.get("path"), f"{calc_id}.schema_links.path")
            scv = str(sc.get("schema_version", "")).strip()
            sha = str(sc.get("sha256_lf", "")).strip()
            if not scv or not HASH_PATTERN.fullmatch(sha):
                raise CalculationFreshnessError(
                    f"invalid schema link fields in {calc_id}"
                )
            schema_links.append(SchemaLink(scp, scv, sha))

        # test_links
        test_links: list[TestLink] = []
        for t in raw.get("test_links", []):
            if not isinstance(t, dict):
                raise CalculationFreshnessError(f"invalid test link in {calc_id}")
            tid = str(t.get("test_id", "")).strip()
            tp = str(t.get("path", "")).strip()
            lvl = str(t.get("level", "")).strip()
            sha = str(t.get("sha256_lf", "")).strip()
            if (
                not tid
                or not tp
                or lvl not in ("parity", "unit", "integration")
                or not HASH_PATTERN.fullmatch(sha)
            ):
                raise CalculationFreshnessError(
                    f"invalid test link fields in {calc_id}"
                )
            test_links.append(TestLink(tid, tp, lvl, sha))

        # citation_links
        citation_links: list[CitationLink] = []
        for c in raw.get("citation_links", []):
            if not isinstance(c, dict):
                raise CalculationFreshnessError(f"invalid citation link in {calc_id}")
            cid = str(c.get("citation_id", "")).strip()
            loc = str(c.get("locator", "")).strip()
            if not cid or not loc:
                raise CalculationFreshnessError(
                    f"invalid citation link fields in {calc_id}"
                )
            citation_links.append(CitationLink(cid, loc))

        # executable_examples
        examples: list[ExecutableExample] = []
        categories: set[str] = set()
        for ex in raw.get("executable_examples", []):
            if not isinstance(ex, dict):
                raise CalculationFreshnessError(
                    f"invalid executable example in {calc_id}"
                )
            eid = str(ex.get("example_id", "")).strip()
            if not EXAMPLE_ID_PATTERN.fullmatch(eid):
                raise CalculationFreshnessError(
                    f"invalid example_id {eid!r} in {calc_id}"
                )
            cat = str(ex.get("category", "")).strip()
            if cat not in ("nominal", "boundary", "failure"):
                raise CalculationFreshnessError(
                    f"invalid category {cat!r} in {calc_id}.{eid}"
                )
            categories.add(cat)
            desc = str(ex.get("description", "")).strip()
            if len(desc) < 3:
                raise CalculationFreshnessError(
                    f"description too short in {calc_id}.{eid}"
                )
            fp = _safe_posix_path(
                ex.get("fixture_path"), f"{calc_id}.{eid}.fixture_path"
            )
            fcid = str(ex.get("fixture_case_id", "")).strip()
            inp = ex.get("inputs")
            out = ex.get("expected_outcome")
            dig = str(ex.get("execution_digest", "")).strip()
            if (
                not fcid
                or not isinstance(inp, dict)
                or not isinstance(out, dict)
                or not HASH_PATTERN.fullmatch(dig)
            ):
                raise CalculationFreshnessError(
                    f"invalid example payload in {calc_id}.{eid}"
                )
            examples.append(ExecutableExample(eid, cat, desc, fp, fcid, inp, out, dig))

        # Check required nominal, boundary, failure coverage!
        if status == "verified-unapproved":
            missing_cats = {"nominal", "boundary", "failure"} - categories
            if missing_cats:
                raise CalculationFreshnessError(
                    f"{calc_id} must provide nominal, boundary, and failure examples; missing: {sorted(missing_cats)}"
                )

        # documented_values
        doc_vals: list[DocumentedValue] = []
        for dv in raw.get("documented_values", []):
            if not isinstance(dv, dict):
                raise CalculationFreshnessError(
                    f"invalid documented value in {calc_id}"
                )
            vid = str(dv.get("value_id", "")).strip()
            fname = str(dv.get("field_name", "")).strip()
            veid = str(dv.get("example_id", "")).strip()
            val = dv.get("value")
            unit = str(dv.get("unit", "")).strip()
            tol = dv.get("tolerance")
            if tol is not None and not isinstance(tol, (int, float)):
                raise CalculationFreshnessError(f"invalid tolerance for {vid}")
            ch = _safe_posix_path(
                dv.get("manual_chapter"), f"{calc_id}.{vid}.manual_chapter"
            )
            anc = str(dv.get("manual_anchor", "")).strip()
            if not vid or not fname or not veid or not unit or not anc.startswith("#"):
                raise CalculationFreshnessError(
                    f"invalid documented value fields in {vid}"
                )
            doc_vals.append(
                DocumentedValue(
                    vid,
                    fname,
                    veid,
                    val,
                    unit,
                    float(tol) if tol is not None else None,
                    ch,
                    anc,
                )
            )

        # documented_tables
        doc_tables: list[DocumentedTable] = []
        for dt in raw.get("documented_tables", []):
            if not isinstance(dt, dict):
                raise CalculationFreshnessError(
                    f"invalid documented table in {calc_id}"
                )
            tid = str(dt.get("table_id", "")).strip()
            ttitle = str(dt.get("title", "")).strip()
            tch = _safe_posix_path(
                dt.get("manual_chapter"), f"{calc_id}.{tid}.manual_chapter"
            )
            tanc = str(dt.get("manual_anchor", "")).strip()
            headers = tuple(str(h) for h in dt.get("headers", []))
            rows = tuple(tuple(r) for r in dt.get("rows", []))
            tdig = str(dt.get("digest", "")).strip()
            if (
                not tid
                or not ttitle
                or not tanc.startswith("#")
                or not HASH_PATTERN.fullmatch(tdig)
            ):
                raise CalculationFreshnessError(f"invalid documented table in {tid}")
            doc_tables.append(
                DocumentedTable(tid, ttitle, tch, tanc, headers, rows, tdig)
            )

        # documented_figures
        doc_figs: list[DocumentedFigure] = []
        for df in raw.get("documented_figures", []):
            if not isinstance(df, dict):
                raise CalculationFreshnessError(
                    f"invalid documented figure in {calc_id}"
                )
            fid = str(df.get("figure_id", "")).strip()
            fpath = _safe_posix_path(df.get("path"), f"{calc_id}.{fid}.path")
            fmime = str(df.get("media_type", "")).strip()
            fcap = str(df.get("caption", "")).strip()
            fsha = str(df.get("sha256", "")).strip()
            if not fid or not fmime or not fcap or not HASH_PATTERN.fullmatch(fsha):
                raise CalculationFreshnessError(f"invalid documented figure in {fid}")
            doc_figs.append(DocumentedFigure(fid, fpath, fmime, fcap, fsha))

        # transitive_dependencies
        trans_deps = tuple(
            _safe_posix_path(item, f"{calc_id}.transitive_dependencies")
            for item in raw.get("transitive_dependencies", [])
        )
        if len(set(trans_deps)) != len(trans_deps) or trans_deps != tuple(
            sorted(trans_deps)
        ):
            raise CalculationFreshnessError(
                f"{calc_id} transitive_dependencies must be unique and sorted"
            )

        calc_entries.append(
            CalculationFreshnessEntry(
                calculation_id=calc_id,
                title=title,
                status=status,
                method_version=method_version,
                generator_script=gen_script,
                source_links=tuple(source_links),
                schema_links=tuple(schema_links),
                test_links=tuple(test_links),
                citation_links=tuple(citation_links),
                executable_examples=tuple(examples),
                documented_values=tuple(doc_vals),
                documented_tables=tuple(doc_tables),
                documented_figures=tuple(doc_figs),
                transitive_dependencies=trans_deps,
            )
        )

    # exemptions
    exemptions: list[ExpiringExemption] = []
    for exm in payload.get("exemptions", []):
        if not isinstance(exm, dict):
            raise ExemptionError("exemption must be an object")
        xid = str(exm.get("exemption_id", "")).strip()
        if not EXEMPTION_ID_PATTERN.fullmatch(xid):
            raise ExemptionError(f"invalid exemption_id: {xid!r}")
        cid = str(exm.get("calculation_id", "")).strip()
        if cid != "ALL" and not CALCULATION_ID_PATTERN.fullmatch(cid):
            raise ExemptionError(f"invalid calculation_id in exemption: {cid!r}")
        dtype = str(exm.get("drift_type", "")).strip()
        if dtype not in (
            "numerical-drift",
            "source-drift",
            "test-drift",
            "schema-drift",
            "missing-dependency",
        ):
            raise ExemptionError(f"invalid drift_type in {xid}: {dtype!r}")
        tpath = _safe_posix_path(exm.get("target_path"), f"{xid}.target_path")
        rat = str(exm.get("rationale", "")).strip()
        if len(rat) < 10 or any(p in rat.casefold() for p in PLACEHOLDER_TOKENS):
            raise ExemptionError(f"invalid or placeholder rationale in {xid}")
        rvw = tuple(str(r).strip() for r in exm.get("reviewed_by", []))
        if not rvw or not all(rvw):
            raise ExemptionError(f"reviewed_by must be non-empty in {xid}")
        created_at = _parse_iso_datetime(exm.get("created_at"), f"{xid}.created_at")
        expires_at = _parse_iso_datetime(exm.get("expires_at"), f"{xid}.expires_at")
        xstatus = str(exm.get("status", "")).strip()
        if xstatus not in ("active", "expired", "revoked"):
            raise ExemptionError(f"invalid status in {xid}: {xstatus!r}")
        exemptions.append(
            ExpiringExemption(
                xid, cid, dtype, tpath, rat, rvw, created_at, expires_at, xstatus
            )
        )

    return CalculationFreshnessManifest(
        schema_version=payload["schema_version"],
        manual_id=payload["manual_id"],
        release_status=payload["release_status"],
        owner_subepic=payload["owner_subepic"],
        calculations=tuple(calc_entries),
        exemptions=tuple(exemptions),
    )


def verify_expiring_exemptions(
    manifest: CalculationFreshnessManifest,
    current_time: datetime | None = None,
) -> int:
    """Verify that all exemptions are valid and unexpired; fail closed on expired or revoked."""
    now = current_time or datetime.now(UTC)
    active_count = 0
    for exm in manifest.exemptions:
        if exm.status == "revoked":
            raise ExemptionError(f"exemption {exm.exemption_id} is revoked")
        if exm.status == "expired" or now >= exm.expires_at:
            raise ExemptionError(
                f"exemption {exm.exemption_id} expired at {exm.expires_at.isoformat()} (current={now.isoformat()})"
            )
        if exm.status != "active":
            raise ExemptionError(
                f"exemption {exm.exemption_id} has invalid status {exm.status!r}"
            )
        active_count += 1
    return active_count


class TransitiveReverseImpactGate:
    """Evaluates the reverse dependency graph to identify impacted calculations and files."""

    def __init__(self, root: Path, manifest: CalculationFreshnessManifest):
        self.root = root.resolve()
        self.manifest = manifest
        self._target_to_calculations: dict[PurePosixPath, set[str]] = {}
        self._all_tracked_paths: set[PurePosixPath] = set()
        self._build_graph()

    def _build_graph(self) -> None:
        for calc in self.manifest.calculations:
            cid = calc.calculation_id
            # sources
            for sl in calc.source_links:
                self._target_to_calculations.setdefault(sl.path, set()).add(cid)
                self._all_tracked_paths.add(sl.path)
            # schemas
            for scl in calc.schema_links:
                self._target_to_calculations.setdefault(scl.path, set()).add(cid)
                self._all_tracked_paths.add(scl.path)
            # tests
            for tl in calc.test_links:
                test_file = PurePosixPath(tl.path.split("::", 1)[0])
                self._target_to_calculations.setdefault(test_file, set()).add(cid)
                self._all_tracked_paths.add(test_file)
            # generator
            self._target_to_calculations.setdefault(calc.generator_script, set()).add(
                cid
            )
            self._all_tracked_paths.add(calc.generator_script)
            # fixtures
            for ex in calc.executable_examples:
                self._target_to_calculations.setdefault(ex.fixture_path, set()).add(cid)
                self._all_tracked_paths.add(ex.fixture_path)
            # manual chapters
            for dv in calc.documented_values:
                self._target_to_calculations.setdefault(dv.manual_chapter, set()).add(
                    cid
                )
                self._all_tracked_paths.add(dv.manual_chapter)
            # transitive deps
            for td in calc.transitive_dependencies:
                self._target_to_calculations.setdefault(td, set()).add(cid)
                self._all_tracked_paths.add(td)

        # Add central governance files
        for p in (
            CALCULATION_REGISTRY_PATH,
            EXEMPLAR_COVERAGE_PATH,
            TEXTBOOK_REGISTRY_PATH,
            CHAPTER_CONTRACT_PATH,
            TOOLCHAIN_LOCK_PATH,
            ARTIFACT_MANIFEST_PATH,
            REFERENCES_PATH,
            FRESHNESS_PATH,
            FRESHNESS_SCHEMA_PATH,
        ):
            self._all_tracked_paths.add(p)

    @property
    def tracked_paths(self) -> frozenset[PurePosixPath]:
        return frozenset(self._all_tracked_paths)

    def check_file_integrity(self) -> None:
        """Ensure no referenced file in the graph was moved, renamed, or deleted."""
        for rel_path in sorted(self._all_tracked_paths):
            disk_path = self.root.joinpath(*rel_path.parts)
            if not disk_path.is_file():
                raise ReverseImpactError(
                    f"transitive reverse-impact failure: referenced file was moved or deleted: {rel_path.as_posix()}"
                )

    def find_impacted_calculations(
        self, changed_files: Sequence[str | PurePosixPath]
    ) -> set[str]:
        """Return the set of calculation IDs impacted by changes to files."""
        impacted: set[str] = set()
        for item in changed_files:
            posix = PurePosixPath(str(item).replace("\\", "/"))
            # If any central schema or registry changed, all calculations are impacted
            if posix in (
                FRESHNESS_SCHEMA_PATH,
                FRESHNESS_PATH,
                CALCULATION_REGISTRY_PATH,
                EXEMPLAR_COVERAGE_PATH,
            ):
                return {c.calculation_id for c in self.manifest.calculations}
            if posix in self._target_to_calculations:
                impacted.update(self._target_to_calculations[posix])
        return impacted


def execute_dplane_calculation(
    case: dict[str, Any],
    frame_id: str = "app_frame:x_target,y_up,z_right",
    root: Path | None = None,
) -> dict[str, Any]:
    """Execute the D-plane geometry calculation with given inputs."""
    _ensure_src_path(root)
    from shared.python.swing_sim.impact.dplane import analyze_dplane

    target = case.get("target")
    up = case.get("up")
    travel = [float(v) for v in case["travel_vector"]]
    face = [float(v) for v in case["face_normal"]]

    kwargs: dict[str, Any] = {"frame_id": frame_id}
    if target is not None:
        kwargs["target"] = [float(v) for v in target]
    if up is not None:
        kwargs["up"] = [float(v) for v in up]

    analysis = analyze_dplane(travel, face, **kwargs)

    return {
        "status": analysis.status.value,
        "travel_direction_unit": (
            list(analysis.travel_direction_unit)
            if analysis.travel_direction_unit is not None
            else None
        ),
        "face_normal_unit": (
            list(analysis.face_normal_unit)
            if analysis.face_normal_unit is not None
            else None
        ),
        "spin_loft_3d_deg": analysis.spin_loft_3d_deg,
        "planar_spin_loft_deg": analysis.planar_spin_loft_deg,
        "signed_planar_gap_deg": analysis.signed_planar_gap_deg,
        "spin_loft_residual_deg": analysis.spin_loft_residual_deg,
        "club_path_deg": analysis.club_path_deg,
        "attack_angle_deg": analysis.attack_angle_deg,
        "face_angle_deg": analysis.face_angle_deg,
        "dynamic_loft_deg": analysis.dynamic_loft_deg,
        "face_to_path_deg": analysis.face_to_path_deg,
        "dplane_normal_unit": (
            list(analysis.dplane_normal_unit)
            if analysis.dplane_normal_unit is not None
            else None
        ),
        "dplane_normal_azimuth_deg": analysis.dplane_normal_azimuth_deg,
        "dplane_tilt_deg": analysis.dplane_tilt_deg,
        "dplane_inclination_deg": analysis.dplane_inclination_deg,
        "ground_intersection_azimuth_deg": analysis.ground_intersection_azimuth_deg,
    }


def verify_calculation_freshness(
    root: Path,
    changed_files: Sequence[str | PurePosixPath] | None = None,
    current_time: datetime | None = None,
    tolerance_deg: float = 1e-9,
) -> FreshnessVerificationSummary:
    """Run full or diff-aware calculation freshness and reverse-impact verification."""
    repository_root = root.resolve()
    _ensure_src_path(repository_root)
    freshness_path = repository_root.joinpath(*FRESHNESS_PATH.parts)
    if not freshness_path.is_file():
        raise CalculationFreshnessError(
            f"calculation freshness manifest missing: {freshness_path}"
        )

    try:
        raw_manifest = json.loads(freshness_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalculationFreshnessError(
            f"unable to parse freshness JSON: {freshness_path}"
        ) from exc

    manifest = load_calculation_freshness(raw_manifest)

    # 1. Verify expiring exemptions
    active_exemptions = verify_expiring_exemptions(manifest, current_time)

    # 2. Build reverse-impact graph and verify file integrity
    impact_gate = TransitiveReverseImpactGate(repository_root, manifest)
    impact_gate.check_file_integrity()

    # Determine exemptions map
    exemption_map: dict[PurePosixPath, ExpiringExemption] = {
        ex.target_path: ex for ex in manifest.exemptions if ex.status == "active"
    }

    nominal_count = 0
    boundary_count = 0
    failure_count = 0
    documented_val_count = 0
    table_count = 0

    impacted_calc_ids = (
        set(impact_gate.find_impacted_calculations(changed_files))
        if changed_files is not None
        else None
    )

    for calc in manifest.calculations:
        if (
            impacted_calc_ids is not None
            and calc.calculation_id not in impacted_calc_ids
        ):
            continue
        # Verify source digests
        for sl in calc.source_links:
            actual_sha = sha256_file_lf(repository_root.joinpath(*sl.path.parts))
            if actual_sha != sl.sha256_lf:
                if (
                    sl.path not in exemption_map
                    or exemption_map[sl.path].drift_type != "source-drift"
                ):
                    raise DriftError(
                        f"source drift detected for {sl.path.as_posix()}: actual={actual_sha} expected={sl.sha256_lf}"
                    )

        # Verify schema digests
        for scl in calc.schema_links:
            actual_sha = sha256_file_lf(repository_root.joinpath(*scl.path.parts))
            if actual_sha != scl.sha256_lf:
                if (
                    scl.path not in exemption_map
                    or exemption_map[scl.path].drift_type != "schema-drift"
                ):
                    raise DriftError(
                        f"schema drift detected for {scl.path.as_posix()}: actual={actual_sha} expected={scl.sha256_lf}"
                    )

        # Verify test file digests
        for tl in calc.test_links:
            test_file = PurePosixPath(tl.path.split("::", 1)[0])
            actual_sha = sha256_file_lf(repository_root.joinpath(*test_file.parts))
            if actual_sha != tl.sha256_lf:
                if (
                    test_file not in exemption_map
                    or exemption_map[test_file].drift_type != "test-drift"
                ):
                    raise DriftError(
                        f"test drift detected for {test_file.as_posix()}: actual={actual_sha} expected={tl.sha256_lf}"
                    )

        # Verify and execute examples
        for ex in calc.executable_examples:
            if ex.category == "nominal":
                nominal_count += 1
            elif ex.category == "boundary":
                boundary_count += 1
            elif ex.category == "failure":
                failure_count += 1

            if ex.category in ("nominal", "boundary"):
                outcome = execute_dplane_calculation(ex.inputs, root=repository_root)
                outcome_digest = canonical_json_digest(outcome)
                if outcome_digest != ex.execution_digest:
                    if ex.fixture_path not in exemption_map:
                        raise DriftError(
                            f"numerical drift in example {ex.example_id}: "
                            f"actual_digest={outcome_digest} expected={ex.execution_digest}"
                        )

                # Validate expected outcome fields
                for field, exp_val in ex.expected_outcome.items():
                    act_val = outcome.get(field)
                    if exp_val is None:
                        if act_val is not None:
                            raise DriftError(
                                f"expected {field} to be null in {ex.example_id}, got {act_val}"
                            )
                    elif isinstance(exp_val, (int, float)):
                        if act_val is None or not math.isclose(
                            act_val, float(exp_val), abs_tol=tolerance_deg
                        ):
                            raise DriftError(
                                f"numerical mismatch for {field} in {ex.example_id}: "
                                f"got {act_val}, expected {exp_val}"
                            )
                    elif isinstance(exp_val, list):
                        if act_val is None or len(act_val) != len(exp_val):
                            raise DriftError(
                                f"vector length mismatch for {field} in {ex.example_id}"
                            )
                        for a, e in zip(act_val, exp_val, strict=True):
                            if not math.isclose(
                                float(a), float(e), abs_tol=tolerance_deg
                            ):
                                raise DriftError(
                                    f"vector value mismatch for {field} in {ex.example_id}: {act_val} != {exp_val}"
                                )
                    else:
                        if act_val != exp_val:
                            raise DriftError(
                                f"mismatch for {field} in {ex.example_id}: {act_val} != {exp_val}"
                            )

            elif ex.category == "failure":
                # Execute expecting failure
                err_spec = ex.expected_outcome
                err_type_name = err_spec.get("error_type", "ValueError")
                err_match = err_spec.get("error_match", "")
                failed_as_expected = False
                try:
                    execute_dplane_calculation(ex.inputs, root=repository_root)
                except Exception as exc:  # noqa: BLE001
                    if (
                        type(exc).__name__ == err_type_name
                        or issubclass(type(exc), ValueError)
                    ) and err_match in str(exc):
                        failed_as_expected = True
                    else:
                        raise DriftError(
                            f"failure example {ex.example_id} raised unexpected exception: {exc!r}"
                        ) from exc
                if not failed_as_expected:
                    raise DriftError(
                        f"failure example {ex.example_id} succeeded but was expected to raise {err_type_name}"
                    )

        documented_val_count += len(calc.documented_values)
        table_count += len(calc.documented_tables)

    return FreshnessVerificationSummary(
        calculation_count=len(manifest.calculations),
        nominal_count=nominal_count,
        boundary_count=boundary_count,
        failure_count=failure_count,
        documented_value_count=documented_val_count,
        table_count=table_count,
        active_exemption_count=active_exemptions,
        verified_at=current_time or datetime.now(UTC),
    )
