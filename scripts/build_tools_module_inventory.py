#!/usr/bin/env python3
"""Build the deterministic Tools module and calculation-candidate inventory."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from scripts.tools_module_inventory_contract import (
    AUTHORITY,
    MODULE_INVENTORY_SCHEMA_VERSION,
    RELEASE_STATUS,
    load_inventory,
)
from scripts.tools_module_inventory_extractors import (
    EXCLUDED_PARTS,
    LANGUAGES,
    TestIndex,
    build_test_index,
    discover_governed_paths,
    mapped_test_paths,
    normalized_bytes,
    public_surfaces,
)
from scripts.tools_module_inventory_storage import (
    check_projection,
    project_shards,
    write_projection,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "manuals" / "tools" / "manifests" / "module-inventory.json"
SCHEMA_PATH = "manuals/tools/schemas/module-inventory.schema.json"
GENERATOR_PATH = "scripts/build_tools_module_inventory.py"
CALCULATION_MARKERS = (
    "aerodynamic",
    "ball_flight",
    "calculation",
    "calculator",
    "calculus",
    "dynamics",
    "energy",
    "filter",
    "force",
    "friction",
    "geometry",
    "impact",
    "inertia",
    "kinematic",
    "optimization",
    "pendulum",
    "physics",
    "pressure_drop",
    "rotation",
    "signal",
    "simulation",
    "solver",
    "statistics",
    "thermodynamic",
    "torque",
    "trajectory",
    "transform",
)
MATH_IMPORT_PATTERN = re.compile(
    r"(?:from|import|use|require\s*\(|#include\s*[<\"])(?:numpy|scipy|math|nalgebra|ndarray|statistics|sympy|casadi)\b",
    re.IGNORECASE,
)
URL_PATTERN = re.compile(r"https?://[^\s)\]}>\"']+")
DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)
ADR_PATTERN = re.compile(r"ADR-\d{3}", re.IGNORECASE)
EQUATION_PATTERN = re.compile(
    r"(?:@eq-|equation[_-]id[\s:=]+)([A-Za-z][A-Za-z0-9_.-]+)", re.IGNORECASE
)
ROUTE_PATTERN = re.compile(
    r"(?:@(?:app|router)\.(?:get|post|put|patch|delete)\s*\(|"
    r"path\s*=\s*)[\"']([^\"']+)[\"']"
)
UNIT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:kg|g|mm|cm|m/s(?:\^2)?|rad/s|rad|deg|N(?:[ ·*\-/]?m)?|Pa|kPa|MPa|bar|J|W|Hz|rpm|ms|µs|s)(?![A-Za-z0-9_])"
)


def _domain(path: Path) -> tuple[str, str]:
    text = path.as_posix().lower()
    mappings = (
        (
            "sidekick/process_calculators",
            "process-calculators",
            "Process calculator maintainers",
        ),
        ("rate_of_closure", "rate-of-closure", "Rate of Closure maintainers"),
        ("signal_toolkit", "signal-processing", "Signal processing maintainers"),
        ("signal_processing", "signal-processing", "Signal processing maintainers"),
        ("swing_sim", "swing-simulation", "Swing simulation maintainers"),
        ("pendulum", "pendulum", "Pendulum maintainers"),
        ("rotation_converter", "rotation-converter", "Rotation converter maintainers"),
        ("p1am_control_system", "control-systems", "Control systems maintainers"),
        ("rust_core/", "native-kernels", "Native kernel maintainers"),
        ("data_processing", "data-processing", "Data processing maintainers"),
        ("media_processing", "media-processing", "Media processing maintainers"),
    )
    for marker, family, owner in mappings:
        if marker in text:
            return family, owner
    if path.parts[0].lower() in {"config", "schemas"}:
        return "repository-governance", "Repository governance maintainers"
    if path.parts[0].lower() in {"scripts", "shared_scripts"}:
        return "repository-tooling", "Repository tooling maintainers"
    return "shared-tools", "Tools module maintainers"


def _classification(path: Path, text: str) -> tuple[str, str]:
    normalized = path.as_posix().lower().replace("-", "_")
    markers = sorted(marker for marker in CALCULATION_MARKERS if marker in normalized)
    if markers:
        return "calculation", f"path-marker:{markers[0]}"
    if MATH_IMPORT_PATTERN.search(text):
        return "calculation", "scientific-library-import"
    return "non-calculation", "no-conservative-calculation-signal"


def _traceability(
    path: Path, text: str, surfaces: list[dict[str, object]], tests: list[str]
) -> dict[str, object]:
    explicit_adrs = {match.upper() for match in ADR_PATTERN.findall(text)}
    adr_paths: set[str] = set()
    if path.parts[0].lower() == "src":
        adr_paths.add("docs/adr/ADR-002-shared-library-module-structure.md")
    if surfaces:
        adr_paths.add("docs/adr/ADR-003-api-stability-policy.md")
    for name in explicit_adrs:
        matches = sorted((ROOT / "docs" / "adr").glob(f"{name}*.md"))
        adr_paths.update(match.relative_to(ROOT).as_posix() for match in matches)
    citations = sorted(set(URL_PATTERN.findall(text)) | set(DOI_PATTERN.findall(text)))
    units = sorted(set(UNIT_PATTERN.findall(text)))
    equations = sorted(set(EQUATION_PATTERN.findall(text)))
    routes = sorted(set(ROUTE_PATTERN.findall(text)))
    validation_paths = sorted(
        test
        for test in tests
        if any(
            marker in test.lower() for marker in ("contract", "scientific", "validat")
        )
    )
    return {
        "adr_paths": sorted(adr_paths),
        "artifact_sha256": [],
        "chapter_paths": [],
        "citation_refs": citations,
        "equation_refs": equations,
        "public_surfaces": surfaces,
        "public_routes": routes,
        "test_paths": tests,
        "unit_mentions": units,
        "validation_paths": validation_paths,
    }


def _states(classification: str, trace: dict[str, object]) -> dict[str, str]:
    calculation = classification == "calculation"
    return {
        "artifacts": "unmapped-pending-TOOLS-D7" if calculation else "not-applicable",
        "adrs": "mapped" if trace["adr_paths"] else "unavailable",
        "chapters": "unmapped-pending-TOOLS-D4",
        "citations": "mapped"
        if trace["citation_refs"]
        else ("unavailable" if calculation else "not-applicable"),
        "equation_pathway": "unmapped-pending-TOOLS-D4"
        if calculation
        else "not-applicable",
        "publication": "blocked",
        "public_surfaces": "mapped" if trace["public_surfaces"] else "unavailable",
        "routes": "mapped" if trace["public_routes"] else "not-applicable",
        "tests": "mapped"
        if trace["test_paths"]
        else ("unavailable" if calculation else "not-applicable"),
        "units": "mapped"
        if trace["unit_mentions"]
        else ("unavailable" if calculation else "not-applicable"),
        "validation": "mapped"
        if trace["validation_paths"]
        else ("unavailable" if calculation else "not-applicable"),
    }


def _identifier(path: Path) -> str:
    slug = re.sub(r"[^A-Z0-9]+", "-", path.as_posix().upper()).strip("-")
    digest = hashlib.sha256(path.as_posix().encode("utf-8")).hexdigest()[:12].upper()
    return f"TOOLS-MODULE-{slug}-{digest}"


def _entry(root: Path, path: Path, tests: TestIndex) -> dict[str, object]:
    normalized = normalized_bytes(root / path)
    try:
        text = normalized.decode("utf-8")
        decode_blocked = False
    except UnicodeDecodeError:
        text = normalized.decode("utf-8", errors="replace")
        decode_blocked = True
    classification, basis = _classification(path, text)
    surfaces, purpose, parse_blocked = public_surfaces(path, text)
    family_base, owner = _domain(path)
    mapped_tests = mapped_test_paths(path, tests)
    trace = _traceability(path, text, surfaces, mapped_tests)
    blocked = decode_blocked or parse_blocked
    risks = {"public-api"} if surfaces else set()
    if classification == "calculation":
        risks.update({"scientific-review-required", "unit-contract-review"})
        if not mapped_tests:
            risks.add("test-evidence-unmapped")
        if not trace["citation_refs"]:
            risks.add("source-provenance-unmapped")
    if path.as_posix().startswith(("src/shared/", "rust_core/")):
        risks.add("downstream-consumer-risk")
    if blocked:
        risks.add("parse-or-encoding-blocker")
    return {
        "authority_status": "blocked"
        if blocked
        else ("provisional" if classification == "calculation" else "not-applicable"),
        "bytes_lf": len(normalized),
        "classification": classification,
        "classification_basis": basis,
        "content_sha256_lf": hashlib.sha256(normalized).hexdigest(),
        "family": f"{family_base}-{classification}",
        "id": _identifier(path),
        "language": LANGUAGES[path.suffix.lower()],
        "maintainer": owner,
        "path": path.as_posix(),
        "purpose": purpose,
        "review_status": "blocked"
        if blocked
        else (
            "review-required"
            if classification == "calculation"
            else "inventory-baseline"
        ),
        "risk_tags": sorted(risks),
        "states": _states(classification, trace),
        "traceability": trace,
    }


def _summary(entries: list[dict[str, object]]) -> dict[str, object]:
    classifications = Counter(str(entry["classification"]) for entry in entries)
    authorities = Counter(str(entry["authority_status"]) for entry in entries)
    families = Counter(str(entry["family"]) for entry in entries)
    reviews = Counter(str(entry["review_status"]) for entry in entries)
    return {
        "authority_status_counts": dict(sorted(authorities.items())),
        "blocked_count": authorities["blocked"],
        "calculation_count": classifications["calculation"],
        "classification_counts": dict(sorted(classifications.items())),
        "family_counts": dict(sorted(families.items())),
        "module_count": len(entries),
        "non_calculation_count": classifications["non-calculation"],
        "provisional_count": authorities["provisional"],
        "review_status_counts": dict(sorted(reviews.items())),
    }


def build_inventory(root: Path = ROOT) -> dict[str, object]:
    """Build the complete payload and validate its public consumer contract."""
    paths = discover_governed_paths(root)
    test_index = build_test_index(root)
    entries = [_entry(root, path, test_index) for path in paths]
    tree_authority = "".join(
        f"{entry['path']}:{entry['content_sha256_lf']}\n" for entry in entries
    )
    families = []
    for family in sorted({str(entry["family"]) for entry in entries}):
        exemplar = next(entry for entry in entries if entry["family"] == family)
        families.append(
            {
                "classification": exemplar["classification"],
                "id": family,
                "maintainer": exemplar["maintainer"],
                "rationale": "Deterministic repository-domain and conservative calculation-signal classification.",
            }
        )
    payload: dict[str, object] = {
        "authority": AUTHORITY,
        "blockers": [
            {
                "id": "TOOLS-D4-exemplar-pathways-required",
                "owner": "Tools documentation epic #4707",
                "resolution": "Register exemplar calculation IDs and map equations, chapters, tests, sources, units, limits, and approval evidence under TOOLS-D4 through TOOLS-D9.",
            }
        ],
        "entries": entries,
        "families": families,
        "hash_contract": {
            "algorithm": "sha256",
            "line_endings": "CRLF-and-CR-normalized-to-LF",
            "tree_encoding": "path-colon-content_sha256_lf-newline",
        },
        "producer": {"generator_path": GENERATOR_PATH, "schema_path": SCHEMA_PATH},
        "release_status": RELEASE_STATUS,
        "schema_version": MODULE_INVENTORY_SCHEMA_VERSION,
        "scope": {
            "discovery": "tracked-implementation-and-governed-configuration-modules",
            "exclusions": sorted(EXCLUDED_PARTS),
            "roots": sorted(
                ["repository-wide tracked implementation", "config", "schemas"]
            ),
            "suffixes": sorted(LANGUAGES),
        },
        "source_tree_sha256": hashlib.sha256(
            tree_authority.encode("utf-8")
        ).hexdigest(),
        "summary": _summary(entries),
    }
    load_inventory(payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """Write or check the deterministic registry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="fail if the registry is missing or stale"
    )
    args = parser.parse_args(argv)
    index, shards = project_shards(build_inventory(ROOT))
    if args.check:
        diagnostic = check_projection(ROOT, OUTPUT_PATH, index, shards)
        if diagnostic is not None:
            print(f"ERROR: {diagnostic}", file=sys.stderr)
            return 1
        return 0
    write_projection(ROOT, OUTPUT_PATH, index, shards)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
