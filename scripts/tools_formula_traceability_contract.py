"""Bidirectional formula-family traceability for governed Tools chapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from scripts.tools_formula_traceability_resolvers import (
    ChapterTraceabilityContext,
    FormulaTraceabilityError,
    FormulaResolverContext,
    TraceabilityDocuments,
    assert_public_symbol,
    assert_test_target,
    load_authority_document,
    safe_repository_path,
)
from scripts.tools_textbook_chapter_contract import (
    CONTRACT_VERSION,
    TextbookChapterError,
    load_chapter_contract,
    load_chapter_registry,
)

CALCULATION_REGISTRY_PATH = PurePosixPath("manuals/tools/calculation-registry.json")
CHAPTER_CONTRACT_PATH = PurePosixPath("manuals/tools/textbook-chapter-contract.json")
CHAPTER_REGISTRY_PATH = PurePosixPath("manuals/tools/textbook-chapters.json")
EXEMPLAR_COVERAGE_PATH = PurePosixPath("manuals/tools/exemplar-coverage.json")
ARTIFACT_MANIFEST_PATH = PurePosixPath("manuals/tools/manifests/artifacts.json")
PLACEHOLDER_TOKENS = frozenset({"fixme", "tbd", "todo", "placeholder"})


@dataclass(frozen=True)
class FormulaTraceabilitySummary:
    """Resolved formula-family counts and stable identifiers."""

    chapter_count: int
    family_count: int
    formula_ids: tuple[str, ...]
    claim_ids: tuple[str, ...]


def load_traceability_documents(repository_root: Path) -> TraceabilityDocuments:
    """Load all owning authorities from normalized repository paths."""
    root = repository_root.resolve()

    def load(relative: PurePosixPath) -> dict[str, object]:
        return load_authority_document(root.joinpath(*relative.parts))

    return TraceabilityDocuments(
        chapter_contract=load(CHAPTER_CONTRACT_PATH),
        chapter_registry=load(CHAPTER_REGISTRY_PATH),
        calculation_registry=load(CALCULATION_REGISTRY_PATH),
        exemplar_coverage=load(EXEMPLAR_COVERAGE_PATH),
        artifact_manifest=load(ARTIFACT_MANIFEST_PATH),
    )


def _object(
    value: object,
    label: str,
    fields: set[str] | None = None,
) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise FormulaTraceabilityError(f"{label} must be an object")
    document = cast(dict[str, object], value)
    if fields is not None and set(document) != fields:
        raise FormulaTraceabilityError(f"{label} fields differ")
    return document


def _array(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise FormulaTraceabilityError(f"{label} must be an array")
    return cast(list[object], value)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise FormulaTraceabilityError(f"{label} must be a trimmed non-empty string")
    if any(token in value.casefold() for token in PLACEHOLDER_TOKENS):
        raise FormulaTraceabilityError(f"{label} contains a placeholder")
    return value


def _texts(value: object, label: str) -> tuple[str, ...]:
    result = tuple(_text(item, label) for item in _array(value, label))
    if not result:
        raise FormulaTraceabilityError(f"{label} must not be empty")
    if len(set(result)) != len(result) or result != tuple(sorted(result)):
        raise FormulaTraceabilityError(f"{label} must be sorted and unique")
    return result


def _formula_symbols(formula: dict[str, object]) -> tuple[tuple[str, str], ...]:
    symbols: list[tuple[str, str]] = []
    for raw in _array(formula["implementation_symbols"], "implementation symbols"):
        item = _object(raw, "implementation symbol", {"path", "symbol"})
        symbols.append(
            (
                safe_repository_path(item["path"], "implementation path").as_posix(),
                _text(item["symbol"], "public symbol"),
            )
        )
    if not symbols:
        raise FormulaTraceabilityError("implementation symbols must not be empty")
    result = tuple(symbols)
    if len(set(result)) != len(result) or result != tuple(sorted(result)):
        raise FormulaTraceabilityError("implementation symbols must be sorted and unique")
    return result


def _calculation_map(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    calculations = _array(payload.get("calculations"), "calculations")
    result: dict[str, dict[str, object]] = {}
    for raw in calculations:
        item = _object(raw, "calculation")
        identifier = _text(item.get("calculation_id"), "calculation ID")
        if identifier in result:
            raise FormulaTraceabilityError("duplicate calculation ID")
        result[identifier] = item
    return result


def _chapter_examples(payload: dict[str, object]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for raw in _array(payload.get("entries"), "exemplar entries"):
        item = _object(raw, "exemplar")
        chapter_id = item.get("chapter_id")
        if chapter_id is None:
            continue
        identifier = _text(chapter_id, "exemplar chapter ID")
        examples = {
            _text(_object(example, "worked example").get("example_id"), "example ID")
            for example in _array(item.get("worked_examples"), "worked examples")
        }
        result.setdefault(identifier, set()).update(examples)
    return result


def _artifacts(root: Path, payload: dict[str, object]) -> set[str]:
    identifiers: set[str] = set()
    for raw in _array(payload.get("artifacts"), "rendered artifacts"):
        item = _object(raw, "rendered artifact")
        path = safe_repository_path(item.get("path"), "rendered artifact path")
        if not root.joinpath(*path.parts).is_file():
            raise FormulaTraceabilityError(f"rendered artifact is missing: {path}")
        identifiers.add(path.name)
    return identifiers


def _authority_sets(calculation: dict[str, object]) -> dict[str, set[object]]:
    implementation = _object(calculation.get("implementation"), "implementation")
    symbols = {
        (
            safe_repository_path(
                _object(raw, "symbol").get("path"), "symbol path"
            ).as_posix(),
            _text(_object(raw, "symbol").get("symbol"), "symbol name"),
        )
        for raw in _array(implementation.get("symbols"), "implementation symbols")
    }
    return {
        "formula_ids": {
            _text(_object(raw, "equation").get("equation_id"), "equation ID")
            for raw in _array(calculation.get("equations"), "equations")
        },
        "formula_anchors": {
            (
                _text(_object(raw, "equation").get("equation_id"), "equation ID"),
                _text(_object(raw, "equation").get("manual_anchor"), "manual anchor"),
            )
            for raw in _array(calculation.get("equations"), "equations")
        },
        "symbols": symbols,
        "tests": {
            _text(_object(raw, "test").get("path"), "test path")
            for raw in _array(calculation.get("tests"), "tests")
        },
        "citations": {
            _text(_object(raw, "source").get("source_id"), "source ID")
            for raw in _array(calculation.get("sources"), "sources")
        },
        "artifacts": set(
            _texts(
                _object(calculation.get("manual"), "manual").get("artifact_ids"),
                "artifact IDs",
            )
        ),
    }


def _verify_formula(
    raw: object,
    context: FormulaResolverContext,
) -> dict[str, set[object]]:
    fields = {
        "formula_id",
        "manual_anchor",
        "implementation_symbols",
        "verification_tests",
        "citation_ids",
        "worked_example_ids",
        "claim_ids",
        "rendered_artifact_ids",
    }
    formula = _object(raw, "formula", fields)
    formula_id = _text(formula["formula_id"], "formula ID")
    anchor = _text(formula["manual_anchor"], "manual anchor")
    symbols = _formula_symbols(formula)
    tests = _texts(formula["verification_tests"], "verification tests")
    citations = _texts(formula["citation_ids"], "citation IDs")
    worked = _texts(formula["worked_example_ids"], "worked example IDs")
    claim_ids = _texts(formula["claim_ids"], "claim IDs")
    artifacts = _texts(formula["rendered_artifact_ids"], "rendered artifact IDs")
    if anchor not in context.anchors:
        raise FormulaTraceabilityError(f"manual anchor is missing: {anchor}")
    for path, symbol in symbols:
        assert_public_symbol(context.root, path, symbol)
    for target in tests:
        assert_test_target(context.root, target)
    if not set(worked).issubset(context.examples):
        raise FormulaTraceabilityError("worked example is missing")
    if not set(claim_ids).issubset(context.claims):
        raise FormulaTraceabilityError("claim is missing")
    if not set(artifacts).issubset(context.rendered):
        raise FormulaTraceabilityError("rendered artifact is missing")
    return {
        "formula_ids": {formula_id},
        "formula_anchors": {(formula_id, anchor)},
        "symbols": set(symbols),
        "tests": set(tests),
        "citations": set(citations),
        "examples": set(worked),
        "claims": set(claim_ids),
        "artifacts": set(artifacts),
    }


def _merge_edges(target: dict[str, set[object]], source: dict[str, set[object]]) -> None:
    for name, values in source.items():
        target.setdefault(name, set()).update(values)


def _verify_family(
    raw: object,
    context: ChapterTraceabilityContext,
) -> tuple[str, dict[str, set[object]]]:
    fields = {
        "family_id",
        "calculation_id",
        "assumptions",
        "dimensions",
        "domains",
        "numerical_method",
        "uncertainty_and_limitations",
        "claims",
        "formulas",
    }
    family = _object(raw, "derivation family", fields)
    family_id = _text(family["family_id"], "family ID")
    calculation_id = _text(family["calculation_id"], "calculation ID")
    for field in (
        "assumptions",
        "dimensions",
        "domains",
        "uncertainty_and_limitations",
    ):
        _texts(family[field], field.replace("_", " "))
    _text(family["numerical_method"], "numerical method")
    claims: set[str] = set()
    anchors: set[str] = set()
    raw_claims = _array(family["claims"], "claims")
    for raw_claim in raw_claims:
        claim = _object(
            raw_claim,
            "claim",
            {"claim_id", "statement", "evidence_class", "authority_status", "manual_anchor"},
        )
        claim_id = _text(claim["claim_id"], "claim ID")
        _text(claim["statement"], "claim statement")
        if _text(claim["evidence_class"], "claim evidence class") != "model-conditioned":
            raise FormulaTraceabilityError("claim evidence class is unsupported")
        if _text(claim["authority_status"], "claim authority status") != "verified-unapproved":
            raise FormulaTraceabilityError("claim authority status is unsupported")
        anchor = _text(claim["manual_anchor"], "claim manual anchor")
        if f"{{{anchor}}}" not in context.chapter_text:
            raise FormulaTraceabilityError(f"claim anchor is missing: {anchor}")
        claims.add(claim_id)
        anchors.add(anchor)
    if not claims:
        raise FormulaTraceabilityError("claims must not be empty")
    if len(claims) != len(raw_claims):
        raise FormulaTraceabilityError("claim IDs must be unique")
    formula_context = FormulaResolverContext(
        root=context.root,
        claims=frozenset(claims),
        anchors=frozenset(anchors),
        examples=context.examples,
        rendered=context.rendered,
    )
    edges: dict[str, set[object]] = {}
    raw_formulas = _array(family["formulas"], "formulas")
    for raw_formula in raw_formulas:
        _merge_edges(
            edges,
            _verify_formula(raw_formula, formula_context),
        )
    if not edges.get("formula_ids"):
        raise FormulaTraceabilityError("formulas must not be empty")
    if len(edges["formula_ids"]) != len(raw_formulas):
        raise FormulaTraceabilityError("formula IDs must be unique")
    if edges.get("claims") != claims:
        raise FormulaTraceabilityError("claim IDs differ across formula mappings")
    edges["calculation_ids"] = {calculation_id}
    edges["family_ids"] = {family_id}
    return calculation_id, edges


def verify_formula_traceability(
    repository_root: Path,
    documents: TraceabilityDocuments | None = None,
) -> FormulaTraceabilitySummary:
    """Resolve every formula edge in both directions or fail closed."""
    root = repository_root.resolve()
    docs = documents or load_traceability_documents(root)
    try:
        contract = load_chapter_contract(docs.chapter_contract)
        registry = load_chapter_registry(docs.chapter_registry, contract)
    except TextbookChapterError as error:
        raise FormulaTraceabilityError(str(error)) from error
    if contract.schema_version != CONTRACT_VERSION:
        raise FormulaTraceabilityError("chapter contract version differs")
    calculations = _calculation_map(docs.calculation_registry)
    chapter_examples = _chapter_examples(docs.exemplar_coverage)
    rendered = _artifacts(root, docs.artifact_manifest)
    all_edges: dict[str, set[object]] = {}
    family_count = 0
    for chapter in registry.chapters:
        chapter_text = root.joinpath(*chapter.path.parts).read_text(encoding="utf-8")
        context = ChapterTraceabilityContext(
            root=root,
            chapter_text=chapter_text,
            examples=frozenset(chapter_examples.get(chapter.chapter_id, set())),
            rendered=frozenset(rendered),
        )
        chapter_edges: dict[str, set[object]] = {}
        family_ids: set[object] = set()
        calculation_ids: set[object] = set()
        for raw_family in chapter.derivation_families:
            calculation_id, family_edges = _verify_family(raw_family, context)
            if family_edges["family_ids"] & family_ids:
                raise FormulaTraceabilityError("derivation family IDs must be unique")
            if calculation_id in calculation_ids:
                raise FormulaTraceabilityError(
                    "each calculation must own exactly one derivation family"
                )
            family_ids.update(family_edges["family_ids"])
            calculation_ids.add(calculation_id)
            authority = calculations.get(calculation_id)
            if authority is None:
                raise FormulaTraceabilityError("calculation is missing")
            expected = _authority_sets(authority)
            for name in (
                "formula_ids",
                "formula_anchors",
                "symbols",
                "tests",
                "citations",
                "artifacts",
            ):
                if family_edges.get(name, set()) != expected[name]:
                    if name == "citations":
                        raise FormulaTraceabilityError("citation is missing")
                    labels = {
                        "formula_ids": "formula IDs",
                        "formula_anchors": "formula anchors",
                    }
                    label = labels.get(name, name)
                    raise FormulaTraceabilityError(f"{label} differ across authorities")
            expected_examples = chapter_examples.get(chapter.chapter_id, set())
            if family_edges.get("examples", set()) != expected_examples:
                raise FormulaTraceabilityError("worked example IDs differ")
            _merge_edges(chapter_edges, family_edges)
            family_count += 1
        if chapter_edges.get("calculation_ids", set()) != set(chapter.calculation_ids):
            raise FormulaTraceabilityError("calculation IDs differ across derivation families")
        _merge_edges(all_edges, chapter_edges)
    return FormulaTraceabilitySummary(
        chapter_count=len(registry.chapters),
        family_count=family_count,
        formula_ids=tuple(sorted(cast(set[str], all_edges.get("formula_ids", set())))),
        claim_ids=tuple(sorted(cast(set[str], all_edges.get("claims", set())))),
    )
