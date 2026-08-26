#!/usr/bin/env python3
"""Fail closed when Tools design-manual authorities drift."""

from __future__ import annotations

import json
import sys
from pathlib import Path, PurePosixPath
from typing import cast

from scripts.design_manual_contract import (
    EXPECTED_CONTRACTS,
    EXPECTED_POLICY_FIELDS,
    GENERATED_SUFFIXES,
    IMPACTED_PATHS,
    REQUIRED_EVIDENCE,
    REQUIRED_FORMATS,
    REQUIRED_UPDATE_FILES,
    DesignManualGovernanceError,
    DesignManualGovernanceSummary,
    is_valid_revision,
)
from scripts.render_tools_design_manual import check_manual
from scripts.tools_formula_traceability_contract import (
    FormulaTraceabilityError,
    verify_formula_traceability,
)
from scripts.tools_exemplar_contract import (
    ExemplarContractError,
    verify_exemplar_repository,
)
from scripts.tools_module_inventory_contract import (
    ToolsModuleInventoryError,
)
from scripts.tools_module_inventory_storage import read_inventory
from scripts.tools_textbook_chapter_contract import TextbookChapterError
from scripts.tools_textbook_chapter_lint import verify_textbook_chapters

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = PurePosixPath("config/design_manual_governance.json")


def _object(value: object, label: str, fields: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise DesignManualGovernanceError(f"{label} must be an object")
    result = cast(dict[str, object], value)
    actual = set(result)
    if actual != fields:
        raise DesignManualGovernanceError(
            f"{label} fields differ: missing={sorted(fields - actual)}, "
            f"extra={sorted(actual - fields)}"
        )
    return result


def _array(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise DesignManualGovernanceError(f"{label} must be an array")
    return cast(list[object], value)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DesignManualGovernanceError(f"{label} must be a non-empty string")
    return value


def _equal(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise DesignManualGovernanceError(
            f"{label} must be {expected!r}; got {value!r}"
        )


def _safe_path(value: object, label: str) -> PurePosixPath:
    text = _text(value, label)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in text
        or path.as_posix() != text
    ):
        raise DesignManualGovernanceError(f"{label} must be a normalized relative path")
    return path


def _verify_source(policy: dict[str, object]) -> tuple[str, PurePosixPath]:
    source = _object(
        policy["canonical_source"],
        "canonical source",
        {
            "manual_id",
            "format",
            "path",
            "repository_role",
            "existing_documentation_relationship",
        },
    )
    _equal(source["format"], "qmd", "canonical source format")
    _equal(
        source["repository_role"],
        "shared-calculation-and-interchange-authority",
        "repository role",
    )
    _equal(
        source["existing_documentation_relationship"],
        "separate-products-not-manual-authority",
        "existing documentation relationship",
    )
    manual_id = _text(source["manual_id"], "manual ID")
    source_path = _safe_path(source["path"], "source path")
    _equal(manual_id, "tools", "manual ID")
    _equal(source_path, PurePosixPath("manuals/tools"), "source path")
    return manual_id, source_path


def _verify_outputs(policy: dict[str, object]) -> None:
    outputs = _object(
        policy["generated_outputs"],
        "generated outputs",
        {"editable", "required_release_formats", "current_artifact_status"},
    )
    if outputs["editable"] is not False:
        raise DesignManualGovernanceError("generated outputs must not be editable")
    _equal(outputs["required_release_formats"], REQUIRED_FORMATS, "release formats")
    _equal(
        outputs["current_artifact_status"],
        "generated-unapproved",
        "artifact status",
    )


def _verify_renderer(policy: dict[str, object]) -> None:
    renderer = _object(
        policy["renderer"],
        "renderer",
        {
            "owner_subepic",
            "status",
            "toolchain_lock",
            "artifact_manifest",
            "toolchain_schema",
            "artifact_schema",
            "semantic_contract",
            "reference_docx",
            "figure_files",
            "reproducibility",
            "review_boundary",
        },
    )
    _equal(renderer["owner_subepic"], 4712, "renderer owner")
    _equal(renderer["status"], "qualified-generated-unapproved", "renderer status")
    expected_paths = {
        "toolchain_lock": "manuals/tools/toolchain-lock.json",
        "artifact_manifest": "manuals/tools/manifests/artifacts.json",
        "toolchain_schema": "manuals/tools/schemas/toolchain-lock.schema.json",
        "artifact_schema": "manuals/tools/schemas/artifact-manifest.schema.json",
        "semantic_contract": "manuals/tools/semantic-contract.json",
        "reference_docx": "manuals/tools/styles/tools-reference.docx",
    }
    for field, expected in expected_paths.items():
        _equal(_safe_path(renderer[field], field), PurePosixPath(expected), field)
    figures = [
        _safe_path(item, "renderer figure")
        for item in _array(renderer["figure_files"], "renderer figures")
    ]
    _equal(
        figures,
        [PurePosixPath("manuals/tools/figures/render-pipeline.png")],
        "renderer figures",
    )
    _equal(renderer["reproducibility"], "byte-and-semantic", "reproducibility")
    _equal(
        renderer["review_boundary"],
        "page-accessibility-publication-human-review-pending-TOOLS-D7-D8",
        "renderer review boundary",
    )


def _verify_freshness(policy: dict[str, object]) -> None:
    freshness = _object(
        policy["freshness"],
        "freshness",
        {"enforcement", "current_gate", "impacted_paths", "exemptions"},
    )
    _equal(freshness["enforcement"], "release-blocking", "freshness enforcement")
    _equal(freshness["current_gate"], "blocked-pending-TOOLS-D6", "freshness gate")
    _equal(freshness["exemptions"], "structured-owned-expiring-only", "exemptions")
    paths = [
        _safe_path(item, "impacted path")
        for item in _array(freshness["impacted_paths"], "impacted paths")
    ]
    _equal(paths, list(map(PurePosixPath, IMPACTED_PATHS)), "impacted paths")


def _verify_chapter_contract(policy: dict[str, object]) -> None:
    chapter_contract = _object(
        policy["chapter_contract"],
        "chapter contract",
        {
            "owner_subepic",
            "status",
            "contract",
            "registry",
            "contract_schema",
            "registry_schema",
            "linter",
            "formula_traceability_checker",
            "registered_chapter_count",
            "next_owner",
        },
    )
    _equal(chapter_contract["owner_subepic"], 4717, "chapter contract owner")
    _equal(
        chapter_contract["status"],
        "qualified-one-exemplar-full-family-traceability-generated-unapproved",
        "chapter contract status",
    )
    expected_paths = {
        "contract": "manuals/tools/textbook-chapter-contract.json",
        "registry": "manuals/tools/textbook-chapters.json",
        "contract_schema": (
            "manuals/tools/schemas/textbook-chapter-contract.schema.json"
        ),
        "registry_schema": (
            "manuals/tools/schemas/textbook-chapter-registry.schema.json"
        ),
        "linter": "scripts/lint_tools_textbook_chapters.py",
        "formula_traceability_checker": (
            "scripts/tools_formula_traceability_contract.py"
        ),
    }
    for field, expected in expected_paths.items():
        _equal(
            _safe_path(chapter_contract[field], field),
            PurePosixPath(expected),
            field,
        )
    _equal(
        chapter_contract["registered_chapter_count"],
        1,
        "registered chapter count",
    )
    _equal(chapter_contract["next_owner"], "TOOLS-D5", "chapter next owner")


def _verify_exemplar_contract(policy: dict[str, object]) -> None:
    exemplar = _object(
        policy["exemplar_contract"],
        "exemplar contract",
        {
            "owner_subepic",
            "status",
            "coverage",
            "schema",
            "checker",
            "verified_exemplar_count",
            "blocked_exemplar_count",
            "next_owner",
        },
    )
    _equal(exemplar["owner_subepic"], 4720, "exemplar owner")
    _equal(exemplar["status"], "qualified-generated-unapproved", "exemplar status")
    for field, expected in {
        "coverage": "manuals/tools/exemplar-coverage.json",
        "schema": "manuals/tools/schemas/exemplar-coverage.schema.json",
        "checker": "scripts/check_tools_exemplars.py",
    }.items():
        _equal(_safe_path(exemplar[field], field), PurePosixPath(expected), field)
    _equal(exemplar["verified_exemplar_count"], 1, "verified exemplar count")
    _equal(exemplar["blocked_exemplar_count"], 1, "blocked exemplar count")
    _equal(exemplar["next_owner"], "TOOLS-D5", "exemplar next owner")


def _verify_publication(policy: dict[str, object]) -> bool:
    publication = _object(
        policy["publication"],
        "publication",
        {
            "default",
            "public_projection_allowed",
            "catalog_repository",
            "projection_manifest_path",
            "current_approval",
            "required_evidence",
        },
    )
    _equal(publication["default"], "deny-until-approved", "publication default")
    allowed = publication["public_projection_allowed"]
    if allowed is not False:
        raise DesignManualGovernanceError("public projection must remain blocked")
    _equal(
        publication["catalog_repository"],
        EXPECTED_CONTRACTS["owner_repository"],
        "catalog",
    )
    _equal(
        _safe_path(publication["projection_manifest_path"], "projection manifest"),
        PurePosixPath("manuals/tools/publication-projection.json"),
        "projection manifest",
    )
    _equal(
        publication["current_approval"],
        "blocked-pending-TOOLS-D4-approvals-and-D5-through-D8",
        "publication approval",
    )
    evidence = [
        _text(item, "evidence item")
        for item in _array(publication["required_evidence"], "required evidence")
    ]
    _equal(evidence, REQUIRED_EVIDENCE, "required evidence")
    return False


def _verify_quality_license_git(policy: dict[str, object]) -> None:
    quality = _object(
        policy["quality"],
        "quality",
        {"python_version", "python_formatter", "principles"},
    )
    _equal(quality["python_version"], "3.11", "Python version")
    _equal(quality["python_formatter"], "ruff format", "python formatter")
    _equal(quality["principles"], ["TDD", "DbC", "DRY", "LoD"], "principles")
    license_boundary = _object(
        policy["license_boundary"],
        "license boundary",
        {"repository_license", "private_content_allowed", "source_license_required"},
    )
    _equal(license_boundary["repository_license"], "MIT", "repository license")
    _equal(license_boundary["private_content_allowed"], False, "private content")
    _equal(license_boundary["source_license_required"], True, "source license")
    git = _object(
        policy["git"],
        "git",
        {"default_branch", "pull_request_target", "direct_push", "auto_merge"},
    )
    _equal(git["default_branch"], "main", "default branch")
    _equal(git["pull_request_target"], "main", "pull request target")
    _equal(git["direct_push"], False, "direct push")
    _equal(git["auto_merge"], False, "auto-merge")


def verify_governance_policy(policy: object) -> tuple[str, PurePosixPath, bool]:
    """Validate producer, contract, artifact, freshness, and release boundaries."""
    document = _object(policy, "governance policy", EXPECTED_POLICY_FIELDS)
    _equal(
        document["schema_version"],
        "tools/design-manual-governance/1.5.0",
        "schema version",
    )
    program = _object(
        document["program"], "program", {"epic", "current_subepic", "next_subepic"}
    )
    _equal(
        program,
        {"epic": 4707, "current_subepic": 4720, "next_subepic": 4722},
        "program",
    )
    manual_id, source_path = _verify_source(document)
    _equal(document["contracts"], EXPECTED_CONTRACTS, "shared contracts")
    inventory = _object(
        document["calculation_inventory"],
        "inventory",
        {
            "path",
            "owner_subepic",
            "module_manifest",
            "module_schema",
            "module_shard_schema",
            "current_status",
        },
    )
    _equal(
        _safe_path(inventory["path"], "inventory path"),
        PurePosixPath("manuals/tools/calculation-registry.json"),
        "inventory path",
    )
    _equal(inventory["owner_subepic"], 4711, "inventory owner")
    _equal(
        _safe_path(inventory["module_manifest"], "module manifest"),
        PurePosixPath("manuals/tools/manifests/module-inventory.json"),
        "module manifest",
    )
    _equal(
        _safe_path(inventory["module_schema"], "module schema"),
        PurePosixPath("manuals/tools/schemas/module-inventory.schema.json"),
        "module schema",
    )
    _equal(
        _safe_path(inventory["module_shard_schema"], "module shard schema"),
        PurePosixPath("manuals/tools/schemas/module-inventory-shard.schema.json"),
        "module shard schema",
    )
    _equal(
        inventory["current_status"],
        "provisional-module-baseline-one-exemplar-qualified-pending-TOOLS-D5",
        "inventory status",
    )
    _verify_outputs(document)
    _verify_renderer(document)
    _verify_chapter_contract(document)
    _verify_exemplar_contract(document)
    _verify_freshness(document)
    allowed = _verify_publication(document)
    _verify_quality_license_git(document)
    _object(
        document["agent_context"],
        "agent context",
        {"required_update_files", "required_gate"},
    )
    return manual_id, source_path, allowed


def verify_calculation_registry(registry: object) -> int:
    """Validate the shared registry envelope without copying its owned schema."""
    document = _object(
        registry,
        "calculation registry",
        {
            "schema_version",
            "manual_id",
            "repository",
            "release_status",
            "inventory_commit",
            "blockers",
            "calculations",
        },
    )
    _equal(document["schema_version"], "1.0.0", "registry schema version")
    _equal(document["manual_id"], "tools", "registry manual ID")
    _equal(document["repository"], "D-sorganization/Tools", "repository")
    calculations = _array(document["calculations"], "calculations")
    blockers = _array(document["blockers"], "blockers")
    status = _text(document["release_status"], "release status")
    if status == "approved":
        if (
            not calculations
            or blockers
            or not is_valid_revision(document["inventory_commit"])
        ):
            raise DesignManualGovernanceError(
                "approved registry requires calculations, immutable commit, and no blockers"
            )
    elif status == "provisional":
        revision = document["inventory_commit"]
        if not blockers or (calculations and not is_valid_revision(revision)):
            raise DesignManualGovernanceError(
                "provisional calculations require an immutable inventory commit and blockers"
            )
        if not calculations and revision is not None:
            raise DesignManualGovernanceError(
                "empty provisional registry requires a null inventory commit"
            )
    else:
        raise DesignManualGovernanceError("release status is unsupported")
    for blocker in blockers:
        item = _object(blocker, "registry blocker", {"id", "owner", "resolution"})
        for field, value in item.items():
            _text(value, f"blocker {field}")
    calculation_ids: list[str] = []
    for calculation in calculations:
        item = cast(dict[str, object], calculation)
        calculation_id = _text(item.get("calculation_id"), "calculation ID")
        approval = cast(dict[str, object], item.get("approval"))
        approval_state = _text(approval.get("state"), "calculation approval state")
        if status == "approved" and approval_state != "approved":
            raise DesignManualGovernanceError(
                "approved registry requires every calculation approval"
            )
        if status == "provisional" and approval_state == "approved":
            raise DesignManualGovernanceError(
                "provisional registry cannot contain an approved calculation"
            )
        calculation_ids.append(calculation_id)
    if calculation_ids != sorted(set(calculation_ids)):
        raise DesignManualGovernanceError("calculation IDs must be sorted and unique")
    return len(calculations)


def _load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_manual_tree(root: Path, source_path: PurePosixPath) -> int:
    manual_root = root.joinpath(*source_path.parts)
    if not manual_root.is_dir():
        raise DesignManualGovernanceError(f"canonical source is missing: {source_path}")
    generated = sorted(
        path.relative_to(root).as_posix()
        for path in manual_root.rglob("*")
        if path.is_file() and path.suffix.lower() in GENERATED_SUFFIXES
    )
    expected_generated = [
        "manuals/tools/dist/tools-engineering-design-manual.docx",
        "manuals/tools/dist/tools-engineering-design-manual.html",
        "manuals/tools/dist/tools-engineering-design-manual.pdf",
        "manuals/tools/dist/tools-engineering-design-manual.tex",
        "manuals/tools/styles/tools-header.tex",
        "manuals/tools/styles/tools-reference.docx",
    ]
    if generated != expected_generated:
        raise DesignManualGovernanceError(
            f"generated/manual style artifact set differs: {generated}"
        )
    qmd_paths = sorted(manual_root.rglob("*.qmd"))
    if not qmd_paths:
        raise DesignManualGovernanceError("canonical manual must contain QMD source")
    return len(qmd_paths)


def _verify_context(root: Path, policy: dict[str, object]) -> None:
    context = _object(
        policy["agent_context"],
        "agent context",
        {"required_update_files", "required_gate"},
    )
    paths = [
        _safe_path(item, "required update file")
        for item in _array(context["required_update_files"], "required update files")
    ]
    _equal(
        paths, list(map(PurePosixPath, REQUIRED_UPDATE_FILES)), "required update files"
    )
    for path in paths:
        if not root.joinpath(*path.parts).is_file():
            raise DesignManualGovernanceError(
                f"required update file is missing: {path}"
            )
    _equal(
        context["required_gate"],
        (
            "python -m scripts.check_design_manual_governance && "
            "python -m scripts.build_tools_module_inventory --check && "
            "python -m scripts.lint_tools_textbook_chapters && "
            "python -m scripts.check_tools_exemplars && "
            "python -m scripts.render_tools_design_manual --check"
        ),
        "required gate",
    )
    for name in ("AGENTS.md", "CLAUDE.md"):
        text = (root / name).read_text(encoding="utf-8")
        for phrase in (
            "manuals/tools",
            "scripts.check_design_manual_governance",
            "scripts.build_tools_module_inventory",
            "scripts.lint_tools_textbook_chapters",
            "scripts.check_tools_exemplars",
            "scripts.render_tools_design_manual",
        ):
            if phrase not in text:
                raise DesignManualGovernanceError(f"{name} is missing manual context")


def verify_repository(root: Path = REPO_ROOT) -> DesignManualGovernanceSummary:
    """Verify repository adoption while retaining the explicit release block."""
    policy = _object(
        _load(root.joinpath(*POLICY_PATH.parts)),
        "governance policy",
        EXPECTED_POLICY_FIELDS,
    )
    manual_id, source_path, allowed = verify_governance_policy(policy)
    inventory = _object(
        policy["calculation_inventory"],
        "inventory",
        {
            "path",
            "owner_subepic",
            "module_manifest",
            "module_schema",
            "module_shard_schema",
            "current_status",
        },
    )
    for field in ("module_manifest", "module_schema", "module_shard_schema"):
        governed_path = _safe_path(inventory[field], field.replace("_", " "))
        if not root.joinpath(*governed_path.parts).is_file():
            raise DesignManualGovernanceError(f"{field.replace('_', ' ')} is missing")
    manifest_path = _safe_path(inventory["module_manifest"], "module manifest")
    read_inventory(root, root.joinpath(*manifest_path.parts))
    check_manual(root)
    registry_path = _safe_path(inventory["path"], "inventory path")
    registry = _load(root.joinpath(*registry_path.parts))
    calculation_count = verify_calculation_registry(registry)
    chapter_summary = verify_textbook_chapters(root)
    verify_formula_traceability(root)
    verify_exemplar_repository(root)
    qmd_count = _verify_manual_tree(root, source_path)
    _verify_context(root, policy)
    for schema in (
        root / "schemas" / "calculation-registry.schema.json",
        root / "schemas" / "publication-projection.schema.json",
    ):
        if schema.exists():
            raise DesignManualGovernanceError(
                f"program-owned schema copy is forbidden: {schema.relative_to(root)}"
            )
    publication = cast(dict[str, object], policy["publication"])
    manifest_path = _safe_path(publication["projection_manifest_path"], "manifest path")
    if not allowed and root.joinpath(*manifest_path.parts).exists():
        raise DesignManualGovernanceError(
            "blocked publication must not have a projection manifest"
        )
    registry_object = cast(dict[str, object], registry)
    return DesignManualGovernanceSummary(
        manual_id=manual_id,
        canonical_qmd_count=qmd_count,
        calculation_count=calculation_count,
        textbook_chapter_count=chapter_summary.chapter_count,
        release_status=_text(registry_object["release_status"], "release status"),
        public_projection_allowed=allowed,
    )


def main() -> int:
    """Run the governance gate with deterministic diagnostics."""
    try:
        summary = verify_repository()
    except (
        DesignManualGovernanceError,
        TextbookChapterError,
        FormulaTraceabilityError,
        ExemplarContractError,
        ToolsModuleInventoryError,
        OSError,
        json.JSONDecodeError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        "Design-manual governance verified: "
        f"{summary.canonical_qmd_count} QMD sources, "
        f"{summary.calculation_count} registered calculations, "
        f"{summary.textbook_chapter_count} registered textbook chapters, "
        f"release={summary.release_status}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
