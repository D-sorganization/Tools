# Documentation Governance

## Canonical Indices

- Root docs index: `docs/README.md`
- Assessment index: `docs/assessments/README.md`
- ADR index: `docs/adr/README.md`

## Freshness Rules

- Changes under `docs/assessments/**` require updating `docs/assessments/README.md` in the same PR.
- Changes under `docs/adr/*.md` (excluding template/index) require updating `docs/adr/README.md`.
- ADRs must use `docs/adr/ADR_TEMPLATE.md`.

## Engineering Design Manual

- Canonical editable source: `manuals/tools` QMD.
- Governance contract: `config/design_manual_governance.json`.
- Calculation inventory: `manuals/tools/calculation-registry.json`.
- Offline gate: `python -m scripts.check_design_manual_governance`.
- Generated LaTeX, PDF, DOCX, and HTML are non-editable artifacts. TOOLS-D7
  and TOOLS-D8 must record semantic, page-render, accessibility, digest, license,
  and human-approval evidence before any public projection.
- Existing user guides, ADRs, API references, and package documentation are
  separate governed products, not alternate mutable manual authorities.
