# Architecture Decision Records (ADRs)

This directory stores architecture decisions for cross-tool boundaries and shared foundations.

## Policy

- Use `ADR_TEMPLATE.md` for every new ADR.
- Naming format: `NNNN-short-title.md`.
- Every ADR must include date, status, and validation approach.
- Superseded ADRs must point to the replacement ADR.

## Initial ADR Backlog

1. Shared tool skeleton and extension boundaries.
2. UI/shared/core layering and import direction.
3. CI gate ownership and blocking scope.

## Records

| ADR                                                   | Status   | Summary                                                                                            |
| ----------------------------------------------------- | -------- | -------------------------------------------------------------------------------------------------- |
| [ADR-001](ADR-001-monorepo-workspaces.md)             | Accepted | Defines monorepo workspace package boundaries and dependency direction.                            |
| [ADR-002](ADR-002-shared-library-module-structure.md) | Accepted | Why the current layered namespace structure was chosen over flat or multi-package alternatives.    |
| [ADR-003](ADR-003-api-stability-policy.md)            | Accepted | Policy for backwards-compatible API changes, deprecation protocol, and contract test requirements. |
| [ADR-004](ADR-004-ruff-formatter.md)                  | Accepted | Why ruff format was chosen over Black as the canonical Python formatter.                           |
| [ADR-005](ADR-005-plugin-discovery-vs-registry.md)    | Accepted | Dual-mode plugin registration: per-tool manifests merged with centralized tools.json.              |
| [ADR-006](ADR-006-type-safety-mypy-strict.md)         | Accepted | Type safety enforcement strategy using mypy delta CI and py.typed marker.                          |
| [ADR-007](ADR-007-markerless-mocap-authority-and-licensing.md) | Accepted | Cross-repository markerless-mocap authority, evidence, coordinate, C3D, privacy, and licensing boundaries. |
