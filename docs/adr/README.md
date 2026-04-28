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

| ADR | Status | Summary |
| --- | --- | --- |
| [ADR-001](ADR-001-monorepo-workspaces.md) | Accepted | Defines monorepo workspace package boundaries and dependency direction. |
