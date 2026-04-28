# ADR-001: Monorepo Workspace Structure

- Status: Accepted
- Date: 2026-04-28
- Decision Makers: Tools maintainers
- Related Issues/PRs: #2357

## Context

The Tools repository contains multiple independent web applications
(scientific calculators, simulators, media processors) each with its own
npm package.  Previously there was no root package.json, preventing
workspace-aware commands like `npm test --workspaces` and making CI
matrix generation manual.

## Decision Flow

```mermaid
flowchart TD
    A[Multiple npm packages in src/] --> B{Root manifest?}
    B -->|No| C[Manual CI per package]
    B -->|Yes| D[Unified workspace commands]
    C --> E[Duplicated config]
    D --> F[Single CI matrix]
    E --> G[Maintenance burden]
    F --> H[Reduced duplication]
    G --> I[Decision: Add root package.json]
    H --> I
```

## Decision

Add a root `package.json` with `workspaces` array enumerating all npm
packages under `src/`.  Mark the root as `"private": true` to prevent
accidental publishing of an umbrella package.

## Alternatives Considered

1. **Keep individual packages**: Rejected because CI duplication is high.
2. **Use pnpm workspaces in separate `pnpm-workspace.yaml`**: Rejected to
   stay within npm ecosystem and reduce toolchain dependencies.

## Component / Deployment Diagram

```mermaid
graph TD
    subgraph "Tools Monorepo"
        R[package.json<br/>workspaces]
        R --> P1[src/media_processing/...]
        R --> P2[src/rotation_converter/web]
        R --> P3[src/ode_solver/web]
        R --> P4[... 7 more]
    end
    subgraph "CI Pipeline"
        R -->|npm test --workspaces| T[Test Matrix]
    end
```

## Consequences

- Positive: Unified install, test, and build commands across all packages.
- Negative: Requires Node >= 18 for modern workspace support.
- Follow-up actions: Add workspace-aware lint and build CI jobs.