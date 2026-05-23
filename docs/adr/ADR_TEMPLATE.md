# ADR-XXXX: <Title>

- Status: Proposed | Accepted | Superseded
- Date: YYYY-MM-DD
- Decision Makers: <names/roles>
- Related Issues/PRs: <links>

## Context

Architecture-impacting problem and constraints.

## Decision Flow

```mermaid
flowchart TD
    A[Identify Problem] --> B{Evaluate Constraints}
    B -->|Option A| C[Alternative A]
    B -->|Option B| D[Chosen Solution]
    C --> E{Trade-off Analysis}
    D --> E
    E --> F[Decision Accepted]
    F --> G[Implementation]
    G --> H[Validation & Monitoring]
```

## Decision

Chosen architectural direction.

## Alternatives Considered

1. <Alternative A>
2. <Alternative B>

## Component / Deployment Diagram (Optional)

```mermaid
graph LR
    subgraph System
        S1[Module A]
        S2[Module B]
    end
    S1 -->|interface| S2
```

## Consequences

- Positive:
- Negative:
- Follow-up actions:

## Validation

How this decision is validated in CI, tests, and operation.
