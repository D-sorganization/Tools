# Variation Plan Persistence and Replay Contract

## Purpose

Variation results are scientifically interpretable only when the sampled plan,
resolved variable semantics, random stream, implementation identity, and source
provenance remain bound. A raw `variation-plan/v2` object is an input recipe;
it is not evidence that a historical execution can be reproduced.

This contract defines the canonical evidence carried by persisted Rate of
Closure variation artifacts. It does not claim numerical equivalence across
Python and React executors. Cross-runtime plan digests are comparable, while
runtime-specific execution identities remain distinct and replay fails closed
when the local executor cannot establish compatibility.

## Canonical Execution Document

`rate-of-closure/variation-execution-document` version 3 has exactly five
fields: `schema_id`, `schema_version`, `plan`, `metadata`, and `provenance`.
The metadata binds:

- the canonical plan SHA-256;
- mode, flight model, and complete resolved variable values with units and
  physical dimensions;
- variable-registry identity, version, and SHA-256;
- RNG algorithm and stream-derivation identities;
- runtime, executor, and solver identities; and
- the SHA-256 of the producer/source provenance record.

Canonical hashing recursively sorts object keys, retains array order, encodes
all finite JSON numbers as IEEE-754 binary64 bytes, normalizes negative zero,
and rejects unsafe integers, nonfinite values, unsupported values, duplicate
JSON fields, or unpaired Unicode surrogates. Python and TypeScript golden
fixtures prove plan and registry digest parity while preserving their different
runtime and RNG identities.

## Persisted Plan Binding

`rate-of-closure/variation-plan-binding` version 1 has exactly six fields:
`schema_id`, `schema_version`, `state`, `document`, `legacy_plan`, and
`legacy_warning`.

- `state = canonical` retains one complete version-3 execution document and no
  legacy fields.
- `state = legacy` retains the exact normalized raw plan and the fixed warning
  that historical execution metadata is unavailable. It never invents source,
  registry, RNG, or implementation evidence.

Loading and saving a legacy workspace or named-library entry preserves that
legacy state and visible warning. Merely opening an old plan does not upgrade
its evidentiary status. Authoring a new plan creates a current canonical
document.

## Artifact Inventory

| Surface | Binding | Compatibility and Failure Behavior |
| --- | --- | --- |
| PyQt6 plan file | Full version-3 document | Legacy raw plans load with a visible warning; altered plan, metadata, or provenance fails. |
| React plan file | Full version-3 document | Same canonical/legacy distinction; duplicate fields and incompatible local replay fail. |
| React named-plan library | Full document or retained legacy plan | Library version 2 preserves evidence state across browser storage. |
| Python and React workspace | Version-1 binding | Workspace version 3 preserves canonical or legacy evidence and rejects plan/ball-setup substitution. |
| Python scalar dataset JSON | Full version-3 document | Dataset version 2 rejects legacy raw-plan documents and digest substitution. |
| React scalar dataset JSON | Full version-3 document | Dataset version 2 emits the same evidence shape with React execution identity. |
| Python complete geometry ensemble | Full version-3 document within dataset | Reader validates the cohesive document before accepting outcomes or traces. |
| React swing ensemble | Full version-3 document within dataset | Export version 3 validates document, sampled inputs, trial rows, and localized commands. |
| Durable archive | Full version-3 document in the checksummed header | Archive version 2 binds the document through the header and manifest hashes. |
| Durable browser-to-Python request | Raw plan plus cross-runtime plan SHA-256 | Request version 2 rejects plan substitution; Python authors its own execution identity. |
| Regional-ground request | Raw plan plus cross-runtime plan SHA-256 | Request version 2 rejects plan substitution in both runtimes. |
| Regional-ground result | Common plan SHA-256 on every retained row | Import rejects malformed or crossed plan digests; per-trial input and regional-plan digests remain separate. |
| Chip-forgiveness JSON | Full version-3 document | Export version 2 binds its population to the plan; no historical reader is claimed. |
| CSV table exports | External-plan only | CSV is a review table, not self-contained replay evidence. A consumer must supply and validate the canonical JSON artifact; the table alone cannot authorize replay. |
| Paired propagation analysis | In-memory only | No paired-study file format is currently public. Pairing requires compatible layouts and row identities; persisted comparisons must retain each source ensemble rather than infer a plan from plots. |
| Plot and visual state | Derived, non-authoritative | Selectors and chart state do not authorize simulation or historical replay. |

## Replay and Cross-Runtime Rules

1. A matching plan digest proves that the canonical plan is identical; it does
   not prove identical solvers, RNG streams, floating-point behavior, or output.
2. A request crossing from React to Python carries the plan digest, not a false
   React-as-Python execution identity. The Python authority creates the
   execution document used by its results.
3. Replay requires an exact supported execution identity and provenance.
   Current execution parsers reject a foreign or obsolete identity. A future
   inspection-only carrier may retain such a document, but it must not execute
   it as if it were locally compatible.
4. Any plan/document, plan/dataset, plan/archive, or row/plan digest mismatch is
   a fatal integrity error. There is no current-registry fallback for canonical
   evidence.
5. Synthetic variation traces remain model-scenario evidence. These contracts
   improve reproducibility and falsifiability; they do not create human
   validation or justify universal coaching advice.

## Verification

The focused Python and TypeScript suites cover canonical round trips, shared
fixtures, legacy retention, duplicate keys, altered plan/metadata/provenance,
request substitution, workspace ball-setup substitution, crossed result-row
digests, and unsupported replay identities. Repository lint, type-check, and
document-governance gates remain required before protected merge.
