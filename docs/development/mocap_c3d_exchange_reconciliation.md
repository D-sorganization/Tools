# TOOLS-M9 C3D Exchange Reconciliation

Status: acceptance contract only; runtime implementation is blocked.

Issue [#4716](https://github.com/D-sorganization/Tools/issues/4716) asks Tools
to own deterministic C3D exchange for markerless motion capture. This audit
reconciles the prior local prototype with current protected `main`. It does not
publish a runtime schema, writer, reader extension, fixture corpus, or completion
claim.

## Authority and exact state

- Protected Tools `main` was
  `cff2909f1585273e10fa49165bfab8521e889da1` when this contract was cut.
- The supplied integration head
  `d38ac0736d9a2dd9620a844520e9ec486331e614` is local-only. Its merge base with
  protected `main` is `31d28b0a0a0435cd47d05bedc61a1357d670a8d8`;
  merging its tree would remove substantial protected-main work.
- The older M9 branch head
  `0bdc4647530950ca63f340c628a81ff771966c61` diverges 10 protected-main
  commits and 8 local commits from merge base
  `e76a7a21408db9ba55b18959f0fc513bf63ec579`.
- M0 issue #4708 and M1 issue #4710 remain open. PR #4734 points at
  `f536eda05cb1824463a11e0b58cc7b94e6ffea1` and is not protected-merged.
- Protected main therefore has no canonical `sidekick.lab.mocap.MocapSession`
  authority on which M9 can safely depend. The existing
  `sidekick.lab.bio.c3d_reader` remains the protected read-only C3D surface.
- AffineDrift issue #3959 is a consumer. It must consume an immutable Tools
  release; it must not define a second schema, writer, or loss policy.

Local commits and files are design evidence only. They are not merge inputs and
must be reconstructed from the future dependency-ready protected main.

## Rejected prototype behavior

| Audited behavior | Acceptance decision |
| --- | --- |
| A separate `C3DRecord` type family | Rejected; canonical input is the future protected `MocapSession`. |
| Repeated-write byte equality | Rejected as conformance; compare normalized semantics. |
| Optional or skipped python-c3d oracle | Rejected; missing qualification tooling fails closed. |
| First-seven contributor masks on overflow | Rejected; 8 or more is unavailable in the standard mask. |
| Embedded extension as the only loss record | Rejected; an adjacent digest-bound sidecar is required. |
| No normalized comparison profile | Rejected; every required semantic family must normalize. |
| No pinned C3D.org reference corpus | Rejected; corpus qualification is mandatory. |

Byte stability may be measured as a diagnostic, but it cannot substitute for
semantic stability. Two compliant readers can serialize equivalent parameter
groups differently. Conversely, byte equality does not prove that units,
timestamps, frames, missingness, or provenance were interpreted correctly.

## Required normalized semantics

The exchange profile must define canonical comparison for frames, coordinate
and measurement units, timestamps, confidence, skeleton topology, events,
point residuals, analog channels, force-platform data, provenance, and unknown
metadata. Each family needs explicit type, units, missing-value behavior,
ordering, tolerance where numeric, and provenance rules.

Unknown metadata must be preserved losslessly or rejected explicitly before
write. Silent discard is prohibited. Determinism means that normalized
`write -> read -> write` results are stable under the versioned semantic
profile; it does not require identical C3D bytes.

## Contributor masks and loss reporting

C3D's standard point camera byte can represent at most seven contributing
cameras. The qualification matrix is 0, 1, 7, 8, and arbitrary `N > 8`:

- 0, 1, and 7 contributors use the exact standard representation.
- For 8 or more, the standard mask is zero/unavailable. The writer must never
  truncate to the first seven, wrap bits, remap identities, or invent a subset.
- Overflow requires an adjacent canonical-JSON loss sidecar containing the
  complete contributor identities, all unrepresented losses, and writer
  provenance.
- The sidecar binds the C3D SHA-256, canonical session SHA-256, and semantic
  profile SHA-256. Missing or tampered required sidecars fail closed.
- The loss record itself must remain stable across normalized
  `write -> read -> write` qualification.

An embedded vendor extension may supplement interoperability, but it is not the
canonical loss record and cannot make overflow appear standard-representable.

## Independent qualification

The implementation gate requires all of the following, with pinned versions
and fixture digests:

1. ezc3d for the primary interoperability path;
2. python-c3d as an independent reader;
3. BTK for legacy ecosystem compatibility; and
4. the C3D.org sample corpus as normative external fixtures.

Oracle absence, fixture absence, parse failure, or semantic mismatch is a
qualification failure. Tests may not skip, xfail, retry away, or replace an
independent oracle with the writer under test. Golden results compare normalized
semantic projections and explicit loss reports, not bytes alone.

## Dependency-ready gate

Runtime work may begin only after all of these are true:

1. M0 #4708 and M1 #4710 are ordinary protected merges rooted on current main.
2. Their public schema owns all required semantic families and invariants.
3. The oracle versions, corpus files, licenses, and SHA-256 digests are pinned.
4. The normalized semantic profile and loss-sidecar profile are versioned.
5. AffineDrift #3959 accepts an immutable Tools version rather than copying
   provider code.

Completion then requires a newly implemented main-rooted runtime, the full
camera-count matrix, deterministic semantic round trips, cross-reader and
corpus qualification, tamper tests, and ordinary protected review. The
machine-readable acceptance manifest intentionally reports
`implementation_eligible: false` until those dependencies exist on protected
main.

## Bounded evidence for this slice

The focused test
`tests/architecture/test_mocap_c3d_acceptance_contract.py` checks only this
acceptance manifest. Its RED state proves that the contract preceded the
manifest; its GREEN state will prove only that the documented dependency gate
is internally consistent. It does not exercise, qualify, or claim a C3D runtime.
