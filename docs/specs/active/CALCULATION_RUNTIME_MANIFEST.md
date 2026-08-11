# Calculation Runtime Manifest v1

Issue [#4261](https://github.com/D-sorganization/Tools/issues/4261) defines the
calculation-authority foundation for the four-surface parity epic #4260. The
wire schema is `calculation-runtime-manifest/v1`; Python and TypeScript parse
the same fixture and emit the same compact canonical JSON bytes.

## Boundary

The contract records evidence supplied by a delivery adapter. It never runs
Git, searches sibling repositories, chooses a provider, reads mutable package
metadata, or infers support from a visible control. This makes contract parsing
deterministic and keeps source-resolution policy outside calculation models.

Every manifest contains exactly one entry, in canonical order, for `impact`,
`flight`, and `ground`. An entry is either:

- `available`: `reason` is null and model, implementation authority, backend,
  integrator, request/result schemas, frame, and unit-system identities are all
  present; or
- `unavailable`: `reason` is substantive, every calculation identity is null,
  and numerical options are empty.

An unavailable entry is not an error and must not be replaced with a fallback
identity. It is the truthful representation of a calculation that the run did
not execute through a qualified authority.

An unavailable reason has one cross-runtime grammar. It must already be
trimmed, contain 16 through 500 Unicode scalar values, and contain at least
three ASCII-letter word tokens of two or more letters. Boundary trimming and
sentinel normalization use exactly the Unicode White_Space code points
U+0009–U+000D, U+0020, U+0085, U+00A0, U+1680, U+2000–U+200A, U+2028,
U+2029, U+202F, U+205F, and U+3000. Sentinel matching folds ASCII `A`–`Z`,
removes that whitespace plus `.`, `/`, `_`, and `-`, and removes terminal `!`
or `?`. The result must not equal `x`, `na`, `none`, `nodata`,
`notavailable`, `notapplicable`, or `unavailable`. Consequently `x`, `n/a`,
bare `unavailable`, punctuation/case variants, and boundary-whitespace variants
cannot masquerade as evidence. A sentence such as “No qualified ground
producer was selected for this run.” is valid.

Valid non-BMP scalar values are permitted in descriptive text. JavaScript
validates UTF-16 pairing rather than rejecting every surrogate code unit;
Python and TypeScript both reject only unpaired surrogates. The shared reason
fixture pins an astral scalar, every boundary-whitespace code point above, and
an unpaired-surrogate rejection case.

## Authoritative fields

The following fields determine how a result must be interpreted and compared:

- `schema_version` and `surface_id`;
- `build.package_name`, `build.package_version`, exact 40-character
  `build.tools_commit`, and `build.build_id`;
- calculation `domain`, `status`, model/version, implementation authority,
  backend, integrator, request/result schemas, frame, and unit system;
- every numerical option ID, value, and unit.

Consumers must fail closed on unknown fields or versions, incomplete available
identities, identities attached to an unavailable domain, duplicate domains or
options, non-finite values, unsafe integers, and contradictory unit semantics.
Numeric options require an explicit unit; categorical and Boolean options use a
null unit. Dimensionless numeric values use unit `1`. Package versions follow
SemVer 2.0.0, including the prohibition on leading zeros in numeric core and
prerelease identifiers.

## Descriptive evidence

`provenance.source_kind`, `source_reference`, and `evidence_ids` identify where
the caller obtained its authoritative values. They support an audit but do not
upgrade an unavailable calculation or prove scientific validation. Placeholder
evidence is rejected, and evidence IDs must be nonempty and unique.

## Canonical serialization

Objects are sorted by Unicode key, arrays preserve their contract order, and
numbers use the shared finite 11-decimal fixed-token policy. Integers must fit
the JavaScript safe range, and every floating-point manifest-option magnitude
must also be no greater than `9,007,199,254,740,991`. Both manifest validators
enforce that bound before serialization, so accepted manifest numbers and
serializable manifest numbers are the same domain. The shared Python numeric
encoder retains its broader established domain for existing non-manifest
consumers. Duplicate JSON fields and unpaired surrogate code points are
rejected. `runtime_manifest_parity_v1.json` pins the exact wire shape, numeric
boundaries, reason grammar, and bytes used by both runtimes.

## Explicit creation

Python `create_runtime_manifest` and TypeScript `createRuntimeManifest` accept
only an explicit surface, build, calculation ledger, and provenance record.
Delivery code must source those values before calling the factory. A future
adapter may read an installed package or immutable build injection, but the
contract layer will not silently inspect ambient state.

## Current limitation

This bounded slice defines and verifies the contract only. It does not attach a
manifest to live impact, flight, ground, workspace, or export results; resolve
the Tools source used by UpstreamDrift; qualify any model; or establish release
parity. Those integrations remain separately reviewed work under #4260 through
#4266 and must preserve the exact contract rather than fabricate provenance.
