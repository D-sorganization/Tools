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
sentinel normalization use exactly the Unicode White*Space code points
U+0009–U+000D, U+0020, U+0085, U+00A0, U+1680, U+2000–U+200A, U+2028,
U+2029, U+202F, U+205F, and U+3000. Sentinel matching folds ASCII `A`–`Z`,
removes that whitespace plus `.`, `/`, `*`, and `-`, and removes terminal `!`or`?`. The result must not equal `x`, `na`, `none`, `nodata`,
`notavailable`, `notapplicable`, or `unavailable`. Consequently `x`, `n/a`,
bare `unavailable`, punctuation/case variants, and boundary-whitespace variants
cannot masquerade as evidence. A sentence such as “No qualified ground
producer was selected for this run.” is valid.

Valid non-BMP scalar values are permitted in descriptive text. JavaScript
validates UTF-16 pairing rather than rejecting every surrogate code unit;
Python and TypeScript both reject only unpaired surrogates. The shared reason
fixture pins an astral scalar, every boundary-whitespace code point above, and
an unpaired-surrogate rejection case.

## Supported Surfaces and Domains

The contract targets four delivery surfaces:

1. `tools.pyqt6`: Tools native desktop application (PyQt6).
2. `tools.react`: Tools web application (React/TypeScript).
3. `upstreamdrift.pyqt6`: UpstreamDrift desktop application embedding Tools (PyQt6).
4. `upstreamdrift.react`: UpstreamDrift web application embedding Tools (React).

Manifest calculations must strictly contain exactly three domains in fixed
canonical sequence:

1. `impact`: Club-ball collision mechanics and initial launch conditions.
2. `flight`: Aerodynamic trajectory integration from launch to landing.
3. `ground`: Surface interaction, bounce, and roll-out physics.

## Wire Schema (`calculation-runtime-manifest/v1`)

A valid runtime manifest is a top-level JSON object with exactly five keys:

- `schema_version`: String literal `"calculation-runtime-manifest/v1"`.
- `surface_id`: Exactly one of the four supported surfaces above.
- `build`: Object containing:
  - `package_name`: Stable identifier string (e.g., `"ud-tools"`).
  - `package_version`: SemVer 2.0.0 text string (leading zeros in numeric core
    or prerelease components are rejected).
  - `tools_commit`: Exact 40-character lowercase hexadecimal git SHA string.
  - `build_id`: Stable identifier string (e.g., `"tools-rate-pyqt6/1.5.0"`).
- `calculations`: Sequence of exactly three calculation authority objects, in
  `impact`, `flight`, `ground` order. Each authority object specifies:
  - `domain`: One of `"impact"`, `"flight"`, `"ground"`.
  - `status`: One of `"available"` or `"unavailable"`.
  - `reason`: `null` if available, or valid explanatory string if unavailable.
  - The nine authoritative calculation identity fields:
    - `model_id`: Stable identifier (or `null` if unavailable).
    - `model_version`: Stable identifier (or `null` if unavailable).
    - `implementation_authority`: Stable identifier (or `null` if unavailable).
    - `backend`: Stable identifier (or `null` if unavailable).
    - `integrator`: Stable identifier (or `null` if unavailable).
    - `request_schema`: Stable identifier (or `null` if unavailable).
    - `result_schema`: Stable identifier (or `null` if unavailable).
    - `frame_id`: Stable identifier (or `null` if unavailable).
    - `unit_system_id`: Stable identifier (or `null` if unavailable).
  - `numerical_options`: Array of `{option_id, value, unit}` objects:
    - `option_id`: Unique stable identifier.
    - `value`: Scalar number (finite, within safe integer magnitude
      `9,007,199,254,740,991`), boolean, or non-placeholder string.
    - `unit`: Nonempty string unit for numbers (dimensionless uses `"1"`);
      must be `null` for text and boolean options.
- `provenance`: Object recording origin of the manifest values:
  - `source_kind`: One of `"installed_package"`, `"source_checkout"`,
    `"embedded_web_build"`, `"test_fixture"`.
  - `source_reference`: Stable identifier string.
  - `evidence_ids`: Nonempty array of unique stable identifier strings.

## Authoritative Fields

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

## Descriptive Evidence

`provenance.source_kind`, `source_reference`, and `evidence_ids` identify where
the caller obtained its authoritative values. They support an audit but do not
upgrade an unavailable calculation or prove scientific validation. Placeholder
evidence is rejected, and evidence IDs must be nonempty and unique.
Placeholder vocabulary is matched as a complete ASCII-alphanumeric-delimited
token. Therefore `.`, `_`, `/`, and `-` stable-ID separators cannot hide a
placeholder, while a legitimate longer substring such as `todolist` remains
valid.

## Canonical Serialization

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

## Cross-Runtime Parity

Both runtimes implement mirrored APIs conforming to the exact same contract:

### TypeScript (`src/rate_of_closure/web/src/model/runtimeManifest.ts`)

- `parseRuntimeManifest(value: unknown): CalculationRuntimeManifest`:
  Validates and deeply freezes an unknown value against the v1 schema.
- `createRuntimeManifest(input: CreateRuntimeManifestInput): CalculationRuntimeManifest`:
  Factory constructing a manifest from explicit caller-supplied fields.
- `runtimeManifestFromJson(textValue: string): CalculationRuntimeManifest`:
  Duplicate-rejecting JSON parser and schema validator.
- `stableRuntimeManifestJson(manifest: CalculationRuntimeManifest): string`:
  Canonical serialization with 11-decimal numeric policy.

### Python (`src/shared/python/swing_sim/runtime_manifest.py`)

- `parse_runtime_manifest(value: object) -> dict[str, Any]`:
  Validates and returns a clean, independent dictionary copy.
- `create_runtime_manifest(*, surface_id: str, build: Mapping[str, Any], calculations: Sequence[Mapping[str, Any]], provenance: Mapping[str, Any]) -> dict[str, Any]`:
  Keyword-only factory matching TypeScript `createRuntimeManifest`.
- `assert_manifests_compatible(manifest_a: Mapping[str, Any], manifest_b: Mapping[str, Any], *, require_same_commit: bool = True) -> None`:
  Cross-surface compatibility validator enforcing consistency across runs.
- `runtime_manifest_from_json(text: str) -> dict[str, Any]`:
  Duplicate-rejecting JSON parser and schema validator.
- `stable_runtime_manifest_json(manifest: Mapping[str, Any]) -> str`:
  Canonical serialization matching TypeScript canonical bytes byte-for-byte.

## Contradiction Detection and Compatibility

The `assert_manifests_compatible` function detects contradictions between two
manifests (e.g., across surfaces `tools.react` vs `tools.pyqt6`, or across test
runs):

1. **Schema Validation**: Both manifests are validated fail-closed with
   `parse_runtime_manifest`.
2. **Commit Alignment**: If `require_same_commit=True` (default), raises
   `ValueError` if `build.tools_commit` differs.
3. **Version Alignment**: Raises `ValueError` if `build.package_version` differs.
4. **Authority Equivalence**: For every domain (`impact`, `flight`, `ground`)
   where **both** manifests have `status == "available"`, all nine authority
   fields (`model_id`, `model_version`, `implementation_authority`, `backend`,
   `integrator`, `request_schema`, `result_schema`, `frame_id`, `unit_system_id`)
   must match identically. If any field differs, a `ValueError` is raised
   identifying the contradicting domain and field.
5. **Partial Availability**: If a domain is `available` on one manifest but
   `unavailable` on the other, or `unavailable` on both, authority divergence is
   not flagged. This allows heterogeneous surfaces to report differing
   domain capabilities while strictly preventing contradicting models for
   domains where both are actively available.

## Current Limitation

This bounded slice defines and verifies the contract only. It does not attach a
manifest to live impact, flight, ground, workspace, or export results; resolve
the Tools source used by UpstreamDrift; qualify any model; or establish release
parity. Those integrations remain separately reviewed work under #4260 through
#4266 and must preserve the exact contract rather than fabricate provenance.
