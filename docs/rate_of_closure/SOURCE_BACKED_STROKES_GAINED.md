# Source-Backed Strokes Gained

Issues: [Tools #4584](https://github.com/D-sorganization/Tools/issues/4584),
[Tools #4229](https://github.com/D-sorganization/Tools/issues/4229)

## Availability Boundary

The application does not bundle an expected-strokes table and does not treat a
URL as a validated baseline. True source-backed calculation becomes available
only after the user loads a licensed artifact conforming to
`launch-monitor-strokes-gained-baseline/1.0.0`. Both PyQt6 and React verify the
artifact's table SHA-256, version, source URL, license declaration, exact state
fields, finite values, and unique lie/distance rows.

An artifact declaration is traceability metadata, not legal advice or an
independent license audit. The user remains responsible for authorization to
use and redistribute the baseline.

## Required Artifact

The artifact is JSON with this shape:

```json
{
  "contract_version": "launch-monitor-strokes-gained-baseline/1.0.0",
  "baseline_id": "licensed-baseline-name",
  "version": "2026.1",
  "source_url": "https://publisher.example/methodology",
  "license": "publisher-license-identifier",
  "table_sha256": "64 lowercase hexadecimal characters",
  "states": [
    {"lie": "fairway", "distance_yards": 100, "expected_strokes": 2.8}
  ]
}
```

`table_sha256` is SHA-256 over UTF-8 JSON for `states`, with object keys sorted,
no insignificant whitespace, and non-ASCII characters retained. The repository
ships a JSON Schema for field validation, while the clients enforce the digest
and cross-row rules.

## Calculation

Each retained shot must explicitly identify:

- before lie and distance;
- after lie and distance; and
- the source unit for both distances.

Distances are converted to yards. Expected strokes are linearly interpolated
only between bracketing distances within the same lie. Extrapolation and an
unknown lie fail closed. For a complete shot:

```text
SG = E(before lie, before distance) - 1 - E(after lie, after distance)
```

Exports retain baseline identity/version/source/license/hash plus every course
state, interpolated expectation, and shot SG. Radial target error and the older
user-supplied expected-strokes bookkeeping remain separately named and cannot
masquerade as source-backed SG.
