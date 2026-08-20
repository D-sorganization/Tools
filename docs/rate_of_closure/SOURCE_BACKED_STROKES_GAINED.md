# Source-Backed Strokes Gained

Issues: [Tools #4584](https://github.com/D-sorganization/Tools/issues/4584),
[Tools #4229](https://github.com/D-sorganization/Tools/issues/4229)

## Availability Boundary

The application does not bundle an expected-strokes table and does not treat a
URL as a validated baseline. True source-backed calculation becomes available
only after the user loads a licensed artifact conforming to
`launch-monitor-strokes-gained-baseline/2.0.0`. Both PyQt6 and React verify the
artifact's table SHA-256, version, source URL, license declaration, exact state
fields, finite values, and unique lie/context/target/distance rows.

When an UpstreamDrift authority URL is configured, both clients submit the
same canonical request to `POST /tools/launch-monitor-analytics/v2/strokes-gained`
and validate the returned `launch-monitor-strokes-gained-result/1.0.0`
envelope. Leaving the URL blank selects a clearly labelled local compatibility
calculation; it does not silently claim service authority.

An artifact declaration is traceability metadata, not legal advice or an
independent license audit. The user remains responsible for authorization to
use and redistribute the baseline.

## Required Artifact

The artifact is JSON with this shape:

```json
{
  "contract_version": "launch-monitor-strokes-gained-baseline/2.0.0",
  "baseline_id": "licensed-baseline-name",
  "version": "2026.1",
  "source_url": "https://publisher.example/methodology",
  "license": "publisher-license-identifier",
  "table_sha256": "64 lowercase hexadecimal characters",
  "states": [
    {
      "lie": "fairway",
      "context": "approach",
      "target": "hole",
      "distance_yards": 100,
      "expected_strokes": 2.8,
      "standard_error": 0.04
    }
  ]
}
```

`table_sha256` is SHA-256 over UTF-8 JSON for `states`, with object keys sorted,
no insignificant whitespace, and non-ASCII characters retained. The repository
ships a JSON Schema for field validation, while the clients enforce the digest
and cross-row rules.

## Calculation

Each retained shot must explicitly identify:

- before lie, context, target, and distance;
- after lie, context, target, and distance; and
- the source unit for both distances.

Distances are converted to yards. Expected strokes are linearly interpolated
only between bracketing distances within the exact same lie/context/target
stratum. Extrapolation and an unknown stratum fail closed. For a complete shot:

```text
SG = E(before lie, before distance) - 1 - E(after lie, after distance)
```

Exports retain baseline identity/version/source/license/hash plus every course
state, interpolated expectation, and shot SG. Radial target error and the older
user-supplied expected-strokes bookkeeping remain separately named and cannot
masquerade as source-backed SG.

The canonical result reports structured exclusions, descriptive Student-t
uncertainty, and optional propagation of benchmark standard errors. Player,
session, club, and longitudinal summaries are included only for identifiers
and order columns the user explicitly selects and attests; filename, row order,
monitor, source partition, or inferred identity never qualify. These summaries
are descriptive and do not establish causal player improvement.
