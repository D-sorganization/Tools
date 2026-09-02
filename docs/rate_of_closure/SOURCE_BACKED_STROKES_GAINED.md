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
stratum. Extrapolation and an unknown stratum are never guessed at. For a
complete shot:

```text
SG = E(before lie, before distance) - 1 - E(after lie, after distance)
```

## Error Posture: Exclude and Audit

Per UpstreamDrift ADR-0048 decision G1-D3, every calculation path — canonical
service, PyQt6 local compatibility, and React local compatibility — handles a
malformed shot the same way: the row is **excluded**, recorded against a
`reason_code`, and counted. A malformed row never destroys the session, and it
is never dropped in silence.

| `reason_code`          | Meaning                                                                                  |
| ---------------------- | ---------------------------------------------------------------------------------------- |
| `missing_course_state` | A lie, context, target, or distance cell is blank or non-numeric.                        |
| `invalid_distance`     | A distance is negative, non-finite, or a boolean.                                        |
| `outside_baseline`     | The stratum is absent from the baseline, or the distance is outside its supported range. |

Every result carries a `status` and an exclusion summary:

- `available` — every supplied row was scored.
- `partial` — at least one row scored, at least one excluded.
- `unavailable` — no row could be scored; the mean is null.

The exclusion summary reports `input_row_count`, `included_row_count`,
`total_excluded`, and `by_reason`, and
`input_row_count == included_row_count + total_excluded` always holds. Each
excluded row is listed individually with its zero-based `source_index`, so a
caller can map an exclusion straight back to the input record. Exports carry
the audit trail alongside the values.

A caller that wants fail-closed behaviour raises on `status != "available"`;
a caller handed an exception could not have recovered the good rows. Defects
in the _request_ — an absent column, a distance unit that is not `yd`/`m`, or a
baseline artifact whose table digest does not verify — remain fatal, because
they are the caller's declaration rather than the data's content.

Exports retain baseline identity/version/source/license/hash plus every course
state, interpolated expectation, and shot SG. Radial target error and the older
user-supplied expected-strokes bookkeeping remain separately named and cannot
masquerade as source-backed SG.

The canonical result additionally reports descriptive Student-t uncertainty and
optional propagation of benchmark standard errors; the local compatibility
calculation reports neither and does not claim to. Player,
session, club, and longitudinal summaries are included only for identifiers
and order columns the user explicitly selects and attests; filename, row order,
monitor, source partition, or inferred identity never qualify. These summaries
are descriptive and do not establish causal player improvement.
