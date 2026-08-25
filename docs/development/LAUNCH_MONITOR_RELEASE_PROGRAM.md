# Launch-Monitor Professional Release Program

Program epic: [#4583](https://github.com/D-sorganization/Tools/issues/4583)
Player analytics platform parent: [#4226](https://github.com/D-sorganization/Tools/issues/4226)

The machine-readable release authority is
[`docs/release/launch_monitor_program.v1.json`](../release/launch_monitor_program.v1.json).
It pins the five participating repositories to exact current HEAD commits, records the approved
ownership and scientific policies, separates Release A from Release B, and assigns every
capability to one repository and tracking issue.

## Architecture

- **UpstreamDrift** (`99cfe284bc9e80d576e027db7f81b8bb9b2af61e`): owns statistical definitions,
  canonical analysis contract v2, scoring baseline models, and API services.
- **Tools** (`535b389e12f574e4a761b5d7a7f41039fe942a36`): owns the Rate-of-Closure PyQt6 and
  React/Vite presentation surfaces, convention registries, and workspace persistence (v3).
- **Launch-Monitor-Flight-Model-Campaign** (`d469b8a427418fa00e99b0ad488e4310b067697d`): owns the
  restricted 261,666-row shot corpus across 27 sources, cohort qualification, campaigns, and trained artifacts.
- **Launch-Monitor-Data** (`5bb753f6d1c1a866c226eacb3672a211fedc04d0`): remains a data-free public
  schema and authenticated access client pinned to the exact private authority.
- **AffineDrift** (`c9c5db7470553ef544262c62c7273bcfb8654812`): publishes the reviewed scientific
  and engineering narrative.

Restricted data never flows into a public repository or browser bundle. Tools
may use an authenticated local/API bridge, source identifiers, and immutable
hashes, but it does not copy private rows into project files by default.
The PyQt desktop client additionally supports a user-authorized, manifest-
verified local load of all 261,666 qualified-authority rows across 27 sources;
its interactive scatter is bounded to 2,000 displayed points while analysis and
explicit export retain the full frame. This desktop-only access does not weaken
the browser or ordinary file-import boundaries.

## Scientific boundaries

Release A is a professional platform release using qualified existing evidence.
It can report agreement with vendor outputs, not independent device accuracy.
Release B requires simultaneous same-shot observations from multiple devices and
an independent reference wherever feasible.

The Release B protocol, capture schema, validator, synthetic fixture, and
confirmatory 252-pair power plan are ready in the private authority. Its status
is `protocol_ready`, not complete: no paired observations have been collected,
so cross-device validation remains unavailable.

ShotLink-derived rows remain internal and cannot train a vendor surrogate.
Foresight, FlightScope, and other vendor-named surrogates remain unavailable
until an approved row-level dataset supports them. Player-level analysis requires
an explicit trustworthy identity field and fails closed otherwise.

## Current Capability Statuses

| Capability | Repository | Issue | Status |
| --- | --- | --- | --- |
| `analysis-contract-v2` | UpstreamDrift | #8790 | **complete** |
| `corpus-use-qualification` | Campaign | #21 | **complete** |
| `multi-source-flight-model-campaign` | Campaign | #21 | **complete** |
| `paired-device-validation` | Campaign | #22 | **protocol_ready** |
| `within-player-covariation` | Tools | #4277 | **complete** |
| `strokes-gained-and-proxy` | Tools | #4584 | **complete** |
| `professional-evidence-publication` | AffineDrift | #3883 | **complete** |
| `rate-client-feature-parity` | Tools | #4226 | **in_progress** |
| `neural-model-lab` | Tools | #4240 | **in_progress** |

## Completion rule

An item is complete only when its current-main implementation, required tests,
documentation, protected checks, and release artifact are all verified. A green
focused test does not override a failing required repository check, and a large
row count does not establish eligibility for a particular scientific analysis.
