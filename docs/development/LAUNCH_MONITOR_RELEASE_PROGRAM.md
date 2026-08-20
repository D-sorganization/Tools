# Launch-Monitor Professional Release Program

Program epic: [#4583](https://github.com/D-sorganization/Tools/issues/4583)

The machine-readable release authority is
[`docs/release/launch_monitor_program.v1.json`](../release/launch_monitor_program.v1.json).
It pins the five participating repositories, records the approved ownership and
scientific policies, separates Release A from Release B, and assigns every
capability to one repository and tracking issue.

## Architecture

- UpstreamDrift owns statistical definitions and the versioned API contract.
- Tools owns the Rate-of-Closure PyQt6 and React/Vite presentation surfaces.
- The private campaign owns restricted rows, cohort qualification, campaigns,
  and trained artifacts.
- Launch-Monitor-Data remains a data-free public schema and authenticated access
  client pinned to an exact private commit.
- AffineDrift publishes the reviewed scientific and engineering narrative.

Restricted data never flows into a public repository or browser bundle. Tools
may use an authenticated local/API bridge, source identifiers, and immutable
hashes, but it does not copy private rows into project files by default.

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

## Delivery order

1. Land the UpstreamDrift v2 contract and private corpus-use qualification.
2. Pin those releases in this manifest and the public access lock.
3. Port current-main Tools features against the released contract.
4. Complete model comparison, player analytics, exports, neural workflows, and
   strokes-gained/proxy analysis.
5. Publish the AffineDrift paper and Release B protocol against exact release
   commits.
6. Run clean-machine, privacy, parity, accessibility, performance, visual, and
   protected-branch release gates.

Historical stacked branches are evidence and recovery sources. They are not
release authorities and must not be merged wholesale into the consolidated
application.

## Completion rule

An item is complete only when its current-main implementation, required tests,
documentation, protected checks, and release artifact are all verified. A green
focused test does not override a failing required repository check, and a large
row count does not establish eligibility for a particular scientific analysis.
