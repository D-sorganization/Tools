# CHARTER.md — Tools

> Proposed by Project Steward 2026-09-23. Derived from README, SPEC.md, open epics,
> and `AGENT_HANDOFF.md`. Update this file when goals or feature statuses change.

## End Goal

Tools is the fleet's shared engineering library and desktop toolset, consumed by
UpstreamDrift and Gasification_Model. It delivers signal processing, URDF
generation, process calculators, P&ID utilities, biomechanics analysis, and
visualization themes under one PyQt6 launcher, with React/FastAPI web mirrors
for key tools. Every public API change here is a potential breaking change for
downstream consumers.

### Goals

1. **Stable shared API.** Every public function is under contract-test cover and
   versioned; breaking changes follow a coordinated deprecation path.
2. **Desktop + Web parity.** Each tool with a web surface maintains behavioral
   parity with its PyQt6 counterpart.
3. **Governed engineering design manual.** A renderable tools design manual with
   a versioned calculation registry, exemplar contracts, and publication projection.
4. **Production distribution.** Rate of Closure web distribution qualified for
   production use via CI, visual baselines, and cross-repo integration gates.
5. **Scientific integrity.** Deferred physical validation gates remain explicit;
   software ships provably correct without fabricated evidence.
6. **Security.** Self-hosted runner attack surface is resolved before the runner
   fleet expands.

## Non-Goals

- Treat numerical fixtures, issue closure or source declarations as physical validation.
- Authorize equipment purchases, data collection or participants through a status update.
- Duplicate the owner planning catalog or replace downstream research responsibilities.

## Features

The first fifteen rows preserve the Project Steward snapshot at `ab5ec5674`;
`open` maps to `planned`, and `in_progress` to `in-progress`. Their descriptions
and priorities are retained, not newly requalified. The final three rows project
the published owner plans. Counts refer to listed features, not overall scientific
completion; software milestones and external validation remain distinct.

| ID         | Feature                                                                  | Status      | Tracking | Notes                                                                                                                                                                       |
| ---------- | ------------------------------------------------------------------------ | ----------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| TOOLS-4464 | Self-hosted runner security hardening                                    | planned     | #4464    | P0; no PR; requires Board decision on fork-PR approval policy                                                                                                               |
| TOOLS-4103 | Swing-Impact-Ball-Flight Platform (`rate_of_closure`)                    | in-progress | #4103    | P1; workspace compositor shipped (#5264); camera cluster (#4571) and WASM/Pages CI (Phase 7) remain                                                                         |
| TOOLS-4433 | Visual-first tab UX (PyQt6 + React)                                      | in-progress | #4433    | P1; 8 verified / 23 partial; acceptance manifests landing                                                                                                                   |
| TOOLS-4377 | Rate of Closure production distribution                                  | in-progress | #4377    | P2; Release A shipped; Release B requires real paired launch-monitor data                                                                                                   |
| TOOLS-4260 | Site-wide Golf Impact & Flight contract authority                        | in-progress | #4260    | P2; impact-interval PyQt tab ruled out (#4946); contract authority under T4/T5 of #5068                                                                                     |
| TOOLS-5068 | Shared flexible impact, shaft, and vibroacoustic models                  | in-progress | #5068    | P2; T1–T4 merged; grip qualification shipped (#5297–#5301); physical evidence outstanding                                                                                   |
| TOOLS-4267 | Qualified ball landing, bounce, roll, and ground models                  | in-progress | #4267    | P2; deferred physical validation preserved in planning catalog                                                                                                              |
| TOOLS-4706 | Vendor-neutral markerless mocap                                          | in-progress | #4706    | P2; reconstruction core in PR #5111; calibration active                                                                                                                     |
| TOOLS-4249 | Theoretical review and professional polish                               | in-progress | #4249    | P2; architecture rulings ratified (#5289)                                                                                                                                   |
| TOOLS-4707 | Engineering design manual (TOOLS-D9 final subepic)                       | in-progress | #4707    | P2; D1–D8 complete; TOOLS-D9 (#4730) is final open subepic                                                                                                                  |
| TOOLS-4089 | Professional-grade SCADA (F01–F16)                                       | in-progress | #4089    | P2; historian licensing boundary ratified; H3/H6 unblocked                                                                                                                  |
| TOOLS-4142 | Variation and sensitivity analysis                                       | shipped     | #4142    | P2; Sobol indices, Spearman, landing ellipse normality merged (#5266)                                                                                                       |
| TOOLS-5218 | Putting launch monitor                                                   | in-progress | #5218    | P2; core + PyQt6 merged; launcher registration in review                                                                                                                    |
| TOOLS-4046 | Plant historian & analytics (TimescaleDB + Grafana)                      | in-progress | #4046    | P3; licensing boundary established; H3/H6 delivery unblocked                                                                                                                |
| TOOLS-4149 | Club Builder solid-CAD families                                          | parked      | #4149    | P3; no active delivery                                                                                                                                                      |
| DV-5068    | Epic: Shared Flexible Impact, Prestressed Shaft and Vibroacoustic Models | parked      | -        | DV-5068: [Owner Plan](https://github.com/D-sorganization/Tools/blob/main/docs/development/planning/DV-5068.md); Board decision pending; external prerequisites unavailable. |
| DV-4729    | [SUBEPIC][TOOLS-M12] Qualification and open release                      | parked      | -        | DV-4729: [Owner Plan](https://github.com/D-sorganization/Tools/blob/main/docs/development/planning/DV-4729.md); Board decision pending; external prerequisites unavailable. |
| DV-4267    | EPIC: Qualified Ball Landing, Bounce, Roll, and Ground-Surface Modeling  | parked      | -        | DV-4267: [Owner Plan](https://github.com/D-sorganization/Tools/blob/main/docs/development/planning/DV-4267.md); Board decision pending; external prerequisites unavailable. |

## Public-API Contract Policy Summary

Every module under `src/shared/python/sidekick/` and vendored packages (`theme`,
`plot_theme`, `golf_club`, `swing_sim`, `launch_monitor`, `contracts`, `safe_eval`)
maintain explicit `__all__` lists and AST-validated baselines in
`tests/sidekick_api_baseline.json` and `tests/api_baselines/`. A breaking change
requires a baseline regeneration in the same PR, a `!` marker in the PR title,
and linked migration issues in UpstreamDrift and Gasification_Model.

## Cross-Repo Dependencies

- **Downstream consumers:** UpstreamDrift, Gasification_Model (via symlink)
- **Tools is a leaf dependency.** No upstream fleet dependencies.
- Every release attaches a `ud_tools-<version>-py3-none-any.whl` wheel and
  CycloneDX SBOM; every push to `main` uploads a `tools-wheel-<sha>` artifact.

## Source Authority

The [planning catalog](../development/planning/catalog.json), source snapshots
and owner plans retain acceptance, evidence and reactivation rules. Update that
single authority before regenerating these project projections. Nothing here
supplies measurement, independent review or resource approval.
