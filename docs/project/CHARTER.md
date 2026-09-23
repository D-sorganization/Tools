# CHARTER.md — Tools

> Proposed by Project Steward 2026-09-23. Derived from README, SPEC.md, open epics,
> and `AGENT_HANDOFF.md`. Update this file when goals or feature statuses change.

## Mission

Tools is the fleet's shared engineering library and desktop toolset, consumed by
UpstreamDrift and Gasification_Model. It delivers signal processing, URDF
generation, process calculators, P&ID utilities, biomechanics analysis, and
visualization themes under one PyQt6 launcher, with React/FastAPI web mirrors
for key tools. Every public API change here is a potential breaking change for
downstream consumers.

## Goals

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

## Feature Register

| Feature | Epic | Priority | Status |
| ------- | ---- | -------- | ------ |
| Self-hosted runner security hardening | #4464 | P0 | **open** — no PR; requires Board decision on fork-PR approval policy |
| Swing-Impact-Ball-Flight Platform (`rate_of_closure`) | #4103 | P1 | **in_progress** — workspace compositor shipped (#5264); camera cluster (#4571) and WASM/Pages CI (Phase 7) remain |
| Visual-first tab UX (PyQt6 + React) | #4433 | P1 | **in_progress** — 8 verified / 23 partial; acceptance manifests landing |
| Rate of Closure production distribution | #4377 | P2 | **in_progress** — Release A shipped; Release B requires real paired launch-monitor data |
| Site-wide Golf Impact & Flight contract authority | #4260 | P2 | **in_progress** — impact-interval PyQt tab ruled out (#4946); contract authority under T4/T5 of #5068 |
| Shared flexible impact, shaft, and vibroacoustic models | #5068 | P2 | **in_progress** — T1–T4 merged; grip qualification shipped (#5297–#5301); physical evidence outstanding |
| Qualified ball landing, bounce, roll, and ground models | #4267 | P2 | **in_progress** — deferred physical validation preserved in planning catalog |
| Vendor-neutral markerless mocap | #4706 | P2 | **in_progress** — reconstruction core in PR #5111; calibration active |
| Theoretical review and professional polish | #4249 | P2 | **in_progress** — architecture rulings ratified (#5289) |
| Engineering design manual (TOOLS-D9 final subepic) | #4707 | P2 | **in_progress** — D1–D8 complete; TOOLS-D9 (#4730) is final open subepic |
| Professional-grade SCADA (F01–F16) | #4089 | P2 | **in_progress** — historian licensing boundary ratified; H3/H6 unblocked |
| Variation and sensitivity analysis | #4142 | P2 | **shipped** — Sobol indices, Spearman, landing ellipse normality merged (#5266) |
| Putting launch monitor | #5218 | P2 | **in_progress** — core + PyQt6 merged; launcher registration in review |
| Plant historian & analytics (TimescaleDB + Grafana) | #4046 | P3 | **in_progress** — licensing boundary established; H3/H6 delivery unblocked |
| Club Builder solid-CAD families | #4149 | P3 | **parked** — no active delivery |

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
