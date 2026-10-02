# Fleet Critic — Tools — 2026-10-02

## Executive Summary

This pass found one **High**-severity finding that is not a documentation
weakness in the usual sense: a merged pull request already reverted a
published correction and a security fix back onto `main`, and the evidence
trail (a closed issue, a deleted regression test) now actively misrepresents
the current state of the code. The other two findings are conventional
Critic material — evidentiary gaps in otherwise careful impact-acoustics
documentation.

1. **High — PR #5400's stale-branch merge reverted PBKDF2 hardening (#5399)
   and the swing-objectives inertia correction (#5401/#5393); the regression
   test for the latter was deleted.** Filed as
   [#5404](https://github.com/D-sorganization/Tools/issues/5404) with the
   verifying `git diff` commands. See `weaknesses.md` §1.
2. **Medium — `IMPACT_INTERVAL_DYNAMICS.md`'s restitution calibration has no
   empirical anchor.** The Kelvin–Voigt damping coefficient is derived from a
   single-DOF linear-oscillator formula for a requested `e`, but no golf-ball
   force-deflection or COR-vs-speed data is cited to bound the error this
   linearization introduces against a real ball's markedly nonlinear,
   speed-dependent compliance. See `weaknesses.md` §2.
3. **Medium — `COMPLEX_FRF.md` cites no spectral-estimation literature.** A
   document that derives Welch-method FFT/coherence formulas from scratch and
   makes load-bearing claims about H1 bias and coherence interpretation cites
   only SciPy's own docs, not the primary statistical literature those
   methods and their failure modes come from. See `weaknesses.md` §3.

## What Would Change My Mind

- Finding 1 is settled by `git diff` on the cited commit ranges; it isn't an
  opinion. It would only soften if those three files have since been
  re-fixed by a commit landed after this report's HEAD (`5d18204ba`) — worth
  re-checking before acting on the filed issue if time passes.
- Findings 2–3 would soften with a citation or appendix the author points to
  that I missed; both docs are extensively self-critical elsewhere, so the
  gaps read as oversight rather than a pattern of overclaiming.

## Scope

Reviewed: all `docs/**` and `AGENT_HANDOFF.md`/`SPEC.md`/`CHANGELOG.md`
entries touched in the last 30 days (`git log --since="30 days ago"`), with
focus on the impact-acoustics program (`docs/physics/`,
`docs/development/impact-acoustics/`) and the swing-objectives realism specs
(`docs/specs/SWING_ACTUATION_AND_REALISM.md`), per the Tools-specific guidance
in the playbook (API contract clarity, cross-repo integration assumptions,
backward-compatibility and performance claims — the closest analogues here
are the model-boundary and evidence-limit claims these docs make about
themselves).
