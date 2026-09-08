# Heavy Hit — Hand/Body Coupling at Impact

> Scientific scope correction, #5068: this is the historical specification of
> the initially relaxed one-dimensional chain. Its rigid-link and sub-percent
> comparisons are fixture-specific, not universal physical bounds. The v1
> `decoupling_fraction` measures clipped relative ball-speed difference, not a
> body mass contribution. [The new research specification](IMPACT_DYNAMICS_ACOUSTICS.md)
> and #5071 govern reconciliation of the historical claims and event/energy
> assumptions below; the legacy wire remains unchanged.

Status: **active** · Epic: heavy-hit (GitHub epic issue) · Owner: shared
(`golf_club`, `swing_sim`) · Related: `CLUB_FITTING_TESTER.md`,
`swing_sim/impact/` (Kelvin-Voigt contact), `delivery_interchange` (C5)

## 1. The Question

How can the declared hand/shaft boundary change a one-dimensional collision?
This model answers that bounded mathematical question. An XML body reduction
is not measured dynamic impedance, and one wave-transit estimate cannot prove
that every shaft or hand mechanism is absent. The distributed, rotating and
acoustic extensions are tracked in #5068; no universal percentage is inferred.
During the ~500 µs of club-ball contact, how much can the golfer's hands and
body actually change the impact — and therefore, how _separate_ is the impact
model from whatever multibody system drives it? The classical claim (Cochran &
Stobbs; Jorgensen) is that the head behaves as a nearly free body: flexural
waves cannot travel grip-ward and return within the contact window. This epic
**quantifies** that claim with a transient coupled model and counterfactuals,
and makes the answer computable for _any_ golfer model exported from the
engines UpstreamDrift features — MuJoCo, Drake, OpenSim, Pinocchio.

## 2. Architecture

```
H2  model interchange   swing_sim/model_interchange/: body-chain wire
                        `swing_sim.body_chain/1` + MJCF / URDF / OSIM parsers
                        (runtime-free XML; URDF covers Drake AND Pinocchio)
                        → GripBoundary {effective mass, stiffness, damping}
H1  coupled transient   golf_club/impact_coupling.py: ball–head–hands lumped
                        chain, Kelvin-Voigt contact (constants reused from
                        swing_sim.impact), event-resolved integration with
                        declared maximum dt; free-head / stiff comparisons
H3  counterfactuals     grip-stiffness/mass sweeps → decoupling fraction →
                        `golf_club.impact_coupling_report/1` (deterministic)
H4  surfaces            GUI panels (follow-on children, after the club-tester
                        C6 pattern lands)
```

## 3. H1 — The Coupled Transient Model (Legacy Bound Semantics)

Lumped longitudinal chain along the hit direction:

```
ball m_b ←KV contact (k_c, c_c)→ head m_h ←shaft (k_s, c_s)→ hands m_g ←grip (k_g, c_g)→ body (rigid)
```

- Contact spring/damper are **the impact package's own** Kelvin-Voigt
  parameters (`ImpactParameters.contact_stiffness/damping`), so the
  free-head limit of this model and `SpringDamperImpactModel` agree — a
  consistency gate, not a coincidence.
- **Comparison semantics:** a stiff-link sweep is a specified fixture, not a
  universal upper bound. Preload, damping, modal history and three-dimensional
  motion can change the ordering. Static tip stiffness is not a measured
  contact-band impedance.
- Integration uses adaptive DOP853 with `dt_s` as maximum step, local relative
  tolerance 2e-9 and absolute tolerance 1e-11. A conservative mass-normalized
  stiffness/damping rate safeguard refuses grossly under-resolved steps.
  This safeguard is not a proof that every possible grazing event is resolved;
  independent step refinement remains necessary.
- First touch is followed to geometric overlap clearance, preserving the v1
  terminal convention. First force release can occur earlier. Timeout is an
  explicit failure, never a successful result. Peak force is sampled at accepted
  steps plus its first-touch right-hand limit; refine dt for peak accuracy.
- `impact_coupling_audit.audit_coupled_impact` adds initial displacement/velocity
  and a complete energy ledger without changing the report v1 fields. The
  translating reference is inertial; it is not a rotating/accelerating body frame.

**Analytic/consistency gates (TDD):** undamped detached duration and impulse;
clipped Kelvin-Voigt force release and geometric clearance; passive damping and
cutoff energy; preload initial energy; step refinement; timeout and resolution
refusal; existing detached-model parity and deterministic v1 report shape.
The old sub-percent/stiff-link tests remain synthetic fixture regressions.
The finite relaxed-spring example exhibits quadratic short-time influence;
shaft damping and preload fixtures exhibit first-order influence. None of
these fixtures identifies a physiological parameter or proves global ordering.

### Energy and Event Convention

With overlap `delta = x_head - x_ball` and rate `u = v_head - v_ball`,
`F = max(k_c * delta + c_c * u, 0)` for positive overlap, otherwise zero.
The coupled model is uncapped. `KelvinVoigtContactLaw` also supports a force cap
for other callers; a capped model is a different law and is not qualified by
this audit's energy equations.

Let `U_c = k_c * max(delta, 0)^2 / 2`. During active force, the contact loss rate
is `c_c * u^2`. During clipped unloading it is `-k_c * delta * u`, which is
nonnegative and accounts for loss of remaining contact potential. Shaft and
grip losses are respectively `c_s * (v_grip-v_head)^2` and `c_g * v_grip^2`.
The fixed anchor does no work. The audit reports
`initial energy + boundary work - terminal mechanical energy - all losses`
as a numerical residual; it never renames numerical drift as damping.

The v1 `energy_balance_fraction` remains terminal retained mechanical energy
(over all three masses and shaft/grip springs) divided by initial energy. It
is less than one in dissipative cases and is not itself an energy-closure error.
The v1 `contact_time_s` remains clearance time. `decoupling_fraction` remains
`clip(1 - abs(v_ball-v_free)/v_free, 0, 1)`, a speed discrepancy, not mass.

The conventional damped-oscillator restitution-to-damping formula assumes the
untruncated linear oscillator's half period. It is not an exact inversion of
force-clipped Kelvin-Voigt restitution. Use the actual event law when comparing
or calibrating restitution; never transfer one fixed restitution across changed
reduced masses or changed cutoff/cap conventions.

**Gates (analytic/consistency, TDD):**

1. Detached limit (`k_s = 0`) reproduces `SpringDamperImpactModel`'s ball
   exit speed for identical contact parameters (tight tolerance).
2. Welded-rigid limit (`k_s, k_g → large`) approaches the infinite-mass
   two-body bound `v_ball → (1+e)·v_head` from below; monotone in `k_g`.
3. **Decoupling law**: influence shrinks as contact duration shrinks
   (stiffer contact ⇒ less hand influence) — monotonicity gate.
4. Energy conservation within integration tolerance at zero damping.
5. Physiological inputs (grip stiffness ~1e4–1e5 N/m, hand+forearm mass
   ~2–4 kg) yield **sub-percent** ball-speed influence — the quantified
   classical claim, asserted as a band, with the rigid-shaft upper bound
   also reported.

## 4. H2 — Importing Golfer Models From the Engines

Wire `swing_sim.body_chain/1`: an ordered chain of bodies
`{name, mass_kg, inertia_diag_kg_m2, joint: {name, type, axis, stiffness_nm_rad | n_m, damping}}`
rooted at the declared attachment. Fail-closed parsing, deterministic
serialization — the C5 posture.

Parsers are **runtime-free XML readers** of each engine's native model
format (no engine imports, fixture-tested):

| Engine    | Format                           | Parser            | Notes                                              |
| --------- | -------------------------------- | ----------------- | -------------------------------------------------- |
| MuJoCo    | MJCF `<body>/<inertial>/<joint>` | `chain_from_mjcf` | joint `stiffness`/`damping` native                 |
| Drake     | URDF `<link>/<joint>`            | `chain_from_urdf` | Drake loads URDF natively                          |
| Pinocchio | URDF                             | `chain_from_urdf` | same parser, documented                            |
| OpenSim   | `.osim` `<Body>`                 | `chain_from_osim` | joint stiffness not native → 0 + explicit override |

`grip_boundary(chain, hand_bodies=..., wrist_joint=...)` reduces a chain to
`GripBoundary{effective_mass_kg, stiffness_n_m, damping_n_s_m, provenance}`
— the selection is **explicit** (caller names the hand-side bodies and the
boundary joint); nothing is guessed from names.

## 5. H3 — Counterfactual Quantification

`impact_coupling_report(...)`: baseline free-head vs coupled outcomes over a
declared grid of `(k_g, m_g, k_s)` counterfactuals, each with ball-speed
delta, launch-relevant impulse ratio, and the **decoupling fraction**
`1 − |Δv|/v_free`. Deterministic sorted-keys JSON
(`golf_club.impact_coupling_report/1`), byte-identical for identical inputs;
carries the `GripBoundary` provenance so an OEM report says which engine
model produced it.

## 6. Standards

Shared-first placement; DbC on every public function; LoD (no >2-level
chains); DRY (contact constants, shaft stiffness, wire idioms all reused);
TDD with analytic gates preceding implementation-tuned assertions; every
wire versioned + fail-closed; SPEC row and handoff update per PR.
