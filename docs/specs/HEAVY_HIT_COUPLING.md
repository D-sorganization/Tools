# Heavy Hit — Hand/Body Coupling at Impact

Status: **active** · Epic: heavy-hit (GitHub epic issue) · Owner: shared
(`golf_club`, `swing_sim`) · Related: `CLUB_FITTING_TESTER.md`,
`swing_sim/impact/` (Kelvin-Voigt contact), `delivery_interchange` (C5)

## 1. The question

During the ~500 µs of club-ball contact, how much can the golfer's hands and
body actually change the impact — and therefore, how *separate* is the impact
model from whatever multibody system drives it? The classical claim (Cochran &
Stobbs; Jorgensen) is that the head behaves as a nearly free body: flexural
waves cannot travel grip-ward and return within the contact window. This epic
**quantifies** that claim with a transient coupled model and counterfactuals,
and makes the answer computable for *any* golfer model exported from the
engines UpstreamDrift features — MuJoCo, Drake, OpenSim, Pinocchio.

## 2. Architecture

```
H2  model interchange   swing_sim/model_interchange/: body-chain wire
                        `swing_sim.body_chain/1` + MJCF / URDF / OSIM parsers
                        (runtime-free XML; URDF covers Drake AND Pinocchio)
                        → GripBoundary {effective mass, stiffness, damping}
H1  coupled transient   golf_club/impact_coupling.py: ball–head–hands lumped
                        chain, Kelvin-Voigt contact (constants reused from
                        swing_sim.impact), semi-implicit integration at the
                        impact model's dt; free-head / welded limits
H3  counterfactuals     grip-stiffness/mass sweeps → decoupling fraction →
                        `golf_club.impact_coupling_report/1` (deterministic)
H4  surfaces            GUI panels (follow-on children, after the club-tester
                        C6 pattern lands)
```

## 3. H1 — the coupled transient model (upper-bound semantics)

Lumped longitudinal chain along the hit direction:

```
ball m_b ←KV contact (k_c, c_c)→ head m_h ←shaft (k_s, c_s)→ hands m_g ←grip (k_g, c_g)→ body (rigid)
```

- Contact spring/damper are **the impact package's own** Kelvin-Voigt
  parameters (`ImpactParameters.contact_stiffness/damping`), so the
  free-head limit of this model and `SpringDamperImpactModel` agree — a
  consistency gate, not a coincidence.
- **Upper-bound semantics, stated loudly:** at contact timescales the shaft
  transmits force through its local impedance, and any lumped `k_s` is an
  approximation. The model therefore reports hand influence with `k_s`
  swept up to a **rigid-link bound** — "even if the shaft were perfectly
  rigid, the hands change ball speed by X%" — and with the static tip
  stiffness (from `solve_cantilever_tip_response`, reused) as the realistic
  low end. Reality lies below the rigid bound; the epic's headline number
  is that bound.
- Integration: semi-implicit Euler at `dt = 1e-7 s` (the spring-damper
  impact model's step), from first contact until the contact force returns
  to zero; ball exit velocity and the energy split (ball / head / shaft
  spring / grip spring) are reported.

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

## 4. H2 — importing golfer models from the engines

Wire `swing_sim.body_chain/1`: an ordered chain of bodies
`{name, mass_kg, inertia_diag_kg_m2, joint: {name, type, axis, stiffness_nm_rad | n_m, damping}}`
rooted at the declared attachment. Fail-closed parsing, deterministic
serialization — the C5 posture.

Parsers are **runtime-free XML readers** of each engine's native model
format (no engine imports, fixture-tested):

| Engine | Format | Parser | Notes |
| --- | --- | --- | --- |
| MuJoCo | MJCF `<body>/<inertial>/<joint>` | `chain_from_mjcf` | joint `stiffness`/`damping` native |
| Drake | URDF `<link>/<joint>` | `chain_from_urdf` | Drake loads URDF natively |
| Pinocchio | URDF | `chain_from_urdf` | same parser, documented |
| OpenSim | `.osim` `<Body>` | `chain_from_osim` | joint stiffness not native → 0 + explicit override |

`grip_boundary(chain, hand_bodies=..., wrist_joint=...)` reduces a chain to
`GripBoundary{effective_mass_kg, stiffness_n_m, damping_n_s_m, provenance}`
— the selection is **explicit** (caller names the hand-side bodies and the
boundary joint); nothing is guessed from names.

## 5. H3 — counterfactual quantification

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
