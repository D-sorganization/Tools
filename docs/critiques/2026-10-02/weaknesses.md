# Weaknesses — Tools — 2026-10-02

## Weakness: A merged PR reverted a security fix and a published scientific correction, and the evidence trail now lies about it

### Summary of Concern

`main` currently (HEAD `5d18204ba`) carries the pre-fix, already-debunked
version of the swing-objectives inertia-equivalence claim and the pre-fix,
weak PBKDF2 iteration count, even though both were fixed by reviewed PRs
one commit earlier. Issue #5393, whose fix this undid, is closed — so the
tracker says the defect is resolved while the code says otherwise.

### Location

- **File/Document:** `docs/specs/SWING_ACTUATION_AND_REALISM.md`;
  `src/pendulum_simulator/src/double_pendulum_golf/swing_objectives/club_equivalence.py`;
  `src/pendulum_simulator/tests/test_club_equivalence.py`;
  `src/folder_packer_pro/encryption.py`
- **Section:** Correction banner and §2a of the spec; `EncryptionManager.derive_key`
- **Claim or Equation:** The committed correction claimed the equivalent-club
  result sits "inside the heuristic reference intervals" (corrected framing)
  and documented the butt-vs-wrist-pivot measurement distinction; `main` has
  reverted to "inside the measured bands" (the overclaiming framing the
  correction existed to fix) and dropped that distinction entirely. Separately,
  `derive_key` is back to a hardcoded `iterations=100000` after PR #5399 raised
  it to 600,000.

### Nature of the Issue

- [x] Empirical insufficiency (the regression test proving the inertia
  arithmetic is gone, so nothing in CI would catch a recurrence)
- [x] Overgeneralization (the reverted text re-asserts "inside the measured
  bands" — a claim against literature-measured bounds — that the correction
  explicitly identified as false)
- [ ] Logical gap
- [ ] Unstated assumption
- [ ] Terminological ambiguity
- [ ] Literature conflict

### Why This Is a Problem

A reviewer who diffs `main` against the closed issue #5393 finds the
documented and coded claim is, right now, the one the issue says was fixed.
Anyone citing `SWING_ACTUATION_AND_REALISM.md` today — including a future
Critic pass — would unknowingly cite the debunked version. The PBKDF2 drop is
independently a live security regression (weak key-derivation iteration
count), reintroduced by the same merge. The root cause (PR #5400, a
stale-branch "Bolt" automation merge whose single parent is `b005eedec` but
whose diff reverts `cabb44914` and `b005eedec`'s content) is exactly the class
of problem this repo's own fleet-guard `stale-push` rule targets — it just
wasn't caught because the Bolt PR's own diff is internally consistent and
touches files its stated scope (`LaunchMonitorComparisonWorkspace.tsx`) never
mentions.

### Evidence / References

- `git diff cabb44914 HEAD -- src/folder_packer_pro/encryption.py` (iterations
  600000 → 100000)
- `git diff b005eedec HEAD -- docs/specs/SWING_ACTUATION_AND_REALISM.md`
  (corrected text reverted)
- `git diff b005eedec HEAD -- src/pendulum_simulator/tests/test_club_equivalence.py`
  (`test_inertia_arithmetic_distinguishes_masses` deleted)
- PR #5400 (merge), PR #5399 (PBKDF2 fix), PR #5401 / issue #5393 (inertia
  correction)
- Filed as [issue #5404](https://github.com/D-sorganization/Tools/issues/5404)

### Severity

- [ ] Low
- [ ] Medium
- [x] High (a live security weakening plus a reinstated, closed-as-fixed
  scientific defect)

### Suggested Remedies

- Re-apply #5399 and #5401 (doc, code, and the deleted test) on top of
  current `main` in one PR; reference both #5393 and the new #5404.
- Reopen #5393 or note in it that the fix regressed, so the tracker doesn't
  keep reporting the defect as resolved.
- Audit the other automated PRs merged in the same window (#5397, #5398,
  #5399 — all same-day "Bolt"/"Sentinel"/"Palette" bot PRs) for the same
  stale-base pattern; confirm whether `main`'s branch protection enforces
  "up to date before merge" for these bot accounts.

---

## Weakness: The restitution-law linearization in `IMPACT_INTERVAL_DYNAMICS.md` has no empirical error bound

### Summary of Concern

The document derives a Kelvin-Voigt damping coefficient from a classical
single-degree-of-freedom linear-oscillator formula,
`zeta = -ln(e) / sqrt(pi^2 + (ln e)^2)`, `c = 2*zeta*sqrt(k*m*)`, to hit a
requested coefficient of restitution `e`. This supplies "a reproducible
instantaneous-limit target" but the document never states how far this
linear approximation departs from a real golf ball's force-deflection
behavior, which is well known in the literature to be markedly nonlinear and
speed-dependent (compression-dependent core stiffness, COR that falls with
impact speed).

### Location

- **File/Document:** `docs/physics/IMPACT_INTERVAL_DYNAMICS.md`
- **Section:** "Moving Contact Point and Force Law" (lines ~68–81)
- **Claim or Equation:** `c = 2*zeta*sqrt(k*m*)` calibrated to a target `e`;
  ball modeled with "the existing uniform-solid-sphere inertia approximation"

### Nature of the Issue

- [ ] Logical gap
- [x] Unstated assumption (that a single linear `(k, c)` pair calibrated to
  one `e` value generalizes across the impact-speed range the solver is used
  for)
- [ ] Overgeneralization
- [x] Empirical insufficiency (no comparison to measured golf-ball
  force-deflection or COR-vs-speed curves)
- [ ] Terminological ambiguity
- [ ] Literature conflict

### Why This Is a Problem

The document is otherwise unusually careful about stating what it has and
has not validated — it explicitly calls out that "nonlinear/viscoelastic ball
constitutive laws... remain tracked extensions" and that its absence "must
not be hidden." But the restitution calibration itself is presented as "a
reproducible instantaneous-limit target" without the same disclosure: a
reviewer could reasonably read the Kelvin-Voigt/`zeta` derivation as settled
physics rather than a linearization whose error against a real ball's
nonlinear compliance is unquantified. The uniform-solid-sphere inertia
approximation for a dimpled, multi-layer, non-uniform-density ball is a
similar unquantified simplification sitting in the same paragraph.

### Evidence / References

- USGA/R&A ball COR and compression testing protocols document markedly
  speed-dependent restitution for modern multi-piece golf balls (COR drops
  measurably from ~90 mph to ~160 mph head speed ranges), which a
  single-`e`-point linear calibration cannot represent across a swept speed
  range.
- The document's own §"Purpose and Publication Boundary" already disclaims
  "nonlinear/viscoelastic ball constitutive laws" as a tracked extension —
  this finding asks that the restitution-calibration paragraph carry the same
  explicit caveat, rather than reading as a settled target.

### Severity

- [ ] Low
- [x] Medium (argument tightening required — the document's own disclosure
  standard, applied elsewhere in the same section, is not applied here)
- [ ] High

### Suggested Remedies

- Add one sentence to the restitution-calibration paragraph stating the
  linear `(k, c)` pair is calibrated at a single `e`/speed point and is not
  validated against measured nonlinear force-deflection or speed-dependent
  COR data.
- Cite a specific ball-compliance dataset (even if only to say none is yet
  wired into the solver) alongside the existing citation-style treatment the
  rest of the document uses for its other open items.

---

## Weakness: `COMPLEX_FRF.md` cites no primary spectral-estimation literature

### Summary of Concern

The document derives Welch-method PSD, H1 transfer-function, and coherence
estimators from scratch, including specific bias/interpretation claims
("Correlated disturbances... violate that inference," "segment count is not
effective degrees of freedom"). These are standard results from the random
signal analysis literature, but the only citations in the entire document
are to SciPy's own API reference pages, not to the statistical literature
those implementations and their known failure modes originate from.

### Location

- **File/Document:** `docs/development/impact-acoustics/COMPLEX_FRF.md`
- **Section:** "Definitions, dimensions and implementation";
  "Identification limits and required experiments"
- **Claim or Equation:** H1/coherence formulas and their bias conditions
  (lines ~30–70, ~90–105)

### Nature of the Issue

- [ ] Logical gap
- [ ] Unstated assumption
- [ ] Overgeneralization
- [ ] Empirical insufficiency
- [x] Terminological ambiguity (claims are stated as self-evident derivations
  rather than attributed results, making it hard for a reader to check them
  against the source theory)
- [x] Literature conflict (not a conflict in content, but an absence that
  prevents a reader from checking for one)

### Why This Is a Problem

Every sibling document in this program
(`SWING_ACTUATION_AND_REALISM.md`, `MEASURED_GRIP_IMPEDANCE.md`) cites named,
dated, linked primary sources for every load-bearing claim, including a test
that enforces resolvable links for the reference-kinematics bands. This
document holds itself to a visibly lower citation standard while making
claims (H1 bias under correlated noise, coherence degrees-of-freedom
limitations) that are exactly the kind of result a hostile reviewer would ask
"according to whom?" about. SciPy's docstring is not a citable primary
source for the statistical theory — it is a description of an
implementation of that theory.

### Evidence / References

- Compare citation density/style against
  `docs/specs/SWING_ACTUATION_AND_REALISM.md` §2/§3 and
  `docs/development/impact-acoustics/MEASURED_GRIP_IMPEDANCE.md`, both of
  which cite primary sources for every quantitative claim.
- The standard primary reference for exactly these claims is Bendat, J. S. &
  Piersol, A. G., *Random Data: Analysis and Measurement Procedures* (Wiley),
  and Welch, P. (1967), "The use of fast Fourier transform for the estimation
  of power spectra," *IEEE Trans. Audio Electroacoust.* 15(2), 70–73 —
  neither is cited.

### Severity

- [x] Low (clarification/citation needed; the derivations themselves are not
  disputed)
- [ ] Medium
- [ ] High

### Suggested Remedies

- Add Bendat & Piersol and Welch (1967) to a references section, matching
  the citation style already used elsewhere in the impact-acoustics program.
- Where a specific bias claim is made (e.g., H1 vs. H2 estimator bias under
  output noise), attribute it to the specific result in the literature
  rather than presenting it as self-evident.
