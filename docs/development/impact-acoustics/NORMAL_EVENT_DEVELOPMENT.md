# Adaptive Spatial Normal Contact

Parent #5073 continues on `feat/5073-contact-events`, from temporal implementation
37e322611. The fixed-step temporal foundation is published separately in
[PR #5149](https://github.com/D-sorganization/Tools/pull/5149), review child #5147.
Review child #5151 is created and leased; a PR is not yet published.
This continuation is not yet a reviewed or physically qualified delivery.

The adaptive method evolves local SE(3) coordinates, material twists and the
existing five work channels with DOP853, resetting charts at requested output
times. It shares the body-Jacobian differential with the existing Lie RK4 and
queries the canonical sphere/plane geometry without invoking grip histories
during geometric root searches. Every constitutive response, including root
and endpoint reports, counts against a hard budget. Position, angle, linear
velocity, angular velocity and work have distinct SI absolute tolerances.

The method records first touch, force release, geometric separation and force
reactivation candidates. A raw-force zero in clearance is excluded. The
one-sided dashpot limit at first touch is reported explicitly and checked
against the force ceiling. It is not a certificate of the force maximum.
Sign-change root searches can miss multiple crossings in one accepted step;
grazing and repeated-root completeness remain outside this evidence. See the
[SciPy solve_ivp event and tolerance contracts](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).
The body differential retains the explicit convention in
[Müller's corrected Lie-group integration review](https://arxiv.org/pdf/2303.07928).

## TDD and Independent Reference

The missing-module RED is retained as `impact-normal-events-red.xml`. The first
implementation passes six controls and fails when dense work interpolation
produces a negative loss estimate near onset. That failure is retained in
`impact-normal-events-green.xml`. No negative loss is clipped or relabelled.
Root records now contain state and constitutive response only; work remains
strictly validated at integration endpoints. This limitation is explicit in
the root contract. All seven initial controls then pass in 51.86 s.

An independent piecewise matrix-exponential three-mass model starts in
clearance, with existing moving-grip polynomial inputs. Brent roots of its
analytic segment solutions give first touch at 0.367303647790 ms, force release
at 2.585395772284 ms and geometric separation at 2.728659763437 ms. Work is
integrated independently by segment quadrature. These coefficients and states
are synthetic; none is fitted to golf measurements.

The expanded 52-control Windows run passes in 50.47 s (JUnit 49.856 s), with
two JUnit record-property format warnings and no failures or skips. Across
relative tolerances 1e-6, 1e-8 and 1e-10 and step ceilings 0.2, 0.1 and 0.05 ms,
state errors are 6.173e-6, 3.625e-10 and 2.508e-11. Fine-grid event errors are
below 2e-9 s, work errors below 2e-9 J and the uncorrected final energy defect
is -3.443e-12 J. Limits were fixed before implementation. This run also retains
the deliberate missing/wrong-sign Jacobian controls, nonplanar instantaneous
power tests, a separated spinning ball, missing-history/ceiling/budget refusal,
SI tolerance domains and incomplete-solver refusal.

Four production modules pass NumPy-aware mypy. API RED concerns only the two
new private modules; ordinary baseline regeneration passes both package tests.
Independent comparison confirms all 101 existing golf-club and 234 existing
swing API records are unchanged. Exact source tree 4e1b198104dd82634c1eeecc0f1805c998299227
passes 386 Windows controls (JUnit 295.035 s) and 386 Linux controls (JUnit
269.206 s), with no failures or skips. Linux coverage is 58.64%, exceeding the
unchanged 20% floor. Source/JUnit hashes and independent work/event results
are in NORMAL_EVENT_RESULTS.json. Final governance, protected CI and review
remain required.

## Publication Typing Repair

The first normal push at50935aefc refused three no-any-return errors under
the isolated follow-imports=skip configuration. Explicit response typing and
builtin scalar return conversions pass that same hook. All15 affected event
controls pass in28.94s; equations, tolerances and gates are unchanged. The
386-test archive remains attributed to its original tree; this later repair
has its own JUnit identity in NORMAL_EVENT_RESULTS.json.

## Remaining Scientific Obligations

This is normal-only contact with fixed material/load laws and prescribed grip
anchors. Finite-duration tangential history, face/hosel modes, changing applied
force/torque histories, independent mesh/mode/event convergence, matched-state
interventions, measured calibration, acoustic radiation and blinded perception
remain required. A small numerical energy residual cannot measure acoustic
energy or establish a sweeter or heavier hit.
