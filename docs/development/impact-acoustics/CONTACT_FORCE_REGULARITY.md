# Contact-force regularity and acoustic qualification

This research note connects the first-contact reference in
`FRICTION_ENTRY.md` to the acoustic program in AffineDrift #4255 and
Tools #5073/#5074. It proposes requirements for subsequent model comparison;
it introduces no replacement contact law or acoustic qualification. The
canonical engineering manual remains `manuals/tools` QMD.

## Existing model and the observed boundary

The current normal-contact port uses a unilateral clipped Kelvin–Voigt law.
For positive compression delta, its raw force is k delta + c delta_dot;
negative raw force is clipped to zero. Force is zero at nonpositive
compression. At a closing first touch with finite incoming speed v0, the
incoming force limit is c v0, although the value assigned exactly at zero
compression is zero. The independent entry reference resolves this boundary
explicitly. Its illustrative incoming limit is about 0.734 N; that number
belongs to a synthetic low-speed test, not a fitted golf strike.

A finite force jump is not a Dirac impulse. Velocity, momentum and accumulated
work can remain continuous across the event. Assigning a different value at
the single event time does not change a continuous integral. An unresolved
finite timestep can nevertheless apply the wrong duration of force, and the
force waveform itself has high-frequency content associated with the jump.
Endpoint velocity/impulse convergence alone does not qualify that content.

## Spectral consequence: an analytical derivation

Use the Fourier convention Fhat(omega)=integral F(t) exp(-i omega t) dt.
For a compactly supported, piecewise smooth force with finitely many jumps
Delta F_j at t_j, its distributional derivative gives

```text
i omega Fhat(omega)
  = sum_j Delta F_j exp(-i omega t_j)
    + integral F'_regular(t) exp(-i omega t) dt.
```

If the regular derivative is integrable, its transform tends to zero.
The jumps therefore produce an oscillatory leading term proportional to
1/omega. Cancellation can create spectral zeros; this is not a monotone
amplitude law or a pointwise lower bound. Units agree: Fhat has N s and
Delta F/omega has N s. The derivation concerns the chosen force history,
independently of any particular acoustic solver or material.

If force is continuous and its first derivative has the corresponding bounded
variation/integrability properties, another integration gives an O(omega^-2)
bound. Stronger regularity can give faster decay, but smooth onset alone does
not establish the decay of the entire pulse. Release, friction transitions,
load discontinuities and the finite analysis-window endpoints can dominate.
The recording window can introduce its own jumps even when the physical
history is smooth. Windowing must be reported, not used to hide a model defect.

Two forces with the same impulse can therefore excite very different modes.
For example, equal-area rectangular and triangular pulses differ in force
regularity and spectral tails. Matching restitution, final ball velocity or
integrated work cannot by itself identify a force spectrum. These are
analytical implications, not measured player-to-player acoustic effects.

The acoustic chain still requires a spatial traction distribution, structural
response and a qualified radiation transfer to pressure. A force tail can be
attenuated or emphasized by modal participation, damping, radiation efficiency
and microphone geometry. It is not itself pressure, sound exposure, sharpness
or sweetness. Numerical or constitutive loss must not be relabelled as sound.

## Alternative contact laws and their limits

Simbody's documented sphere/sphere-plane Hunt–Crossley implementation uses
F=k delta^(3/2)(1+1.5 c delta_dot), with recoverable storage
(2/5) k delta^(5/2). It relates its dissipation parameter to the slope of
restitution versus speed in a low-speed approximation. This is a concrete
implementation precedent, with declared geometry and speed limitations,
not evidence that these coefficients apply to a multilayer golf ball and a
flexing clubface. The factor of 1.5 and the parameter definition must remain
attached to that convention. [1]

For finite incoming speed and positive indentation exponent, the generic
Hunt–Crossley form F=delta^n(k+lambda delta_dot) approaches zero at first touch.
That follows directly from the equation. Yet the multiplier can become
negative on sufficiently rapid unloading. Carvalho and Martins explicitly
report unwanted adhesion under external forcing and propose extensions and an
exponential alternative. Their institutional abstract supports this caution;
the full derivation has not been retrieved for this continuation. No claimed
exact restitution mapping is adopted from the abstract. [2]

The historical Hunt–Crossley paper is the foundational citation associated
with this family. Its publisher full text was not accessible in this lookup;
the equation and applicability statements above are taken from the directly
read Simbody documentation, not presented as a full-text review of 1975. [3]

An unforced, frictionless two-body restitution calibration must not silently
become a guarantee for an externally loaded, rotating, elastic shaft/grip
system. A coefficient fitted to one speed or one boundary condition needs
held-out force histories and speed/boundary variations. Clipping a candidate
law also needs a separate energy/storage audit; it is not merely a display
operation. Smoother force does not establish better agreement with a real ball.

## Integration requirements for the existing epics

1. Preserve the present law as a named baseline with explicit units, unilateral
   domains, storage, dissipation, cutoff handling and source identity. Inventory
   the existing normal-law protocol before introducing another abstraction.
2. Compare candidate continuous-onset laws through the canonical normal port;
   share the force/torque and shaft/grip mechanics. Require TDD controls for
   first touch, unloading, separation, external forcing, positive dissipation,
   nonadhesion and finite/invalid inputs. Do not duplicate contact mechanics.
3. Resolve event times and per-port work before using force histories for
   acoustics. Retain independent reference and timestep convergence for
   impulse, force peaks, waveform norms and phase as separate quantities.
4. Use analytical prescribed pulses to qualify Fourier normalization, units,
   windowing and expected regularity effects. Then compare coupled-model
   spectra on a declared frequency band. Sampling and timestep studies must
   expose aliasing; a fixed audio sample rate is not an integrator certificate.
5. Carry force-law identity and calibrated bandwidth into modal/radiation
   studies. Vary contact patch and strike position; compare head/face modes,
   grip coupling, prestress and initial elastic state without double-counting
   rotational terms or attributing their effects solely to a scalar mass.
6. Identify force-history parameters from physical impact data and validate
   held-out speeds, balls, delivery and grip conditions. Separate force/FRF,
   pressure and blinded listening evidence. A better-sounding synthesized
   waveform is not a successful physical model fit.

These requirements belong in the existing contact, radiation and synthesis
epics; they do not warrant closing any physical/perceptual acceptance item.
The current first-contact numerical result supplies a concrete reason to
investigate onset regularity, not a conclusion about why one player sounds
better than another.

## Sources and access boundaries

1. Simbody 3.7 documentation, _SimTK::HuntCrossleyForce Class Reference_,
   normal-force components and applicability, generated 2019; accessed
   2026-09-10.
   https://simbody.github.io/3.7.0/classSimTK_1_1HuntCrossleyForce.html
2. André S. Carvalho and Jorge M. Martins (2019), _Exact restitution and
   generalizations for the Hunt–Crossley contact model_, Mechanism and Machine
   Theory 139, 174–194, DOI 10.1016/j.mechmachtheory.2019.03.028.
   Institutional abstract/metadata read; full text unavailable in this lookup.
   https://researchportal.ulisboa.pt/pt/publications/exact-restitution-and-generalizations-for-the-huntcrossley-contac/
3. K. H. Hunt and F. R. E. Crossley (1975), _Coefficient of Restitution
   Interpreted as Damping in Vibroimpact_, Journal of Applied Mechanics 42(2),
   440–445, DOI 10.1115/1.3423596. Historical bibliographic reference retained
   through the directly read implementation documentation; publisher full text
   not retrieved here. https://doi.org/10.1115/1.3423596
