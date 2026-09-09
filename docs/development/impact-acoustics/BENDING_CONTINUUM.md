# Bending, Shear and Rotary-Inertia Verification: Tools #5072

This checkpoint adds independent verification of the published spatial
shaft/grip operators and Galerkin reduction. Production calculations and API
entries are unchanged. A continuum boundary-value solution supplies a reference
independent of the element assembly, followed by separate mesh and modal error
checks. It is a synthetic stationary planar control, not an identified club.

## Continuum and boundary derivation

Let w(s,t) be transverse displacement and theta(s,t) the section rotation,
with s in [0,L] along +z and theta about +y. The shear strain is w'-theta;
Q=S(w'-theta) and M=B theta', where S is shear rigidity and B=EI. Define line
mass mu and transverse rotary inertia per length j. The kinetic and elastic
energies are

```text
T = integral_0^L (mu*wdot^2 + j*thetadot^2)/2 ds
U = integral_0^L (S*(w'-theta)^2 + B*theta'^2)/2 ds.
```

Virtual work gives mu*wtt=Q' and j*thetatt=M'+Q. For exp(+i*omega*t), define
the spatial state a=[w,theta,Q,M]. It obeys a'=A a with

```text
A = [[0, 1, 1/S, 0],
     [0, 0, 0, 1/B],
     [-mu*omega^2, 0, 0, 0],
     [0, -j*omega^2, -1, 0]].
```

At the root, internal [Q,M]=Dg [w,theta], where
Dg=Kg+i*omega*Cg-omega^2*Bg is the stationary grip's dynamic stiffness.
At the tip, internal [Q,M]-omega^2*Mh [w,theta]=[F,tau]. The latter sign
retains the attached head's acceleration; it is not an added static stiffness.

For P=exp(A L)[I;Dg], split P into upper displacement block Pu and lower
internal-force block Pf. The tip compliance is

```text
H = Pu * solve(Pf-omega^2*Mh*Pu, I).
```

The reference evaluates this four-state continuum equation with SciPy expm;
the finite-element solver assembles its existing SE(3) section and inertia
operators. They share a numerical library but not a discretization or assembled
matrices. This reference is bounded to the stated fixture samples. It does not
supply a high-frequency conditioning or certified forward-error guarantee.

Khasawneh and Segalman, _Exact and Numerically Stable Expressions for
Euler-Bernoulli and Timoshenko Beam Modes_ (2019), equations 8–10, provides the
underlying coupled field equations and force definitions. Their paper also
explains numerical difficulties with high-mode beam expressions. This test
uses the field equations with the explicit dynamic boundaries above; it does
not claim their stabilized eigenmode algorithm.
[Primary paper](https://arxiv.org/pdf/1811.03222).

## Fixture, units and independent static check

The existing unit-length synthetic rod supplies S=500 N, B=10 N m²,
mu=0.2 kg/m and j=1e-4 kg m. These coefficients are independent inputs; no
isotropic modulus or cross-section geometry is inferred. At the bending root,
Kg=diag(300 N/m,50 N m/rad), Cg=diag(1.5 N s/m,0.03 N m s/rad), and
Bg=diag(0.03 kg,0.0008 kg m²). Mh=diag(0.1 kg,0.005 kg m²) comes from the
existing principal-axis head at the tip, with zero COM offset. Other shaft
directions remain present in the assembled model; they decouple only under
this straight, stationary, principal-axis prescription.

The zero-frequency reference is first checked independently by integrating
the static shear force and linearly varying bending moment:

```text
H(0) = [[1/300 + 1/50 + 1/30 + 1/500, 1/50 + 1/20],
        [1/50 + 1/20,                  1/50 + 1/10]].
```

Columns are tip force and tip couple; rows are tip displacement and rotation.
Setting the reference shear compliance to zero removes exactly 1/500 from
the force/displacement entry at zero frequency. This checks the shear term
and boundary signs; it does not justify ignoring shear dynamically.

At each sample, reciprocity H=H.T and nonnegative Hermitian part of i*omega*H
check the passive collocated response. The error metric is the maximum of
`abs(Hfe_ij-H_ij)/abs(H_ij)` over the four components. Each ratio is
dimensionless; a norm mixing untranslated force/moment units is not used.
The reference components exceed an explicit 1e-8 fixture amplitude floor at
these samples. This is a numerical guard for this fixture, not sensor noise.

## Mesh evidence

| omega (rad/s) | 4 elements | 8 elements | 16 elements |
| ------------- | ---------- | ---------- | ----------- |
| 0             | 0.00887784 | 0.00221946 | 0.00055487  |
| 4             | 0.01044277 | 0.00261309 | 0.00065342  |
| 8             | 0.02294396 | 0.00579462 | 0.00145237  |
| 20            | 0.07646843 | 0.01899015 | 0.00473964  |
| 40            | 0.12441245 | 0.03281699 | 0.00831759  |

The predeclared fine-mesh error ratio was between 3 and 5 and maximum
sixteen-element component error below 2%. All cases pass without changing
those thresholds. The first six checks pass in 70.52 s on Windows, with each
mesh case taking at most 10.97 s under the existing 60-second deadline.

The joint check separately compares 8/16-element full responses to the
continuum and 2/4/8/all retained planar modes to their own full mesh. Modal
loads and observations use the same work-conjugate basis mapping as the axial
study, including translation length scaling and unscaled rotation. All modes
must reproduce the full driven planar transfer to roundoff. Eight modes must
have less than 3% sampled componentwise truncation error; this threshold is
independent of the mesh-error threshold. Coupled damping remains a full matrix.
The shared test transfer helper now accepts either vector or matrix ports;
the original four axial tests still pass. All twelve combined bending/axial
cases pass in 86.44 s. The corresponding bending/axial/API suite passes all
21 tests on Linux in 294.88 s (three unavailable-plugin configuration warnings),
using the preserved Python 3.11.15/NumPy 2.3.5/SciPy 1.15.3 environment and one
BLAS/OMP/MKL thread. Repository Ruff 0.14.10 passes (3,798 formatted files).
All nine final inventory/governance gates and normal publication hooks pass;
536ca60ebd4c5d2802245a7753e65f034a1f5706 is remotely verified. No production calculation
or public API changes require a new runtime-behavior implementation.

| Elements | Retained planar modes | Max modal error against full mesh | Max total error against continuum |
| -------- | --------------------- | --------------------------------- | --------------------------------- |
| 8        | 2                     | 0.6855250                         | 0.6915062                         |
| 8        | 4                     | 0.0086277                         | 0.0272544                         |
| 8        | 8                     | 0.0002574                         | 0.0326516                         |
| 8        | all 18                | 1.44e-13                          | 0.0328170                         |
| 16       | 2                     | 0.6808303                         | 0.6823449                         |
| 16       | 4                     | 0.0088915                         | 0.0103364                         |
| 16       | 8                     | 0.0004217                         | 0.0080156                         |
| 16       | all 34                | 4.87e-13                          | 0.0083176                         |

Total error need not decrease monotonically as modes are added at fixed mesh.
For eight elements, the four-mode maximum total error is 2.73%, versus 3.28%
for all modes. A close continuum match alone therefore does not bound the
truncation error; both comparisons must be retained.

## Limits and next work

The maximum frequency here is 40 rad/s, approximately 6.37 Hz. This is a
low-frequency numerical control, not acoustic-band qualification. Finite
samples do not bound intervening peaks/antiresonances. These synthetic material
and boundary values are not measured golf shaft, grip or head parameters.

Retain independent mesh and modal refinement, input/output-specific peak and
phase assessment, parameter validity and transfer conditioning. Loaded rotating
and moving-boundary models require their own comparisons; these stationary
results do not prove universal centrifugal stiffening, nonlinear stability,
impact launch effects or sweetness. Flexible head/contact, calibrated radiation,
physical/blinded evidence and exact-pin consumers remain required. All epics
remain open and the separate classifier #5103 is still required for delivery.
