# Consistent Section Inertia: Private Verification

Tools #5072 remains partial. Static root finding is published at
`7d586c49467a1a5cf14a2ff860461022c6ad4128`, with every normal hook passing.
This checkpoint supplies section kinetic quadrature using the same SE(3)
interpolation as the elastic element. It does not yet assemble rotating loaded
equilibrium, a stable operating point, boundary work or a transient/FRF model.

## Source and Reuse

[Sonneville, Cardona and Brüls (2014), author manuscript](https://orbi.uliege.be/bitstream/2268/159471/1/pa_SonnevilleCardonaBruls2014_GeometricallyExactBeamFEOnSE3.pdf),
equations 69–71 and 78–82, supplies the section velocity-map and material inertia
formulation. Manuscript pages 13, 14, 21 and 28 were rendered and visually read.
The paper's material velocities must not be confused with finite rotation-vector
rates. Its mass matrix depends on relative configuration, and its inertial bias
includes the changing interpolation map as well as material-frame transport.

Implementation reuses `ComponentMassProperties` and `tip_spatial_inertia` for
mass, COM offset and full COM inertia. It reuses strict finite-array, fraction,
pose, logarithm and matrix-exponential/Frechet primitives. The relative-log
derivative moves from the elastic module into the common kinematics module;
both elastic and kinetic paths use it. No duplicate spatial inertia, principal
rotation convention, constitutive law or inferred section density is introduced.

## Coordinates and Derivation

Let `d=Log(H_left^-1 H_right)` and `alpha=s/L`, with linear-first local twists.
The section pose is `H(s)=H_left Exp(alpha*d)`. Define

```text
P(d) = [-Jr(-d)^-1, Jr(d)^-1]
B(alpha,d) = alpha Jr(alpha*d) Jr(d)^-1
Q(alpha,d) = [I-B, B]
d_dot = P(d) v_nodes
v_section = Q v_nodes
```

The same Q maps virtual nodal motion into virtual section motion. For a relative
direction delta, with A=Jr(d)^-1, differentiation gives

```text
DB[delta] = alpha (DJr(alpha*d)[alpha*delta] A
                  - Jr(alpha*d) A DJr(d)[delta] A)
DQ[delta] = [-DB[delta], DB[delta]]
Q_dot = DQ[d_dot]
```

Production uses matrix-exponential Frechet derivatives; it has no finite
difference, small-angle dead zone, imposed symmetry or eigenvalue clipping.

Each `InertiaSample` supplies a fraction and an existing physical body record.
Its mass and COM inertia already include the quadrature length/weight. The COM
offset is from the interpolated section origin in material axes. The sample
must not also include the centerline spread of a finite segment in its COM
inertia: the sample locations represent that spatial distribution. All samples
share one material-axis convention. That convention is distinct from the common
observer frame used for the two poses. Input provenance remains in the samples.

For sample spatial inertia D, point motion w=Q\*v, and nodal material acceleration a,

```text
M = sum(Q^T D Q)
M_dot = sum(Q_dot^T D Q + Q^T D Q_dot)
bias = sum(Q^T (D Q_dot v - ad(w)^T D w))
inertial_wrench = M a + bias
T = v^T M v / 2
```

Because `ad(w) w=0`, the transport contribution does no section quadratic work,
and `v^T bias = v^T M_dot v/2`. Bias is therefore not generally zero-work at the
nodes: it accounts for changing kinetic energy when the mass map changes. It
must not be called material damping. Adding arbitrary damping to compensate for
an omitted Q_dot term would violate this energy identity.

## Numerical and Physical Boundaries

Fractions lie in [0,1]; poses must be proper rigid transforms in a common frame;
relative rotations remain inside the explicit principal-logarithm margin.
Velocities are physical material twists in m/s and rad/s. Samples and input
arrays are copied; output matrices are fresh. Nonfinite, malformed or coerced
motion inputs are refused. Existing body contracts govern physical inertia.

The sample set is explicit. Repeated locations can represent distinct pieces;
a single lumped sample remains rank deficient. The kernel neither repairs its
rank nor certifies distributed accuracy. A curved synthetic configuration checks
2/4/8-point Gauss convergence against 20 points; this does not establish a
universal quadrature order, mesh convergence or measured bandwidth. Integration
over a physical beam must use reference-arclength weights and density consistently.

This kernel has no elastic strain-domain or physical stability certificate.
Those belong to the complete loaded model. It must not be joined unchanged to
the older straight-state rotating coefficients and labelled a loaded model.
Complete rotating residual/derivatives, passive boundary coupling, stability and
mesh/time/modal/FRF qualification remain required. No acoustic prediction follows.

## Verification Evidence

TDD first failed on the missing module. Initial 12 tests pass; expanded 19 tests
cover the velocity map, all six relative derivative directions, the zero-angle
limit, input contracts, frame agreement, copy isolation and rank preservation.
General 4x4 matrix logarithms/exponentials independently differentiate section
poses for every nodal velocity basis. COM translation and spin velocities give
an independent kinetic-energy matrix. Differenced COM acceleration and angular
momentum give an independent inertial-force oracle. Mass-rate, energy-rate and
common-observer changes also agree without production projection or clipping.

All 136 combined inertia/equilibrium/chain/load/section/SE(3) tests pass (23.71 s).
The full golf/API suite passes 543 tests with two optional build123d CAD skips
(75.67 s). Ruff 0.14.10 and the actual three-module pre-push mypy check pass.
The additive API entry is private with empty exports. Numerical fixtures are
synthetic and do not identify golf-club parameters or player-dependent sound.

Static-checkpoint manual gates were green before these numerical changes.
Inventory and handoffs are refreshed, and all nine manual gates pass again.
The private module raises the golf inventory count from 59 to 60. As with the
previous checkpoint, the provisional name-based index adds a Rust math-primitives
test association; this is not a Rust implementation or verification change.
Normal commit/push checks remain required before publishing this checkpoint.
