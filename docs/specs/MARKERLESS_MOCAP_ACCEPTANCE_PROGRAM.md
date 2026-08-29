# Markerless Mocap Acceptance Program

Version: 1.0.0
Issues: #4706, #4708, #4710
Status: Required release gates; passing the current schema slice does not qualify a physical lab

## Contract levels

| Level | Required evidence |
|---|---|
| Unit | DbC on identities, capabilities, clocks, frames, coordinates, transforms, observations, methods, policies, and manifests |
| Property | Transform/unit invariance, deterministic serialization, perturbation/noise response, missing data, order independence where declared |
| Component | Synthetic/prerecorded driver, capture lifecycle, calibration solver, pose mapping, association, reconstruction, C3D reader/writer |
| Integration | Tools-to-Upstream schema, biomechanics adapters, C3D independent reader, AffineDrift sanitized fixture projection |
| System | Mixed-camera matrix, cancellation, disconnect/reconnect, dropped frames, storage limits, crash recovery, multi-surface accessibility |
| Physical | Shop calibration target/phantom, synchronized reference event, coverage, lighting, repeatability, task-specific reference comparison |
| Release | Exact commit, clean export, license/SBOM, protected checks, merge SHA, artifacts, post-merge smoke and public verification |

## Phase gates

### A. Authority and schema

- One canonical schema owner and explicit adapter plan.
- `affinedrift-world-v1`, SI units, `wxyz`, transform direction, clock types, availability, privacy, and license boundaries are versioned.
- Unknown fields and incompatible major versions fail closed.
- Golden JSON is deterministic and schema-valid.

### B. Acquisition

- Synthetic and prerecorded drivers pass the same public `FrameSource` contract as optional hardware plugins.
- Zero, one, many, and mixed-camera configurations produce typed support results.
- Identity survives permitted reconnect/reboot cases or reports that stable identity is unavailable.
- Backpressure, bounded queues, timeout, cancellation, disconnect, duplicate and dropped frames are fault-injected.

### C. Synchronization

- Device, trigger, host monotonic, and UTC presentation clocks remain distinct.
- A common optical/electrical event measures pairwise clock skew and drift.
- Requested versus effective timing and uncertainty are recorded.
- Unqualified host-time or rolling-shutter input cannot receive synchronized status.

### D. Calibration

- Intrinsic tests cover pinhole/fisheye declarations, distortion, image mode, focus state, residuals, covariance, degeneracy, and identity binding.
- A Known-geometry synthetic rig recovers parameters inside declared uncertainty under nominal and perturbed observations.
- Global arbitrary-layout solve records gauge, observations, solver/version, robust loss, residuals, covariance, conditioning, validity, and approval.
- Camera movement is detected and invalidates or segments calibration; no silent self-adjustment is permitted.

### E. Pose and reconstruction

- Backend skeleton/keypoint mappings are versioned and license/provenance tested.
- Single-camera depth remains model-conditioned; it is never reported as triangulated 3-D.
- N-view tests cover two through reference maximum cameras, occlusion, outliers, missing cameras, association ambiguity, cheirality, covariance, and minimum-view rules.
- Observed input remains immutable; filtering, biomechanical constraints, IK, and derivatives are separate derived stages.

### F. C3D

- Characterize and refactor the current reader without behavior loss.
- Writer covers points, units/rates, residuals/camera masks, events, analog channels, force platforms when available, and processing provenance.
- Write-read and read-write fixtures include gaps and unknown metadata.
- An independent C3D reader verifies every golden export.
- A structured loss report is reviewed before any lossy export.

### G. Operational and UI

- Local-first/no-cloud, consent, retention, no-store, access, export, and deletion policies are visible and tested.
- UpstreamDrift web/PyQt surfaces use common theme and unit contracts, persistent help and hover/focus hints, actionable disabled reasons, keyboard navigation, screen-reader semantics, and responsive visual inspection.
- Health and scientific qualification are separate statuses.

### H. Shop commissioning

- Serial-numbered hardware/lens/network/storage/lighting inventory and layout.
- Measured bandwidth, storage margin, temperature, exposure/blur, clock skew/drift, dropped frames, calibration residuals/covariance/observability, useful-view coverage, and repeatability.
- Mixed-camera, reduced-camera, disconnect, occlusion, clothing, speed, body and implement scenarios.
- Reference-system comparison and reference limitations.
- Qualified, degraded, prohibited, and Unavailable uses plus recalibration triggers.

### I. Release

- Focused tests, full repository gate, and exact-HEAD clean export pass.
- Downstream consumer compatibility passes against exact commits.
- SBOM, model weights, vendor components, AGPL isolation, licenses, privacy, security, packaging, installation, examples, and extension docs pass.
- Full non-draft PR has exact-head protected checks and required review; no bypass or force/admin merge.
- Merge SHA and post-merge artifacts/smoke tests are verified.
- Root and module handoffs are current.

## Status taxonomy

- **Pass**: all declared evidence and bounds met.
- **Degraded**: usable only under explicit narrower bounds and visible reason.
- **Blocked**: a required external approval, hardware, evidence, or dependency is absent.
- **Unavailable**: the system cannot produce the quantity without inventing authority.

No phase inherits a pass from another phase. A green software schema suite is not camera, algorithm, biomechanical, or physical qualification.
