# ADR-007: Markerless Mocap Authority, Interchange, and Licensing

Date: 2026-08-25
Status: Accepted for contract implementation
Issues: #4706, #4708, #4710

## Context

UpstreamDrift, Tools, and AffineDrift already contain markerless ingestion, biomechanics, C3D, geometry, and publication work, but no repository owns a complete live, arbitrary-camera lab contract. Duplicate observation models, ambiguous transform directions, multiple C3D readers, product-specific adapters, and incompatible model licenses would otherwise produce another competing pipeline.

The program must be camera agnostic: a provider advertises actual capabilities, and the caller negotiates supported, degraded, or unsupported results. The phrase does not imply that an unsynchronized rolling-shutter webcam has the same scientific authority as a hardware-triggered global-shutter camera.

## Decision

### Tools owns neutral authority

Tools owns the MIT, vendor-neutral device identity/capability, frame/capture, clock/drop, session, calibration, 2-D/3-D observation, skeleton, uncertainty, provider, and C3D interchange contracts. It owns reference algorithms and golden fixtures that do not require a vendor SDK, model weight, cloud service, or AGPL program.

### UpstreamDrift owns application authority

UpstreamDrift owns discovery/configuration orchestration, the lab setup and calibration wizard, live preview and health, storage consent/retention, reconstruction and biomechanics integration, project persistence, API/web/PyQt UX, themes, units, help, accessibility, and shop commissioning.

### AffineDrift owns publication authority

AffineDrift owns evidence review, camera-selection pedagogy, calculation explanations, limitations, sanitized validation fixtures, public visualization, and immutable publication. It does not own camera drivers, synchronization, calibration solvers, pose inference, triangulation, IK, or C3D serialization.

### Tools_Private boundary

Tools_Private has no required role in the open mocap runtime. Its code, manuals, source references, artifacts, and metadata remain private unless a human owner explicitly approves a sanitized projection. Public workflows do not clone or inspect it.

## Coordinates, units, and transform direction

The v1 candidate shared world frame is right-handed, SI, with x toward target, y up, and z right. Unit quaternions serialize in `wxyz` order. The exact convention is versioned as `affinedrift-world-v1` so a later migration cannot silently reinterpret data.

Every transform names its direction. `T_world_from_camera` maps a point expressed in a camera optical frame into the declared world frame. An unqualified `extrinsics` field is prohibited. Wall clock does not order frames; device, trigger, host-monotonic, and UTC-presentation clocks remain distinct with offset/drift/uncertainty evidence.

## Evidence and availability

Observed, derived, model-conditioned, provisional, and unavailable values remain distinct. In particular, single-camera or body-model depth is model-conditioned and may not be labelled triangulated 3-D. A derived triangulated point requires at least two unique, geometrically qualified camera observations; higher minimum-view policies may apply per task.

A session is not labelled synchronized because a user selected a trigger mode. The system records timing topology, sequence identity, measured clock skew/drift, uncertainty, drops, duplicates, and qualification tolerance.

## Calibration learning

Camera-position learning produces an immutable, versioned calibration from retained observations, solver/version, residuals, covariance, conditioning, validity, and approval. Camera motion invalidates or segments that calibration. Inference may not silently mutate historic camera poses to improve a result.

## C3D boundary

C3D is a terminal interchange format, not the internal processing database. Tools owns the shared C3D reader/writer/validator and explicit loss report. UpstreamDrift routes import/export through that gateway. Points, units, rates, residual/camera masks, events, analog/force data, coordinate metadata, and processing provenance are preserved when representable; any loss is reported or blocks export.

## Licensing

The MIT core may use reviewed permissive libraries and optional vendor plugins. Vendor SDK and GenTL producer licenses remain separate from core compatibility.

FreeMoCap identifies as AGPL-3.0 and SkellyCam as AGPL-3.0-or-later. The MIT distribution will not copy, vendor, import, dynamically link, bundle, or install either as a dependency. A separately installed and separately started process may implement a general independently authored `ExternalMocapService` file/CLI/HTTP/WebSocket protocol. A product-specific derived bridge belongs in a separate compatible repository/process. Distribution remains blocked pending architecture and legal review because process separation is not itself a legal conclusion.

Non-commercial model weights or restricted body-model assets are optional and unavailable in the commercial default. Session provenance records model/weight source, version, digest, license, and approval.

## Privacy and security

The system is local-first and no-cloud by default. Recording, biometric storage, retention, export, and deletion require explicit operator policy and consent. No-store operation reports its reproducibility limits. Paths, sizes, queues, timeouts, workers, and logs are bounded; logs avoid frame and personal data.

## Consequences

- Existing UpstreamDrift observation and C3D paths must migrate through explicit versioned adapters before new UX treats them as canonical.
- A camera or pose provider can be added without changing core records.
- Unsupported capabilities remain visible rather than guessed from product names.
- Strict schema changes require versioned migrations, golden fixtures, downstream coordination, manual updates, and handoff updates.
- AGPL and restricted-weight integrations remain possible as explicit optional systems without weakening the MIT core boundary.

## Validation

- AST/API stability and strict JSON Schema tests.
- DbC unit/property tests for identities, clocks, coordinates, observations, uncertainty, availability, and deterministic serialization.
- Known-geometry, mixed-camera, timing-fault, reconstruction, C3D, clean-export, protected-CI, and post-merge gates defined in the acceptance program.
- Coordinated consumer tests in UpstreamDrift #9069, Gasification_Model impact qualification in #4751, and sanitized compatibility publication in AffineDrift.
