# Neural Model Lab v2

The Rate of Closure web and PyQt applications expose the same separate Neural Model Lab workspace. It is a safe client of the private launch-monitor campaign, not a browser or in-process trainer.

## Authority and current availability

Vendor availability is parsed from `launch-monitor-capability-manifest/v1`. A bundled aggregate-only snapshot supports offline inspection; users may load an authorized private manifest from disk without persisting its path or any corpus rows. The canonical private evidence is merged commit `78f0a42540e523ac883d843394b30a636311bf9d`, `results/v2/capability_manifest.json`, file SHA-256 `1906d19fcace3284ae99d9dd8de213a0e2dabe8c062e4d63edc06c98f7eaf92e`.

Current training eligibility fails closed:

- TrackMan: 11,699 rows, 9,298 strict model-input rows, no approved repeating split group; the historical artifact is `retired_non_group_safe`.
- Foresight: 4 rows, 2 strict model-input rows.
- FlightScope: 2,794 rows, 0 strict model-input rows.

The UI never invents vendor availability. It reads `allowed_operations.vendor_training`, approved groups, artifact status, and quantified training blockers from the manifest.

## Group-safe training request

A custom request names an immutable dataset repository, 40-character commit, path, SHA-256, and row count, plus explicit vendor, features, targets, and split group. `shot_id`, `source_row_number`, and `row_index` are forbidden split groups. The selected group must be explicitly policy-approved, contain at least three distinct groups, and repeat at least one group. Features and targets must be present, non-empty, and disjoint.

The request contains no rows. React submits it to an explicitly configured private API or exports it for the private CLI. PyQt invokes an explicitly configured private CLI and shows process status/output. Neither UI trains in its own process.

## Portable inference

Only `launch-monitor-neural-bundle/v2` JSON is accepted. The loader validates dataset and training-manifest hashes, bounded dimensions, supported `linear`/`relu`/`tanh` activations, finite weights, feature units and applicability ranges, model card, held-out metrics, and explicit residual availability. Executable pickle/joblib/framework checkpoints are never deserialized. Queries produce unit-labelled predictions and out-of-domain warnings.

Residual plots render only when the artifact provides aligned held-out residual rows. Otherwise both UIs show a typed unavailable reason.

All vendor-comparable models are descriptive surrogates. They are not vendor-device emulation or certification.

## UpstreamDrift v2 seam

Python and TypeScript clients target `POST /tools/launch-monitor-analytics/v2/analyze` and validate contract `2.0.0`, evidence lineage, backing-record shape, and safe claims. New player/performance workspaces label embedded calculations as local v1 compatibility/offline fallbacks when no v2 authority is configured. The current canonical v2 response does not expose row-aligned residuals, so those plots fail closed rather than guessing alignment.

The expected-strokes column mode is named **user-supplied expected-strokes SG**. A URL is only a user citation; Tools does not claim to reproduce or validate that baseline. Source-backed strokes gained remains unavailable until a versioned baseline manifest/table supplies source URL, version, SHA-256, lie/distance state schema, deterministic lookup/interpolation, and the dataset supplies the required course-state inputs. The separate radial target-error proxy remains explicitly not strokes gained.
