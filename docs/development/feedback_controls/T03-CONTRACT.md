# T03: Bounded Experiment Resources and Preview Interchange

Tools exposes generic resource and preview contracts through
`sidekick.lab.mocap`; it does not schedule fleet jobs, start a simulation, or
qualify a scientific result.

`ExperimentResourceBudget` caps workers, memory, disk and completed preview
count/bytes. Call `validate_request` with the requested estimates and current
free disk before dispatch. `ExperimentCancellationToken` is cooperative and
must be checked at safe worker boundaries; it does not interrupt native code.
`ExperimentTiming` records elapsed seconds against the deterministic
`make_experiment_cache_key` result. The key includes model/provider identity
and the T01 integrity digests for capabilities, state, input mapping, time grid,
applied input and executed policy. It deliberately excludes the opaque run ID so
identical replay payloads can reuse a cached result. Bump
`EXPERIMENT_CACHE_KEY_VERSION` when key semantics change.

Preview manifests use `preview-manifest/1.0.0` and
`schemas/mocap/preview-manifest-v1.schema.json`. A manifest requires at least
one completed video reference and declares only opaque IDs, SHA-256 digests,
byte sizes and approved video media types. It contains no local path or source
metadata. `build_preview_root` resolves `MOTION_MATCHING_PREVIEW_ROOT`, or uses
the caller-supplied desktop directory plus `Motion_Matching_Previews`; it never
creates an empty directory or implies a result exists. `verify_preview_artifacts`
resolves references under the selected root and checks containment, media type,
size and digest. Release of a preview sourced from private data requires an
opaque authorization bound to the exact manifest digest.

`RemoteResourceReceipt` is optional preflight evidence, not a Tailscale client.
It checks expected host, provider and license digests, configured preview-root
identity, access confirmation and worker/memory/disk capacity before dispatch.
Returned manifest identity is checked separately, and its artifact files must
still pass `verify_preview_artifacts`. `cleanup.py` exposes a non-destructive
`CleanupDecision`:
eligibility requires task ownership, verified merge, a clean worktree, retained
receipts and no user data, raw capture or shared directory. It authorizes no
direct deletion.

The tests use synthetic bytes and identities only in
`tests/shared/python/sidekick/lab/mocap/test_experiment_resources.py` and
`test_preview_artifacts.py`. No private capture was read, copied or published,
and this contract emits no preview video itself.
