# Rate of Closure Campaign Manifest

`rate_of_closure_campaign.v1.json` is the machine-readable authority for
implementation and release claims in the Rate of Closure, impact, and flight
campaign. It complements the issue tracker and handoff; it does not replace
either one.

## What It Records

The manifest normalizes metadata that was previously repeated in narrative
handoffs:

- one record for every primary campaign epic, program, or release gate;
- canonical repository specifications and GitHub issue authorities;
- shared pull-request carriers and exact observed branch heads;
- all four supported product surfaces;
- commit-bound local or hosted test evidence;
- explicit evidence gaps, limitations, dependencies, and child issue IDs;
- release status and the exact `main` SHA when a release is eventually proven.

Program delivery uses four mutually exclusive stages:

1. `specified_only` — acceptance scope exists, but no implementation carrier is
   claimed;
2. `implemented_on_feature_stack` — implementation exists outside a qualified
   protected release;
3. `protected_merged_to_parent` — a parent merge and passing protected checks
   are both recorded;
4. `released_to_main` — the program release SHA matches a merged carrier into
   `main`.

A feature-branch merge is not automatically stage 3. A local test pass is not
a protected merge, and a protected parent merge is not automatically a
default-branch release.

## Validation

Run the deterministic validator from the repository root:

```powershell
python scripts/rate_campaign_manifest.py
```

Emit the versioned JSON Schema generated from the same Pydantic models:

```powershell
python scripts/rate_campaign_manifest.py --schema
```

The validator fails closed when it finds undeclared fields, missing programs,
duplicate IDs, malformed SHAs, unresolved carrier or evidence references,
missing repository paths, placeholders, or contradictory delivery/release
states.

## Maintenance Procedure

Update the manifest in the same commit as a material carrier, test-evidence, or
release-state change:

1. Refresh only the affected carrier head, state, and merge evidence.
2. Add commit-bound test evidence; do not replace failed or queued evidence
   with narrative optimism.
3. Advance a program stage only when its validator preconditions are true.
4. Record limitations and missing evidence even when implementation exists.
5. Run the focused manifest tests and validator.
6. Update the root and Rate campaign handoffs in the same commit.

The checked-in snapshot does not query GitHub at validation time. This keeps CI
deterministic and avoids API-rate coupling, but means a maintainer must refresh
carrier observations deliberately after a merge, push, or protected run.
