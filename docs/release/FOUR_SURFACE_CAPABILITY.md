# Four-Surface Capability Contract

`four_surface_capability.v1.json` is the machine-readable declared-scope inventory for
issue #4264. It classifies the standalone Tools PyQt6 and React applications
and the UpstreamDrift PyQt6 and React consumers with stable surface IDs.

The governed scope contains every structured campaign program, every unique
active Markdown specification authority at `SPEC.md`, `docs/specs/**/*.md`, or
`docs/rate_of_closure/*.md` linked by those programs, and six curated
evidence-backed capabilities. A capability is supported only
when it cites source and test evidence at the exact Tools commit pin. An
unsupported or deprecated state must carry a visible reason. UpstreamDrift
consumer support additionally requires an installed consumer commit pin; a
launcher entry, copied implementation, or unrelated route is insufficient.

## Validate

From the Tools repository root:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
python scripts/four_surface_capability.py
python scripts/four_surface_capability.py --normalize
python scripts/four_surface_capability.py --schema
python scripts/four_surface_capability.py --declared-scope
python -m pytest tests/rate_of_closure/test_four_surface_capability.py -q
```

The default command validates the strict typed contract, checked-in schema
digest, repository evidence paths, and freshness window. `--normalize` emits
canonical JSON; `--schema` deterministically generates the consumer schema;
and `--declared-scope` emits the program/spec declarations derived from the
campaign authority.

## Maintenance contract

When capability evidence changes:

1. refresh the exact Tools commit pin and every Tools evidence commit;
2. classify every declared capability on all four stable surface IDs;
3. add evidence before claiming support and a reason before declaring an
   unsupported or deprecated state;
4. refresh `observed_on`, `max_age_days`, and derived `expires_on`;
5. regenerate the schema and update its SHA-256 digest if the typed contract
   changed;
6. update the #4260 record in `rate_of_closure_campaign.v1.json` without
   promoting its delivery or release stage unless protected evidence supports
   that promotion; and
7. update the root, Rate-specific, and campaign handoffs plus `SPEC.md` in the
   same implementation commit.

Expiry is deliberate: stale inventory must fail CI until its source and
consumer observations are refreshed. Narrative headings and feature bullets
remain outside deterministic coverage until promoted into a structured
campaign program or linked active specification; installed consumer parity and
conformance evidence remain required before issue #4264 can close.
