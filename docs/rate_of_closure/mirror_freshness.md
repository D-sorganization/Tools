# Rate of Closure Explorer — Mirror Freshness Check

Issue #4624 (WS4 of UpstreamDrift EPIC #8965).

## The channel

```
Tools main: src/rate_of_closure/web          (canonical, parity-tested vs Python)
        |  manual scripts/sync-from-tools.ps1 run in the mirror repo
        v
D-sorganization/rate-of-closure-explorer     (public mirror; web/* mapped to repo root)
        |  Pages deploy workflow (re-runs parity tests — the deploy gate)
        v
https://d-sorganization.github.io/rate-of-closure-explorer/
```

The sync is manual, so canonical `web/` can move on `main` while the public
site silently decays. `scripts/check_mirror_freshness.py` detects and
surfaces that drift.

## Running the check

```
py -3.12 scripts/check_mirror_freshness.py          # human-readable summary
py -3.12 scripts/check_mirror_freshness.py --json   # machine-readable report
py -3.12 scripts/check_mirror_freshness.py --deep   # per-file blob-SHA diff
```

Requirements: a `git` checkout of Tools (for the canonical subtree) and an
authenticated `gh` CLI (for the mirror repo's commits/trees — no clone and
no extra Python dependencies).

Exit codes: `0` fresh, `1` drifted, `2` error.

Signals, in priority order:

1. **tree** (`--deep`): every canonical-tracked file under `web/` must exist
   in the mirror with an identical git blob SHA. Mirror-only scaffolding
   (LICENSE, README, `scripts/`, `.github/`) is ignored.
2. **recorded-sha**: if the mirror's latest sync commit message records a
   canonical Tools SHA (e.g. `Tools commit: <sha>`, per the release-process
   invariant that mirror content is a pure function of a recorded canonical
   commit), fresh iff it matches canonical `HEAD`.
3. **timestamp** (default): fresh iff the mirror's last commit date is at or
   after the last-change committer date of canonical `web/`.

The drift logic is a pure function (`assess_freshness`) tested with injected
fixtures in `tests/ops/test_mirror_freshness.py`; only the thin CLI wrapper
touches git and the network.

## Resyncing when drifted

Run `scripts/sync-from-tools.ps1 -ToolsPath <tools checkout>` in the mirror
repo, record the canonical Tools SHA in the sync commit message
(`Tools commit: <sha>` — this upgrades future checks to the exact
recorded-sha signal), and push. The mirror's Pages workflow re-runs the
parity tests before deploying; never bypass that gate.

## Follow-up (out of scope here)

Per workflow governance, no CI workflow is added ad hoc. The intended
completion of #4624 is a **governed** scheduled workflow that runs this
script post-merge/nightly and, on exit code 1, auto-opens a release issue in
`public-web-management` per its RELEASE_PROCESS (releases are tracked there;
the property bug tracker stays this repo). Optionally the automation can
also open the sync PR against `rate-of-closure-explorer` — keeping the
mirror's parity-test deploy gate intact.
