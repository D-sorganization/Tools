# Vendor-Pin Process

This document describes how shared Tools code is pinned in consumer repositories
(UpstreamDrift, Gasification_Model) and how those pins are kept up to date.

## Why vendor pinning exists

The Tools repository exports reusable Python modules (e.g. shared theme engine,
plot theme, shared contracts) that are consumed by downstream repos. Rather
than publishing a PyPI package for every change, the downstream repos vendor a
snapshot at a fixed git SHA. This gives consumers:

- A reproducible build that does not break when Tools main advances
- An explicit paper trail of when an upgrade happened and why

## Pin locations

| Consumer           | Pin location                                       |
| ------------------ | -------------------------------------------------- |
| UpstreamDrift      | `src/shared/python/` (git subtree / submodule SHA) |
| Gasification_Model | `vendor/ud-tools/` (git subtree / submodule SHA)   |

## Bump cadence

Bumps are triggered **per release tag** — every time a new git tag is pushed to
the Tools repository a bump PR is opened in each consumer repo automatically
(see Automation below).

There is no time-based schedule (weekly, monthly, etc.). Tags should be pushed
deliberately, after integration testing confirms the shared code is stable.

## Automation

`scripts/bump_vendor_pin.py` is the authoritative helper script. It:

1. Resolves the latest (or a given) Tools release tag.
2. Resolves the commit SHA for that tag.
3. Opens a bump PR in each consumer repo via the `gh` CLI.

### Local invocation

```bash
# Dry run — shows what would happen without opening any PRs:
python scripts/bump_vendor_pin.py --dry-run

# Target a specific tag:
python scripts/bump_vendor_pin.py --tag v1.2.3 --dry-run

# Open real PRs in all consumers:
python scripts/bump_vendor_pin.py --tag v1.2.3

# Target a single consumer only:
python scripts/bump_vendor_pin.py --tag v1.2.3 --consumers D-sorganization/UpstreamDrift
```

### CI/workflow integration

A GitHub Actions workflow that calls this script on every new tag push is
described in the acceptance criteria for issue #2948. The workflow file
**must not** be added without explicit user permission (per repository memory
rules). When that permission is granted, the workflow should:

1. Trigger on `push` events filtered to `tags: ['v*.*.*']`
2. Call `python scripts/bump_vendor_pin.py --tag $GITHUB_REF_NAME`
3. Report success/failure in the workflow run summary

## PR lifecycle in consumer repos

| State               | Meaning                      | Action                                           |
| ------------------- | ---------------------------- | ------------------------------------------------ |
| Open, CI green      | Bump is safe                 | Reviewer merges                                  |
| Open, CI red        | Smoke tests fail in consumer | Leave open; add a comment explaining the blocker |
| Closed (not merged) | Rejected intentionally       | Comment must explain why                         |

**Do NOT auto-close bump PRs when smoke tests fail.** The PR acts as a
visible blocker until the compatibility issue is resolved.

## Manual bump procedure

If automation is unavailable, bump manually:

```bash
# 1. In the Tools repo, find the SHA for the desired tag:
git rev-list -n 1 v1.2.3

# 2. In the consumer repo, update the pin file to that SHA.
#    For a git subtree the update looks like:
git subtree pull --prefix=vendor/ud-tools \
    https://github.com/D-sorganization/Tools.git v1.2.3 --squash

# 3. Open a PR with the title: "chore: bump Tools vendor pin to v1.2.3"
```

## Verification

After a bump PR merges, verify the consumer's smoke-test suite still passes
end-to-end. If new failures appear, file an issue against Tools or the
consumer repo and link it to the bump PR.

## Related references

- Issue #2948 (Tools) — original requirement
- Gasification_Model #3635 — phantom-close that motivated this process
- Audit doc: `docs/audits/2026-05-17_fleet_audit.md`
