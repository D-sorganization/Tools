# Fleet Runner Capacity — Steady-State Documentation

## Overview

The Tools CI fleet uses self-hosted GitHub Actions runners.  
Runner-pool saturation is a P0 issue: when the queue depth spikes, multiple
unrelated repos timeout simultaneously, creating misleading failures that
obscure real bugs.

## Steady-State Configuration

| Parameter                    | Value         | Notes                            |
| ---------------------------- | ------------- | -------------------------------- |
| Self-hosted runner count     | 4+            | Minimum; scale up as fleet grows |
| Queue-depth alert threshold  | 10 jobs       | Trigger `ALERT` advisory         |
| Target queue-wait SLO        | 300 s (5 min) | Jobs should not wait > 5 min     |
| Average job runtime estimate | 120 s         | Used for capacity math           |

## Capacity Math (Little's Law approximation)

To drain `Q` queued jobs within `T` seconds, given jobs averaging `J` seconds:

```
needed_runners = ceil(Q × J / T)
```

Example: `Q=30, J=120s, T=300s` → `ceil(30 × 120 / 300) = ceil(12) = 12 runners`

## Tooling

`scripts/runner_capacity_check.py` — capacity planning and alert CLI.

```bash
# Check queue depth and get advisory
python scripts/runner_capacity_check.py \
    --token "$GITHUB_TOKEN" \
    --org D-sorganization \
    --current-runners 4

# JSON output for machine consumption
python scripts/runner_capacity_check.py \
    --token "$GITHUB_TOKEN" \
    --org D-sorganization \
    --current-runners 4 \
    --json
```

Output levels:

- `OK` — queue is empty
- `WARN: ...` — queue is non-zero but below alert threshold
- `ALERT: ...` — queue depth ≥ alert threshold; add runners immediately

## Alerting

Wire `scripts/runner_capacity_check.py` into a cron job or GitHub Actions
scheduled workflow (on a GitHub-hosted runner, not self-hosted, to avoid
circular dependency):

```yaml
# .github/workflows/runner-capacity-alert.yml  (do NOT add — example only)
on:
  schedule:
    - cron: "*/15 * * * *"
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: |
          python scripts/runner_capacity_check.py \
            --token "${{ secrets.GITHUB_TOKEN }}" \
            --current-runners ${{ vars.RUNNER_COUNT }} \
            --alert-threshold 10
```

## Overflow Strategy

When self-hosted runners are saturated, short-term relief options:

1. **GitHub-hosted runners** — add `runs-on: ubuntu-latest` to non-sensitive jobs
2. **Increase concurrency limit** — raise `concurrency.group` limits in workflow YAML
3. **Add runners** — provision additional self-hosted runners (contact @dieterolson)

## Historical Context

Root cause of 2026-05-17 fleet-wide CI timeout:

- Maxwell_Daemon, Tools, and Pinocchio_Models hit runner-pool saturation
  simultaneously during a busy push window
- All three showed "runner did not start within 10 minutes" errors
- Unrelated to code changes in those PRs (infrastructure issue)

See also: `docs/audits/2026-05-17_fleet_audit.md`
