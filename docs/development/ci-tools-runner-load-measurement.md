# Tools runner load measurement (TOOLS-CI-011 / #5695)

Epic: [Repository_Management#2215](https://github.com/D-sorganization/Repository_Management/issues/2215)

## Methodology

RM-CI-000: `scripts/measure_workflow_runner_share.py` queries
`/repos/D-sorganization/Tools/actions/runs` with one UTC calendar day per request
(`created=YYYY-MM-DD..YYYY-MM-DD`) and paginates at `per_page=100`, reusing
`runner_capacity_check._github_get` for authenticated GETs. This avoids the
GitHub `total_count` ceiling of 1000 on undivided windows.

Regenerate the snapshot:

```bash
python3 scripts/measure_workflow_runner_share.py \
  --token "$(gh auth token)" \
  --window-days 14 \
  --end-date 2026-10-20 \
  --output-json config/ci_tools_runner_load_baseline.json \
  --print-markdown
```

Fixture-backed aggregation tests:
`tests/ops/test_measure_workflow_runner_share.py` and
`tests/fixtures/measure_workflow_runner_share/sample_day_slices.json`.

## Measured snapshot (14d UTC ending 2026-10-20)

Source: `config/ci_tools_runner_load_baseline.json` (generated 2026-10-21).

| Metric | Value |
| --- | ---: |
| Total workflow runs | 5259 |
| `pull_request` + `merge_group` | 3150 (59.9%) |
| `schedule` + `repository_dispatch` + `workflow_dispatch` | 698 (13.3%) |
| `schedule` alone | 691 (13.2%) |

### Counts by `event`

| Event | Runs |
| --- | ---: |
| `pull_request` | 2988 |
| `push` | 738 |
| `schedule` | 691 |
| `pull_request_target` | 649 |
| `merge_group` | 162 |
| `dynamic` | 24 |
| `workflow_dispatch` | 7 |

### Top 10 workflow names (all events)

| Workflow | Runs |
| --- | ---: |
| Merge Hold Guard | 1215 |
| CI Standard | 574 |
| Detect Secrets | 451 |
| Anti-Phantom-Merge Guard | 408 |
| Docs Governance | 401 |
| Jules Diff Verifier (Anti-Phantom-Merge Gate) | 378 |
| Bot CI Trigger | 328 |
| Spec Check | 310 |
| Topology Governance | 227 |
| File Size Budget | 182 |

### Cap note

`2026-10-12` returned `total_count>=1000` for its UTC day slice, so that day is
a **lower bound** in this snapshot. Per-day slicing explains draft baselines that
show identical `total_count=1000` across repos: the API caps before pagination
when the window is too wide.

## Recommendations (follow-up issues)

Each scheduled workflow below should be triaged in its own issue (no trigger
changes in #5695). Suggested direction from cron cadence and measured volume:

| Workflow file | Schedule | Follow-up |
| --- | --- | --- |
| `Merge-Hold-Guard.yml` | `20,50 * * * *` | [#5698](https://github.com/D-sorganization/Tools/issues/5698) |
| `conflict-disarm.yml` | `15 */4 * * *` | [#5699](https://github.com/D-sorganization/Tools/issues/5699) |
| `full-suite-nightly.yml` | `0 5 * * *` | [#5700](https://github.com/D-sorganization/Tools/issues/5700) |
| `heavy-integration-tests.yml` | `0 4 * * *` | [#5701](https://github.com/D-sorganization/Tools/issues/5701) |
| `benchmark-suite.yml` | `0 6 * * *` | [#5702](https://github.com/D-sorganization/Tools/issues/5702) |
| `nightly-full-repo-quality.yml` | `0 3 * * 0` | [#5703](https://github.com/D-sorganization/Tools/issues/5703) |
| `codeql-analysis.yml` | `17 6 * * 1` | [#5704](https://github.com/D-sorganization/Tools/issues/5704) |
| `Bot-CI-Trigger.yml` | `0 8 * * 0,4` | [#5705](https://github.com/D-sorganization/Tools/issues/5705) |
