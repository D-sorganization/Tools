# Review Rules: Tools

Repo-specific rules for anyone reviewing a Tools pull request. The author
checks them first with the `fleet-review` skill. A breach is a finding only
when it's cited here or in `AGENTS.md` / `CLAUDE.md`, by file and section.

## Scope and Size

- Under 500 changed lines after formatters, targeting `main`, never stacked.
- No new file over 400 lines; over 800 is a critical violation (AGENTS.md,
  5d).
- Tools is the shared library UpstreamDrift and Gasification_Model import.
  Logic duplicated downstream belongs here (CLAUDE.md, Coding Standards).

## Public API

- Renaming, removing or changing the signature, return type or exception
  behaviour of a public function is a breaking change. It needs a deprecation
  path and coordinated downstream PRs (CLAUDE.md, Known Constraints).
- Every module under `src/shared/python/sidekick/` defines `__all__`, and
  private names start with `_`.
- A planned break regenerates its baseline in the same PR
  (`tests/sidekick_api_baseline.json` or `tests/api_baselines/`), carries the
  `!` marker in the PR title and links the downstream migration issues
  (CLAUDE.md, Public-API Contract Policy).
- Public functions validate inputs: `TypeError` for wrong types, `ValueError`
  for out-of-range values.

## Evidence

- New behaviour needs a test that fails on `main` (TDD, AGENTS.md section 4).
  Every new public function has tests, and API surface downstream repos use
  is guarded by `contract` tests.
- Each test carries its marker from `pyproject.toml`.
- Coverage holds the `fail_under` floor and the per-package ratchets in
  `config/coverage_policy.json`. Touched sidekick modules keep at least 50 %
  per-file coverage.
- A new module adds its manifest entry.
- Loosening a tolerance, budget or coverage floor needs a
  `Tolerance-Change-Evidence: #N — <numbers>` commit trailer.

## Code

- No `print()` in `src/`; use logging. No wildcard imports or bare `except:`.
- No method chains deeper than two levels, and no imports across package
  boundaries.
- No TODO or FIXME without a tracked issue.

## Handoffs and Shared Files

- A PR touching `src/**` gets one SPEC.md change-log row keyed by its PR
  number (from its change fragment under `changes/`), or carries the
  `spec-exempt` label. A serial spec version, a reused
  key or a bumped `Spec Version` header is a finding.
- Actively developed tools keep a current-state `AGENT_HANDOFF.md` of 150
  lines or fewer, from `docs/AGENT_HANDOFF_TEMPLATE.md`.
- A new root entry is added to `scripts/check_root_allowlist.py`.

## Merging

- PRs open ready, never as drafts.
- A PR that deletes files says `deletions-acknowledged: yes` on its own line
  in the body, or carries the `deletions-acknowledged` label. Otherwise the
  arm guard refuses it.
- PRs are armed only with `scripts/automerge_guard.py`, never with a
  merge-method flag, `--admin` or a ruleset change.
- Commits use a GitHub noreply email. No secrets, tokens, personal data or
  binaries over 50 MB.
- `Closes #N` sits on its own line. An issue closes only through a merged PR
  or an exempt label.
