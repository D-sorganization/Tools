# Consistent PyQt renderer — Tools #4844

## Current delivery state

- Branch: `fix/4844-consistent-pyqt-renderer`; commit: `SELF`; PR: [#5090](https://github.com/D-sorganization/Tools/pull/5090), open.
- Worktree: `C:/Users/diete/Repositories/Tools-impact-render`.
- Initial base: `20fff16ac`; current-main merge includes `669b478e1` (2026-09-08).
  The only conflict was the generated root handoff digest/count; both root
  handoff updates and the independent main impact-energy audit (#5079) are preserved.
- Codex lease: `impact-acoustics-01a07d8a-render`, expires 2026-09-08 08:21 UTC.
- This prerequisite affects impact-program PRs #5077/#5082; it does not complete
  the impact physics or empirical program (#5068).

## Observed failure and causal limits

Latest reference PNG provenance is trusted run 34045862045, job 101522812023
(OGLaptop). Candidate run 34180114955, job 101931507111 is GitHub-hosted.
Both report Qt runtime 6.11.2 / compiled Qt 6.11.0 and PyQt 6.11.0;
a Qt downgrade does not follow from the evidence.
Reference FreeType/Fontconfig packages are 2.14.2+dfsg-1ubuntu0.1 /
2.17.1-3ubuntu1; candidate packages are 2.13.2+dfsg-1ubuntu0.1 /
2.15.0-1.1ubuntu2. Matplotlib FreeType is 2.14.3 in both.
The old checker accepted either version independently against one baseline.
All 23 rendered candidate tests passed; nine screenshot comparisons failed.
A pixel-region audit found exactly 705 changed left-panel pixels in each of
10 captures. This supports a common renderer difference; it does not prove
that every changed pixel has that cause. The clubhead pair was inspected;
all-view reference review is still required.

## Implementation and compatibility

Both PyQt paths use the official Ubuntu 24.04 container index digest
`sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517`,
resolved from Docker Registry on 2026-09-08. Its amd64 image digest is
`sha256:1e0a86e57d247923571b75e0aaf48a1449cf8c543d51fb3e07a4a7d7bfa79316`.
The PR remains GitHub-hosted; trusted capture remains on `d-sorg-fleet`.
No runner guard, permission, required check, or image tolerance is weakened.
Both install the same font set including Playwright's Ubuntu 24.04 font
packages, so browser dependency installation does not introduce a different
font selection into the PR capture. Package repositories may advance;
the exact font checker must fail on an unqualified update. The base-image
pin alone is not a complete immutable apt repository snapshot.

The environment checker now verifies PyQt6-Qt6 and PyQt6-sip alongside the
binding and scientific dependencies. Font authorities reject alternative
version lists, empty/non-object documents, and unknown identifiers.
The committed font authority includes Matplotlib FreeType and selects the
Ubuntu 24.04 stack. Root lock additions match existing renderer constraints.
No application calculation, public shared API, or reference image changes.

GitHub documents container jobs on Linux/Docker hosts and their default
`sh` shell: [job containers](https://docs.github.com/en/actions/how-tos/write-workflows/choose-where-workflows-run/run-jobs-in-a-container).
Font package inventory was checked against [Playwright native dependencies](https://github.com/microsoft/playwright/blob/main/packages/playwright-core/src/server/registry/nativeDeps.ts).

## Validation and next actions

- TDD RED: 10 failures reproduced old checker/host behavior; separate container
  contract failed without the common container. GREEN: 49 focused and SPEC tests passed.
- Command: `python -m pytest tests/scripts/test_check_rate_pyqt_environment.py tests/scripts/test_rate_pyqt_renderer_identity.py tests/ops/test_rate_web_playwright_workflow.py tests/architecture/test_spec_version_freshness.py -q -n0 --no-cov`.
- Docker is not installed in this Windows shell; no local Linux render claim.
- All nine manual governance checks pass after inventory regeneration. Only the
  index digest/count for the already-merged theme shard needed repair; the theme
  implementation and shard contents are unchanged.
- Repository Ruff 0.14.10 lint/format, changed-script mypy with
  `--follow-imports=silent`, runner-policy guard, and 1,610-file shard partition pass.
- Implementation `53f072a58bce42a9c11a639329da01b093e30557` passed all normal
  commit/push hooks and is published in ready PR #5090.
- Current-main merge validation: 68 tests pass, including all 19 impact-interval
  solver checks from preserved main PR #5079.
- Merge push exposed inherited `test_solver.py:282` mypy `no-any-return` in the
  deliberate 1.05 force perturbation. The failed push was stopped; an explicit
  float return now passes mypy and all 19 impact-interval tests (10.94 s).
  The perturbation and residual assertion retain their values.
- Next: execute the Linux
  workflow, inspect every new reference candidate and its exact provenance,
  then propose only justified reference changes through protected review.
- Preserve old references until fresh capture exists. A passing unit suite
  is not a successful Linux render or human acceptance of the visual product.

## Container shell checkout follow-up

Published merge head `d01f42eda248e4d1588003622982caca52431725`
contains protected main/T1 `f7254461399ac18e5667a0215afd90a9ebff9d22`.
Run 34205681486, job 101994493233 successfully starts the Linux container,
installs system dependencies, and checks out the repository. The first shell
Git diff then exits 129 before capture. Checkout's log shows its safe-directory
entry is written under a temporary HOME. The container runs as root over the
host-mounted checkout. An isolated Git test with a simulated different owner
reproduces exit 129; applying the actual new workflow step changes it to zero.
This supports ownership trust as the cause; the next real run must confirm it.

Both container jobs now register only `$GITHUB_WORKSPACE` in their ephemeral
Git configuration and verify the checkout with `rev-parse` before subsequent
work. Candidate output paths use that runtime mounted path through GITHUB_ENV;
the earlier context expression expanded to the unmapped host path in CI logs.
No wildcard trust, exemption, ignored Git error, image or tolerance change is
introduced. The same experiment verifies paths containing spaces and continued
rejection of an invalid base revision. The workflow regression failed before
the new step; 52 related contracts pass afterward. All normal commit and push
checks must still pass, followed by Linux capture and reference review.

Checkout's temporary configuration behavior is visible in its pinned
[primary implementation](https://github.com/actions/checkout/blob/3d3c42e5aac5ba805825da76410c181273ba90b1/src/git-auth-helper.ts#L76-L115).

The normal commit hook also detected one newly inferred test link in the
Rust AI-backend inventory shard from the workflow regression. The standard
inventory generator adds only that link and its shard digest; no backend
implementation changes. Re-run the complete hooks with this generated repair.

## Firefox Home Ownership Follow-up

Linux run 34209196902/job 102005880697 at published head
`a2d4fdf07ac09b11a9bb2fc91c3828b999c6c811` confirms the scoped workspace
initialization and evidence co-change gate pass. Browser installation succeeds;
72 web tests pass. Firefox alone cannot start because the root container process
inherits /github/home owned by host uid 1001. The log explicitly identifies
that ownership mismatch. PyQt capture remains skipped after this web failure.

The PR job now creates one private browser home under RUNNER_TEMP with mktemp.
Browser installation and execution receive that same home as their subprocess
HOME, while the mounted runner home is untouched. Scoped Git trust is also
initialized in the private home. Browser caches stay consistent between install
and execution. No browser project, sandbox policy, assertion or failure is
skipped. The workflow regression first failed without this setup; 53 related
contracts now pass. An isolated shell experiment executes the actual setup and
verifies directory/config creation. That experiment clears this Windows host's
inherited GIT_CONFIG_GLOBAL override from its test subprocess; the one incidental
safe.directory entry created before noticing the override was removed exactly.
No Linux Firefox launch success is claimed until the next real run.
