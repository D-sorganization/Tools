# Provider UI Delivery Investigation

Renderer prerequisite [PR #5090](https://github.com/D-sorganization/Tools/pull/5090)
is open at `b8c6e6013ad9dab6eab0bbfe0096b449feb16abb`. Its fresh production
workflow passes; remaining protected checks and merge are pending. Historical
investigation follows, with the superseding capture evidence at the end.

## Evidence Inspected

Tools T2 head `2f975d06ee9e4192a4b35ba5172bb7b3b8950e8a`,
[run 34180114955, job 101931507111](https://github.com/D-sorganization/Tools/actions/runs/34180114955/job/101931507111):
all 23 rendered interaction/accessibility tests pass. The subsequent comparison
fails for **nine** PyQt views: clubhead, plots, calculation_description,
flight_explorer, launch_monitor_analytics, neural_model_lab, variation, putting
and glossary. The earlier turnover count of seven was incomplete.

[Artifact 10040345545](https://github.com/D-sorganization/Tools/actions/runs/34180114955/artifacts/10040345545)
was downloaded to a temporary analysis directory. Its name is
`rate-web-playwright-pr-34180114955-2`, reported archive size 38,823,824 bytes.
The committed and candidate clubhead PNGs were visually inspected. The main
plot/content remain present; differences appear around widget labels and borders.
An RGB-delta >1 region audit found exactly 705 changed left-panel pixels in all
ten views. A complete all-view visual review remains open; no reference image
was replaced.

The candidate manifest declares
`posix-offscreen-qt-6.11.0-pyqt-6.11.0-matplotlib-3.11.1-font-dejavu-sans-dpi-1.0-1440x900`.
The same job's pytest header explicitly reports PyQt6 6.11.0, **Qt runtime
6.11.2**, and Qt compiled 6.11.0. In the inspected source,
`tests/rate_of_closure/pyqt_visualization_tab_probe.py` records
`QT_VERSION_STR`; that identifies the compiled version. The environment checker
in `scripts/check_rate_pyqt_environment.py` verifies distribution versions for
NumPy, SciPy, PyQt6 and Matplotlib, but does not check the separately supplied
Qt runtime. The font-stack check passes in this job.

The latest reference PNGs trace to trusted run 34045862045, job 101522812023
on OGLaptop, **also Qt runtime 6.11.2**. Its FreeType/Fontconfig packages are
2.14.2+dfsg-1ubuntu0.1 / 2.17.1-3ubuntu1; the candidate uses
2.13.2+dfsg-1ubuntu0.1 / 2.15.0-1.1ubuntu2. The old checker accepted both stacks
under one screenshot identity. These observations support a common rendering
difference, not proof that every changed pixel is attributable to fonts.

## Current Remediation and Next Steps

Existing issue #4844 was claimed and updated with the actual runtime/font
provenance. PR #5090 uses one digest-pinned Ubuntu 24.04 container and common
font installation for both paths while retaining the PR-hosted/trusted-fleet
runner boundary. It checks the exact font stack and the Qt runtime/SIP pins.
TDD and SPEC checks pass 49 tests; all nine manual checks and normal commit/push
hooks pass. No image, tolerance or runner-policy guard was weakened.

Continue in `C:/Users/diete/Repositories/Tools-impact-render`, branch
`fix/4844-consistent-pyqt-renderer`, with its
`docs/development/rate-pyqt-renderer-4844.md`. Linux capture, every-candidate
inspection and a protected reference update remain required. The local shell
has no Docker executable, so no local container render is claimed.

The T2 UpstreamDrift consumer job 101951943569 passes after upstream PR #9745
merged. T1's separate UD retry job 101955835753 also passes. The carried-forward Gasification
failure remains a private-repository checkout error; secret-metadata access
returned 403, so credential availability is unknown. These are separate gates.

## Repeatable Capture and Reviewed Reference Update

Linux run 34213771459 attempts 1 and 2, source `df4101f2825b3b2d255dad1d6f8746818fc82812`,
each pass 73 browser and 23 PyQt capture/accessibility tests. The ten PyQt PNGs
are byte-identical; React repeat differences remain within unchanged thresholds.
All twenty initial desktop references were individually reviewed and copied
unedited from the first capture; both candidate sets pass the production
comparator against them. No tolerance or historical calibration was changed.

The [complete review ledger](https://github.com/D-sorganization/Tools/blob/b8c6e6013ad9dab6eab0bbfe0096b449feb16abb/docs/development/rate-pyqt-renderer-4844-reference-review.md)
records exact artifact/hash provenance and existing initial-state/clipping
limits. Agent review does not replace human or whole-product acceptance.
All 60 local contracts, nine manual gates and normal hooks pass. Fresh
run 34217794994/job 102033568943 now passes the production workflow, including
the new references. PR #5087 is closed, unmerged. Protected delivery is pending.

Current T2 Gasification checkout still fails with 404 at job 102009926186 on
head `476eaa98b`. The workflow references RUNNER_CHECK_TOKEN with github.token
fallback, but masked logs do not identify which credential was selected. The
browser is signed out and secret metadata is inaccessible. Existing access
configuration was requested without secret values. Do not copy a local login
token into CI, weaken checkout failure handling or claim consumer test success.

The renderer head's fresh Gasification job 102039836106 again fails private
checkout before tests. Its Rust quality job 102033732755 (run 34217794944) fails
`watcher::tests::debounces_rapid_changes`: four callbacks exceed a ceiling of
three, while the five watcher tests take 21.01 s. The test assumes ten writes
separated by sleep(5ms) fit a 500ms quiet window; scheduling delays can legitimately
split them. Scoped open-issue search found no matching owner. Issue #5095 now
owns deterministic production debounce-state verification, retaining filesystem
integration coverage and forbidding increased ceilings/sleeps or ignored tests.
Claim was free; codex lease `impact-acoustics-01a07d8a-debounce` expires at
2026-09-08T13:44:58Z. No Rust repair has been implemented yet. Renderer #4844's
lease is renewed through 2026-09-08T13:42:43Z. These remain distinct CI gates.

## Deterministic Debounce and Reviewed-Identity Compatibility

Rust repair #5095 is published as PR #5097 at `c0f277406c526c4172e3cb15e8fdd6da649ada87`.
Eight exact-time tests plus four filesystem tests pass with default and Python
features; Clippy, formatting, all manual gates and applicable normal hooks pass.
The PR initially conflicted with newly merged T5 ingestion #5084; its main merge
preserves that module and resolves the adjacent handoff/digest. Remote Rust CI
must still run on the combined head.

Fresh renderer UD job 102039836077 passed thirteen contracts and failed the old
variation hash assertion. UpstreamDrift #9783 reproduces that RED, accepts only
the two reviewed source/hash pairs and rejects swaps/unknowns. All eighteen
shared contracts pass against both the exact pin and renderer candidate; pinned
Ruff 0.15.17, governance and commit hooks pass. PR #9784 publishes head `aa08217c5` with every normal commit/push hook passing. Pixel tolerances and vendor pin are unchanged. Private checkout access
remains independently unresolved; the credential-configuration question is pending.

At current PR #9784 head `aa08217c520c68867945b999a8c77425e6b3ae2b`, the actual
shared-tools-consumer-contracts job 102066036416 passes, as do normal Python,
Rust and security lanes. Three companion gates fail in run 34227729340:
rolling native semantics lacks execution provenance; optional-stack manufactured
source keys disagree; the authority lane cannot load liburdfdom_sensor.so.4.0.
Main is still `98c68af2f`. Diagnose those concrete failures without weakening
research provenance or making an unsupported claim of a green head.

Rust PR #5097's main merge is committed at `b268228bd` but its push was refused:
Prettier reformatted the incoming swing_sim API JSON and mypy rejected an
unused ignore in the incoming ingestion test helper. A claimed #5074 fixture
annotation correction passes the actual mypy hook and all eighteen ingestion
tests. The JSON formatting preserves parsed content exactly. Corrected head
`e0369fdda2a2fdb5570e177c5fc35901fd691ccc` is published with every normal push
hook passing. A newer main change again makes the PR conflict; protected merge
and current-head Linux CI remain outstanding.

UpstreamDrift #9787 now owns the manufactured-authority prerequisite. In an
isolated Linux CPython 3.11.15 environment, the original hash lock installs and
passes pip check but actual native import reproduces the missing URDF ABI 4.
Pinocchio 3.8.0 links against liburdfdom 4; the lock installed version 6. Using
cmeel-urdfdom 4.0.1 exposes a second missing TinyXML2 ABI 10. With
cmeel-tinyxml2 10.0.0, actual Pinocchio/MuJoCo/NumPy/SciPy imports and pip check
pass while all four core scientific versions remain pinned. The lock has been
regenerated with uv 0.10.11 and hash-enforced reinstallation passes. Actual
two-process evidence generation, source consistency and claim registration
remain pending; no old result has been relabeled with fabricated provenance.

## Current Delivery Reconciliation

Renderer compatibility UpstreamDrift #9784 merged at `9aa26e4f8`.
Manufactured-authority #9787 / PR #9804 publishes `d1563dffa`: actual compatible
URDF/TinyXML wheels retain the pinned scientific engines; two native processes
produce identical current evidence (SHA-256 355157bd57d0c2eb19652d3bd109c8074cc6ba83dc949abda478551a470d390c).
All 116 strict authority tests, three tests in a distinct rolling environment,
canonical claim/numeric checks and all 253 existing PDF pages pass. Normal
push hooks pass, including the approved-CLI hook alignment tracked in
Repository_Management #1572. No measured/acoustic or archival approval follows.
Open #9726's additional governed bootstrap source will require regeneration if
merged. Current-head protected CI remains outstanding.

Tools Rust #5097 publishes `76b982b73` with T5/T6 main integrated and every
normal local hook passing. Run 34245271235 failed only its Python 3.12 inventory
worker and dependent aggregate. The first retry request was rejected because
the workflow was still running; after completion the normal job-rerun request
succeeded. Attempt 2 passes. A subsequent PR read confirms #5097 had already merged
at `d9dec3602605f5c03eb7dff507a02ef7c2995b1b` (15:51:55 UTC). Separate
file-watcher Python 3.10 and private-consumer checks still fail; no whole-head
green claim or worker-crash cause is inferred.
T6 #5083 merged; issue #5075 records four confirmed schema/evidence gaps.
The loaded-operator checkpoint's separate Windows timeout and Linux 567-test
qualification are recorded in `LOADED_DYNAMIC_OPERATORS.md`.
