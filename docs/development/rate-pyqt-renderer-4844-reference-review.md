# Pinned Renderer Reference Review: Tools #4844 / PR #5090

Reviewer: Codex. Review date: 2026-09-08 UTC. This is an agent review of
20 initial desktop reference images, not human approval or complete product
acceptance. It does not close the separate every-state/viewport, assistive-
technology, hardware-performance or scientific-validation gates.

## Immutable Capture Identity

Source: `df4101f2825b3b2d255dad1d6f8746818fc82812`.
Run [34213771459, attempt 1](https://github.com/D-sorganization/Tools/actions/runs/34213771459/attempts/1),
job `102020593108`, artifact `10051181659` (`rate-web-playwright-pr-34213771459-1`).
Both surface manifests bind this exact source. Every candidate SHA-256 was
checked before inspection. All images are 1440x900; PyQt references are 100% DPI.
The run separately passes 73 cross-browser tests and 23 PyQt capture/accessibility
checks including 100% and 150% DPI. Only baseline comparison fails (nine views).

The immutable Ubuntu 24.04 container is
`sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517`.
The successful runtime checker records Python 3.12.14, NumPy 2.3.5, SciPy 1.17.1,
PyQt6 6.11.0, PyQt6-Qt6 6.11.2, SIP 13.12.0 and Matplotlib 3.11.1.
Font checks record Matplotlib FreeType 2.14.3, system FreeType
2.13.2+dfsg-1ubuntu0.1 and Fontconfig 2.15.0-1.1ubuntu2. The legacy screenshot
identity's `qt-6.11.0` is the compile-time Qt string, not evidence of runtime
6.11.0; the distribution/runtime gate and job log supply the actual locked stack.
No Qt downgrade or tolerance change is proposed.

## Per-Image Review

Mean and changed-pixel values below are differences from the existing reference,
in millionths (channel delta normalized by 255; pixel change means any channel
exceeds 1). They are not the two-capture repeatability measurements. Existing
limits remain React 4000/50000 and PyQt 200/250, with the already-authorized
PyQt simulation exception 10000/10000.

| Surface | Tab                      | Mean / changed | Candidate SHA-256                                                  | Inspection                                                                              |
| ------- | ------------------------ | -------------- | ------------------------------------------------------------------ | --------------------------------------------------------------------------------------- |
| react   | explorer                 | 1270 / 17205   | `5d95afd547d704e1548061d132a6855c442d9a923c853d6c37ce4d382b061f1e` | Representative head, reference marker and units visible.                                |
| react   | calculation              | 1027 / 14771   | `eeaad66c186188496e544581305fd7f2d7760bfbbd565d2500656f306e8eea26` | Formula narrative and frame definitions visible.                                        |
| react   | simulation               | 1798 / 26838   | `3524145be1f39b196a6f22ab59f61519ed0473c8e17ccbc85a92dd0549f73573` | Delivery-inspection policy and completed manual scenario visible.                       |
| react   | plots                    | 2155 / 28966   | `3d53581802b5291cd6cafcc1a9dc9f276b6d858d9385d2066e4f07d16a516367` | Closure curve and units visible; crowded legend retained.                               |
| react   | flight                   | 1236 / 17037   | `7f616bf28fa30be05621c9aa45dd4659f7c389d261d6c38c670005163b0b75a6` | No-flight prompt and disabled playback visible.                                         |
| react   | launch-monitor-analytics | 3048 / 41439   | `b34fc003a7281105e88e5b26e20275bb685fe18c4502be92fad3e30e97b91f85` | Demonstration source, row count and association boundary visible.                       |
| react   | neural-model-lab         | 3478 / 45593   | `351401caf0dab802648e90d46d66b0372271052ffe2101b3546335d1b0696090` | Unavailable training and source-policy reasons visible.                                 |
| react   | variation                | 1880 / 24397   | `90ed3963ad123d9dc320e97ba98816a01f43147f7541451591ecb724ee4e9d16` | Ready state and unsupported Morris input refusal visible.                               |
| react   | putting                  | 0 / 22         | `ec8e468a8aaee2d081a7178fb92956f95c23601206984479d4aced109c91b1ae` | Displayed scenario, timing, trajectory and units visible.                               |
| react   | glossary                 | 1027 / 14716   | `0032e515db91eee19fe73c609c6491de654ec2125843f73ea56369365424d585` | Search, term list and selected definition visible.                                      |
| pyqt    | clubhead                 | 54 / 904       | `5ab1bd9f278e87e0b8b984add8c78f1abc39db7c1b4cbd0219e1ec6b63d83629` | Representative head, frame axes and controls visible.                                   |
| pyqt    | plots                    | 57 / 924       | `823ae0b7c9260853f6889a6d9977bafc852cd3e1792e4892345ea9c5cac59b49` | Existing computing state and blank plot retained; not a result claim.                   |
| pyqt    | calculation_description  | 124 / 2052     | `ec0aca97e903f2650f319a7ddc8e5b6481a5b868656341cabe85b3bb3ea18baa` | Frame narrative and formulas visible.                                                   |
| pyqt    | simulation               | 98 / 1467      | `111c753131650ade867e2e87014df90b4f7ff73fa5154ffe06ad0b3938402c90` | Manual delivery, impact header and 3-D scene visible.                                   |
| pyqt    | flight_explorer          | 92 / 1410      | `1bf239ea7c966eb89cd5477ce335c899ee049e479c4517ae8d0a3e75c62bffa0` | No accepted flight and disabled playback visible; existing side-pane clipping retained. |
| pyqt    | launch_monitor_analytics | 90 / 1488      | `06d11bafdd03aff8ffea078c7f7f86d1c1caf7a6e6c08334c1d685862af4e8e4` | Demonstration source and 120 retained rows visible; existing pane clipping retained.    |
| pyqt    | neural_model_lab         | 86 / 1473      | `0b06ce2be450dc1c0080bfcc632940841e37910c9abd87072aad120f99fea083` | Unavailable artifact reasons visible; existing bar-label clipping retained.             |
| pyqt    | variation                | 76 / 1106      | `22f6640e896e9ea5c740e9db7e3d3201cdf7264f2cf1cff33966b539354f1d40` | Ready state, zero progress and empty axes visible; existing pane clipping retained.     |
| pyqt    | putting                  | 103 / 1681     | `b4eb9c2e20f19e99f96f7420df14713a22aa33a1e5321a08998434ab28ee4f79` | Scenario and three plots visible; existing lower annotation crowding retained.          |
| pyqt    | glossary                 | 98 / 1676      | `aedf5325b2908cadd1f86ad52faa3ad9bb40312fb8a48a01849df92ebe2d88bd` | Search, term list and definition visible.                                               |

## Findings and Disposition

All 20 initial candidate views were individually inspected. Existing PyQt
plots, flight, neural-lab and variation references were also opened directly;
the blank/loading view and pane/label clipping already occur in those references.
These are retained findings for the existing visualization program, not fixes or
newly accepted user experience. A renderer refresh must not rename a loading,
unavailable or no-result state as a completed scientific result. The changed
font rasterization is consistent with the verified stack difference, but one
capture alone is insufficient evidence of repeatability.

A second job on the same immutable source was requested via the normal failed-job
rerun endpoint solely to measure repeatability. It is expected to reject the old
nine references again. The reviewed attempt-1 reference set is now proposed; no limit was changed.
The original calibration ledger remains historical evidence for the existing
tolerances; this review will record the new controlled repeatability separately.
Do not invent human sign-off or treat a protected CI capture as physical impact
or acoustic validation.

## Controlled Repeatability and Proposed References

Attempt 2 is job `102024600581`, artifact `10051637507`
(`rate-web-playwright-pr-34213771459-2`), on the same source and container.
Its runtime/font checks also pass, followed by 73 browser and 23 PyQt tests.
All ten PyQt PNG hashes are identical between attempts. Five React PNG hashes
are identical; the others differ by at most 22 changed-pixel millionths, with
mean channel differences rounding to zero millionths. That is far below the
unchanged React 4000/50000 envelope. Both surface manifests and every image
hash were checked before comparison. This establishes observed repeatability
for these captures; it does not guarantee all future hardware or software.

The proposal copies all 20 individually reviewed attempt-1 PNGs without editing
their pixels, and binds the baseline manifest to their exact source/digests.
Updating the complete set keeps the manifest's single source-artifact commit
truthful for both surfaces. The existing calibration history and every tolerance,
including the earlier PyQt simulation exception, remain byte-for-byte unchanged.
The packaged-provenance test first failed on the old source commit; all 60
baseline/environment/workflow/SPEC tests pass with the proposed set. Both actual
candidate sets pass the unmodified production comparison CLI. Direct invocation
needed the checkout's src on PYTHONPATH; the initial missing-package failure was
an uninstalled local CLI path, not a pixel or runtime-environment failure.

Related PR #5087 is closed without merging. Normal hook checks and a fresh PR
run still need to pass; protected merge is the baseline approval event. This
agent review does not supply human product acceptance or resolve the pre-existing
clipping/loading findings listed above.
