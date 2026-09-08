# Newly Merged Vibroacoustic Ingestion: Reuse and Qualification

Tools PR #5084 merged as `86a725c6c2d4765d61aa9e7c59d771c3e075f6bd`.
It explicitly implements an early subset of IA-T5 #5074 and leaves that issue
open. Its `swing_sim.vibroacoustics` package adds waveform metadata, source-kind
checks, sample hashing, clipping/Nyquist checks, circular correlation, Welch PSD,
single-mode decay estimates and H1 magnitude estimates. The PR reports eighteen
tests. Reuse this package when extending measurement handling; do not create a
competing ingestion/PSD module inside `golf_club` or UpstreamDrift.

No calibrated golf experiment, radiation prediction, observer transfer,
held-out validation or blinded sweetness result follows from that merge.
The source distinguishes measured and synthesized records by a caller-supplied
enum. That label alone cannot authenticate a measurement or calibration.

## Exact-Commit Boundary Probes

Read-only synthetic probes of the merged modules reproduced these results:

| Probe                                                                   | Observed Behavior                                        | Qualification Need                                                                                           |
| ----------------------------------------------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Change sensitivity from 1 to 2, retain samples/rate/unit/calibration ID | `raw_data_hash` remains identical                        | Specify and bind the complete calibration identity; distinguish source/raw and calibrated-data hashes.       |
| Impulse at index 8, response at index 5 of a 16-sample record           | `align_time_shift` returns 13 for an advance of 3        | Explicit circular versus signed physical lag, overlap, ambiguity and synchronized timebase contracts.        |
| `psd_welch` with allowed `segment_length=2`                             | Nonfinite PSD and divide warning                         | Nondegenerate window-energy domain and finite-output policy.                                                 |
| H1 on two zero recordings, segment length 4                             | Nonfinite response and divide warning                    | Excitation/identifiability masks or explicit refusal; finite values cannot be invented for unsupported bins. |
| Complex input `[1+2j, 2+3j]`                                            | Accepted as `[1, 2]` with imaginary-part discard warning | Strict real waveform input, including boolean/coercion policy.                                               |

These are software boundary counterexamples, not measurement results. No code
was changed in the ingestion package during this review. `estimate_frf_h1`
currently returns magnitude only, so phase-sensitive radiation/observer transfer
needs an explicit extension and compatibility plan. The writable sample array,
sensitivity semantics, anti-alias/source bandwidth, coherence/uncertainty and
single-mode/noise/truncation fit limits also need qualification before use in
physical claims. Current labels such as `adequate` establish only the implemented
Nyquist inequality, not a calibrated measurement chain.

Continue this work under #5074 after checking its claim and existing issue
ownership. Add failing tests for these cases before fixes; preserve public
contracts or coordinate explicit versioned changes. The new #5083 study-wire
PR is also mentioned by main's handoff and needs inventory review before T6
implementation. All cross-repo impact/acoustic epics retain their full scope.

PR #5083 is open at `98d29ee71c21f99eaf760ba4cd67e9970efbf62a`, branch
`bot/issue-5075-impact-study-wire`. Its description reports an ImpactStudyV1
wire with separate launch/contact/vibration/acoustic metrics, provenance,
uncertainty, evidence-tier checks and explicit absent-acoustics handling, plus
nineteen tests. Those are reported PR capabilities, not source/CI qualification
by this review. Its own follow-up scope retains UI cutover and UpstreamDrift
integration after provider qualification; #5075 must remain open.
