# Measured Data Candidates for Later Acoustic Qualification

This is a source-access and suitability record, not experimental validation.
No dataset recordings have been downloaded, inspected or fitted. A bounded
remote ZIP directory has been inspected as described below. The physical
golf/player and blinded preference requirements remain open in PROGRESS.md.

## RealImpact

Primary sources inspected on 2026-09-07 local date:

- [Clarke et al., CVPR 2023, full paper HTML](https://arxiv.org/html/2306.09944v1)
- [Authors' project and recording setup](https://samuelpclarke.com/realimpact/)
- [Repository at inspected commit fca2bd6](https://github.com/samuel-clarke/RealImpact/tree/fca2bd6cbb7e9f96ac61328d2a0d51594bf01987)

The paper describes 150,000 recordings of 50 household objects, with measured
impact force, impact location and microphone coordinates. Synchronized force
and audio capture uses 48 kHz sampling. Objects rest on a thread mesh in a
treated room to approximate free vibration. This is potentially useful for
testing identified acoustic transfer and spatial prediction on measured data.
It contains neither swinging clubs nor varied human grip conditions in the
described experiment. Angular undersampling at higher frequencies is a stated
limitation. It cannot establish a golfer's grip effect or preferred impact sound.

The authors' README offers preprocessed data and says remaining code and raw
data are still being packaged. Its download script retrieves per-object ZIP
archives from Stanford over HTTP. Archive contents, checksums, availability,
sizes and dataset-specific redistribution terms have not been verified here.
The repository's MIT license explicitly describes software/documentation;
that is not a separately verified license for every external recording.

The [inspected preprocessing script](https://github.com/samuel-clarke/RealImpact/blob/fca2bd6cbb7e9f96ac61328d2a0d51594bf01987/preprocess_measurements.py)
saves normalized sounds and deconvolved arrays, plus gain-adjusted variants,
force arrays, vertex identifiers and listener positions. Its deconvolution
divides Fourier transforms without explicit spectral regularization. The
hammer window uses a 2% threshold; the paper describes 1%. Normalized arrays
cannot independently establish absolute SPL. The presence of a force conversion
constant is not, by itself, a verified calibration chain.

Before T5 uses this candidate, pin the source and downloaded bytes; inspect
units, microphone sensitivity, gains, sample alignment, clipping, force spectra
and preprocessing losses. Register held-out impact/listener locations before
fitting. Assess transfer phase, weak-input frequency bins, repeatability and
spatial sampling limits. Keep normalized shape metrics separate from calibrated
pressure error. Report results as household-object method evidence, with a
separate golf-equipment validation gate. Do not run the authors' bulk download
script blindly or copy its unregularized division as a qualified estimator.

## Iron-Plate Archive Access Check

On 2026-09-08 UTC, HTTPS HEAD for the authors' `67_IronPlate.zip` URL returned
200, length 2,310,122,028 bytes, byte-range support, Last-Modified
`Mon, 10 Apr 2023 09:47:24 GMT`, and ETag `"6433db2c-89b1aa2c"`.
A subsequent request for only the last 131,072 bytes returned 206 and the exact
requested Content-Range. Python's ZIP reader inspected that directory without
extracting files or reading the audio payload. ETag and partial access are not
a complete archive SHA-256 or a recording-integrity check.

The directory lists `preprocessed/deconvolved_0db.npy` (2,504,508,128 bytes
uncompressed), `transformed.obj`, material assets, and NPY arrays for micID,
vertexXYZ, listenerXYZ, angle, vertexID and distance. It lists no separate force,
raw sound, or calibration file. The available archive is therefore narrower
than the preprocessing script's possible outputs. It may support studying
already-deconvolved responses; it does not by itself provide paired raw
force/audio for an independent transfer-identification audit. Array headers,
values, units, clipping and calibration remain uninspected. No model was fitted.

Source: [authors' hosted archive](https://downloads.cs.stanford.edu/viscam/RealImpact/67_IronPlate.zip),
selected from the pinned repository's `dataset/object_names.txt`. No external
recordings or derived data are redistributed in this repository.

## KIT Hand-Arm Impedance Candidates

Primary KITopen metadata inspected on 2026-09-08 UTC identifies a measured
boundary-model candidate family, distinct from golf impacts or acoustic
preferences. No payload has been downloaded, inspected, fitted or approved.

| Record                                                                                                           | Declared coverage                                                                      | Potential use                                                        |
| ---------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| [Translation xh, 1000194060](https://publikationen.bibliothek.kit.edu/1000194060), DOI 10.35097/t28dnwf7rhmv2dwm | 13 participants, 10–500 Hz, complex impedance/apparent mass and validation records     | Translational boundary identification within the recorded conditions |
| [Rotation xh, 1000185119](https://publikationen.bibliothek.kit.edu/1000185119), DOI 10.35097/1r2qfkquz7mzar0s    | Six participants, 10–500 Hz, direct and cross-axis response, raw and evaluated records | Rotational and coupling controls with an unloaded-handle reference   |
| [Rotation zh, 1000185118](https://publikationen.bibliothek.kit.edu/1000185118), DOI 10.35097/vfrrc2hbxkujanz8    | Six participants, 10–500 Hz, direct and cross-axis response                            | Complementary rotational identification and calibration checks       |
| [Rotation yh, 1000194062](https://publikationen.bibliothek.kit.edu/1000194062), DOI 10.35097/h5p055shpntph522    | Ten participants, 10–100 Hz, complex rotational impedance/apparent inertia             | Restricted-band handle-axis response and validation                  |

The translation-xh record declares CC BY 4.0 and separate main/validation MATLAB
structs. Reported response values at 100, 200, 300 and 400 Hz are interpolated
because of electrical interference. Those bins must be flagged as processed
values, not treated as independent measured response points. Its overlapping
segments are not independent participants or independent trials. [Source](https://publikationen.bibliothek.kit.edu/1000194060).

Both xh/zh rotational records declare CC BY 4.0, measured grip/push force,
condition exclusion indicators, raw torque/angular-acceleration channels and
an unloaded-handle file. Their reported cross-axis terms are useful partial
coupling evidence; they do not alone identify every element of a full 6×6
impedance. Coordinate conventions and calibration must be checked against
actual file contents before comparing directions. [xh](https://publikationen.bibliothek.kit.edu/1000185119), [zh](https://publikationen.bibliothek.kit.edu/1000185118).

The yh rotational metadata declares torque/angular-velocity impedance in
N m s/rad and apparent inertia in kg m², but its excitation paragraph uses
linear-acceleration units while describing a rotational experiment. Resolve
this metadata inconsistency from acquisition/calibration records before using
amplitudes or units. Preserve its narrower frequency domain and condition
ranges. [Source](https://publikationen.bibliothek.kit.edu/1000194062).

A related [model-parameter record, 1000185357](https://publikationen.bibliothek.kit.edu/1000185357)
(DOI 10.35097/r7uckpr276vanaua) describes fitted three-, four- and five-mass
oscillators and errors for weighted/unweighted fitting. Its declared license
is CC BY-SA 4.0, distinct from the measurement records. These are candidate
published comparisons, not parameters already imported into Tools. Its metadata
calls the contents CSV while describing sheets; inspect the actual format.

Inference and next steps: pin payload hashes, retain provenance and processing
flags, verify complex phase/sign, axes, calibration and physical units, then
register participant/trial/condition holdouts before fitting. Keep overlapped
segments grouped to avoid leakage. Compare the simple constant-coefficient
coordinate law against frequency-dependent passive alternatives using held-out
complex response and uncertainty. Do not extrapolate these handle measurements
to swinging two-hand grips, unobserved matrix entries, kilohertz golf ringdown
or preferred sound. These data cannot close the golf/physical/blinded gates.

Access evidence is limited: DOI resolution by HEAD reaches the RADAR dataset
landing page with HTTP 200. The web reader refuses that landing URL as unsafe
(non-retryable); no payload fetch is attempted through another tool to bypass
that refusal. File sizes, archive members, byte hashes and independent
calibration validation remain unverified. KITopen metadata above remains the
reviewed source-access boundary.
