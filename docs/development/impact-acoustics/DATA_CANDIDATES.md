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
