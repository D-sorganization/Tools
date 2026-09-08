# Measured Data Candidates for Later Acoustic Qualification

This is a source-access and suitability record, not experimental validation.
No dataset recordings have been downloaded, inspected or fitted. The physical
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
