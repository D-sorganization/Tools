# AGENT_HANDOFF â€” Vibroacoustic Ingestion

Last updated: 2026-09-10. Update with each implementation PR and main delivery.

## Where This Tool Is Headed

Tools #5074 owns calibrated impact-sound analysis under #5068. Reuse this package
for recordings and spectral preparation. Golf dynamics live in golf_club;
UpstreamDrift consumes the exact provider revision through thin adapters.
No calibrated golf data or perceptual evidence is available in this work.

## Recent Activity (grounding)

PR #5084 introduced ingestion; #5106 merged its strict boundary repair.
PR #5156 supplies complex H1/coherence, with hosted Python 3.11/3.12 evidence
on the exact recorded tree. Review #5157 now adds explicit affine calibration,
exact identity and shared first-order/exact independent-block uncertainty.
All 168 Windows ingestion/report/API controls pass (97.09% coverage), as do
NumPy-aware and isolated-hook typing and root Ruff. Final governance/publication
pass through normal publication at d5f842278 in PR #5159. Protected CI/review
remain. See CALIBRATION_DEVELOPMENT.md and CALIBRATION_RESULTS.json under
docs/development/impact-acoustics. No Linux calibration result is claimed.

## Must-Read Architecture Pointers

1. `docs/specs/IMPACT_DYNAMICS_ACOUSTICS.md`: full program requirements.
2. `docs/development/impact-acoustics/SIGNAL_BOUNDARY_QUALIFICATION.md`: derivation,
   behavioral compatibility, tests and unresolved measurement authority.
3. `measurement.py` and `_waveform_contracts.py`: samples, legacy hashes and lag.
4. `spectral.py` and `_spectral_frames.py`: shared windows and finite estimators.
5. `tests/test_signal_boundaries.py`: independent refusal and SciPy controls.

## Gate Commands (this tool)

```bash
python -m pytest src/shared/python/swing_sim/vibroacoustics/tests -o addopts= -q
python -m pytest tests/test_shared_package_api_stability.py -o addopts= -q
uvx --from ruff==0.14.10 ruff check .
uvx --from ruff==0.14.10 ruff format --check .
pre-commit run mypy --hook-stage pre-push --files src/shared/python/swing_sim/vibroacoustics/measurement.py src/shared/python/swing_sim/vibroacoustics/spectral.py src/shared/python/swing_sim/vibroacoustics/_waveform_contracts.py src/shared/python/swing_sim/vibroacoustics/_spectral_frames.py src/shared/python/swing_sim/vibroacoustics/tests/test_signal_boundaries.py
```

Also run all nine manual gates specified by the root AGENTS.md before and after
calculation changes. Stage new modules before generating module inventory;
review every API baseline delta and retain publication blockers.

## Do-Not List

- Do not interpret a measured label or legacy hash as calibration authentication.
- Do not convert complex samples or undefined bins into apparently valid reals.
- Do not claim a unique correlation peak establishes synchronization.
- Do not equate positive H1 excitation or Nyquist margin with identification.
- Do not copy these computations into UpstreamDrift or create parallel ingestion.

## Roadmap (ordered)

1. Resolve protected CI/review on PR #5156 and #5159; retain source evidence.
2. Integrate reviewed complex FRF and explicit calibration through thin adapters.
3. Qualify uncertainty, sensor/observer transfer and radiation assumptions.
4. Validate held-out physical measurements and run the blinded sweetness study.
