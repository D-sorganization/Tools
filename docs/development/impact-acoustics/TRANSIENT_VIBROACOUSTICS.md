# Transient Vibroacoustic Radiation & Acoustic Field Solver (IA-T5, #5074)

This technical reference documents the implementation of discretized transient vibroacoustic radiation,
retarded-time Rayleigh surface integrals, observer directivity transfer, ball impact dipole radiation,
standardized psychoacoustics (ISO 532-1, DIN 45692), and calibrated sound recordings for Issue #5074.

## 1. Architectural Overview

Building upon `shared.python.swing_sim.vibroacoustics`, this module provides a physically qualified,
transient acoustic radiation and observer evaluation tier:

- `_boundary_radiation_records.py`: Radiating surface element and mesh representations (`RadiatingElement`, `RadiatingSurfaceMesh`, `AcousticMedium`).
- `radiation.py`: Retarded-time Rayleigh surface integral solver (`TransientRadiationSolver`, `ModalRadiationTransfer`).
- `ball_radiation.py`: Ball impact acoustic dipole radiation (`BallAcousticRadiation`).
- `observer.py`: Field evaluation geometries (`ObserverLocation`, `MicrophoneArray`, `HeldOutObserverComparison`).
- `psychoacoustics.py`: Standardized sound quality metrics (`calculate_specific_loudness`, `calculate_sharpness`, `compute_spl`, `compute_peak_spl`, `compute_sound_exposure_level`, `compute_equivalent_sound_level`, `AcousticReferenceAlgorithm`).
- `calibrated_sound.py`: Rigorous provenance binding and timebase synchronization (`CalibratedSoundRecording`, `synchronize_timebases`).

## 2. Discretized Transient Boundary Radiation (Rayleigh Surface Integral)

For a planar or mildly curved vibrating boundary (such as the golf club face plate) mounted in an effective baffle,
the transient acoustic pressure at an observer location $x_{\text{obs}}$ is governed by Rayleigh's first integral:

$$p(x_{\text{obs}}, t) = \frac{\rho_0}{2 \pi} \sum_{e=1}^N \frac{\ddot{w}_e(t - R_e / c_0)}{R_e} A_e$$

where:
- $\rho_0$ is the acoustic medium density (default $1.204 \text{ kg/m}^3$ in standard air).
- $c_0$ is the acoustic medium sound speed ($343.2 \text{ m/s}$ at $20^\circ\text{C}$).
- $A_e$ is the radiating surface area of element $e$.
- $R_e = \|x_{\text{obs}} - x_e\|$ is the Euclidean distance from element centroid $x_e$ to observer $x_{\text{obs}}$.
- $t - R_e / c_0$ is the retarded time accounting for finite acoustic propagation delay.
- $\ddot{w}_e$ is the outward normal surface acceleration of element $e$.

### Spatial Discretization & Element Convergence Criterion

To avoid spatial aliasing and numerical dispersion in high-frequency modal radiation up to $f_{\text{max}}$:
$$h \le \frac{\lambda_{\text{min}}}{6} = \frac{c_0}{6 f_{\text{max}}}$$

For a golf driver face analyzed up to $f_{\text{max}} = 6000\text{ Hz}$, $h \le 9.53\text{ mm}$, satisfied by
at least a $16 \times 8$ mesh on a $0.10 \times 0.05\text{ m}$ driver face plate.

### Modal Radiation Transfer

Given normal modes $\Phi_m(x)$ and generalized modal accelerations $\ddot{\eta}_m(t)$:
$$\ddot{w}_e(t) = \sum_{m} \Phi_m(x_e) \ddot{\eta}_m(t)$$

Linear superposition guarantees that the total radiated acoustic field equals the sum of modal contributions:
$$p(x_{\text{obs}}, t) = \sum_{m} p_m(x_{\text{obs}}, t)$$

## 3. Ball Impact Acoustic Dipole Radiation

In addition to structural boundary radiation from the clubhead face and body, the unsteady contact force
$F_c(t)$ exerted between clubhead and ball acts as an acoustic dipole source:

$$p_{\text{dipole}}(x_{\text{obs}}, t) = \frac{x_{\text{rel}} \cdot e_{\text{force}}}{4 \pi c_0 R^2} \dot{F}_c(t - R / c_0) + \frac{x_{\text{rel}} \cdot e_{\text{force}}}{4 \pi R^3} F_c(t - R / c_0)$$

where:
- The first term is the far-field radiating dipole proportional to the force time-derivative $\dot{F}_c$.
- The second term is the near-field hydrodynamic dipole proportional to $F_c$.
- Directivity exhibits a classic $\cos\theta$ pattern along the impact force normal axis $e_{\text{force}}$.

## 4. Observer Directivity & Microphone Arrays

- **Directivity**: Evaluated across spherical coordinates $(\theta, \phi)$ at constant radius $R$, capturing
  both dipole directivity and boundary baffle diffraction.
- **Microphone Array**: Multi-channel synchronous evaluation across designated field coordinates
  (e.g., player ear positions, ground microphones, launch monitor mic fixtures).
- **Held-Out Observer Comparison**: Quantifies prediction fidelity against physical recordings at
  unfitted receiver positions, computing L2 normalized relative error and Pearson correlation.

## 5. Standardized Psychoacoustic Metrics

Impact sound signatures are characterized through standardized auditory metrics:

- **Sound Pressure Level (SPL)**: RMS pressure referenced to $p_0 = 20 \ \mu\text{Pa}$:
  $$\text{SPL} = 20 \log_{10}\left(\frac{p_{\text{rms}}}{p_0}\right)$$
- **Peak Sound Pressure Level ($L_{\text{peak}}$)**:
  $$L_{\text{peak}} = 20 \log_{10}\left(\frac{\max |p(t)|}{p_0}\right)$$
- **Sound Exposure Level (SEL)**:
  $$\text{SEL} = 10 \log_{10}\left(\frac{1}{t_0} \int_0^T \frac{p^2(t)}{p_0^2} dt\right), \quad t_0 = 1\text{ s}$$
- **Equivalent Continuous Sound Level ($L_{\text{eq}}$)**:
  $$L_{\text{eq}} = 10 \log_{10}\left(\frac{1}{T} \int_0^T \frac{p^2(t)}{p_0^2} dt\right)$$
- **Stationary Loudness (ISO 532-1 / Zwicker)**: Evaluated across 24 critical Bark bands,
  calibrated against a $1\text{ kHz}$ reference pure tone at $40\text{ dB SPL} = 1.0\text{ sone}$.
- **Spectral Sharpness (DIN 45692)**: Weighted center of gravity of specific loudness over critical bands $z$:
  $$S = c_{\text{sharp}} \frac{\int_0^{24} N'(z) g(z) z \, dz}{\int_0^{24} N'(z) \, dz}$$
  with $g(z)$ high-frequency weighting and calibration constant $c_{\text{sharp}} = 0.1176$ yielding
  $1.0\text{ acum}$ ($\pm 0.05$) for the standard narrow-band reference noise centered at $1\text{ kHz}$ at $60\text{ dB SPL}$.

## 6. Calibrated Sound Recordings & Synchronization

- **Provenance Binding**: `CalibratedSoundRecording` strictly pairs a `CalibratedWaveform` (unit `'Pa'`)
  with its spatial `ObserverLocation`. Synthesized tones are structurally rejected from being passed
  as measured physical recordings.
- **Timebase Synchronization**: `synchronize_timebases` computes the exact sub-sample or integer lag
  between two microphone recordings using cross-correlation, validating agreement with acoustic time-of-flight
  propagation $\Delta t = \Delta R / c_0$.
