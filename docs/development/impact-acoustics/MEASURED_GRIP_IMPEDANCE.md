# Measured Grip Translation/Rotation Impedance, Passivity and FRF Agreement

## Ownership and Status

Tools #5072, parent #5068 (Impact Acoustics program).
This checkpoint establishes the measured grip impedance data schema, passivity auditing,
passive model parameter identification, full/reduced FRF agreement under quantified
uncertainty, and consumer integration into `GripBoundary`.

## Wire Format and Data Contracts

The versioned explicit data format is:

    golf_club.measured_grip_impedance/1

Implemented in `src/shared/python/golf_club/measured_grip_impedance.py` and
`_measured_grip_contracts.py`.

### Schema Attributes

- `dataset_id`: Unique identifier for the measured dataset (e.g. `kit-xh-translation-p1`).
- `frame_id`: Coordinate frame convention (e.g. `grip`).
- `axis`: Declared motion axis (`tx`, `ty`, `tz` for translations; `rx`, `ry`, `rz` for rotations; or `spatial_6dof`).
- `grip_force_n`: Measured gripping/clamping force in Newtons.
- `push_force_n`: Measured axial push/feed force in Newtons.
- `frequency_band_hz`: Closed physical frequency interval `[f_min, f_max]`.
- `sources`: Provenance declarations, requiring `artifact_sha256` and `calibration_sha256` for measurement-derived data.
- `samples`: Ordered list of `GripFrequencySample` records.

### Sample Fields and Units

- `frequency_hz`: Excitation frequency in Hertz ($f > 0$).
- `angular_frequency_rad_s`: Angular frequency in radians/second ($\omega = 2\pi f$).
- `impedance_real`: Real part of complex mechanical impedance (damping):
  - Translation: $\text{N}\cdot\text{s}/\text{m}$.
  - Rotation: $\text{N}\cdot\text{m}\cdot\text{s}/\text{rad}$.
- `impedance_imag`: Imaginary part of mechanical impedance ($\omega M - K/\omega$):
  - Translation: $\text{N}\cdot\text{s}/\text{m}$.
  - Rotation: $\text{N}\cdot\text{m}\cdot\text{s}/\text{rad}$.
- `magnitude_std`: Measured standard deviation of impedance magnitude.
- `phase_std_rad`: Measured standard deviation of impedance phase in radians.
- `is_interpolated`: Boolean flag indicating whether the bin was processed/interpolated (e.g. electrical interference bins at 100, 200, 300, 400 Hz).

## Passivity and Dissipated Power

For a positive frequency convention $\exp(+i\omega t)$, linear impedance $Z(\omega)$ relates velocity to wrench:

$$w(\omega) = Z(\omega) v(\omega)$$

Cycle-average dissipated power is:

$$P_{\text{diss}}(\omega) = \frac{1}{2} \text{Re}\big(v^H Z(\omega) v\big) = \frac{1}{2} v^H Z_H(\omega) v \ge 0$$

where $Z_H(\omega) = \frac{1}{2}(Z(\omega) + Z(\omega)^H)$ is the Hermitian part.

- `audit_grip_passivity(dataset)` computes the minimum real eigenvalue and dissipated power across all sample frequencies. Any negative real impedance indicates active behavior / measurement artifact and violates passivity.
- `fit_passive_grip_impedance(dataset)` fits positive semi-definite Gram factors:

$$M = F_M^T F_M \succeq 0, \quad C = F_C^T F_C \succeq 0, \quad K = F_K^T F_K \succeq 0$$

guaranteeing that the identified continuous impedance:

$$Z_{\text{model}}(\omega) = C + i\left(\omega M - \frac{K}{\omega}\right)$$

has $\text{Re}(Z_{\text{model}}(\omega)) = C \succeq 0$ and is unconditionally passive for all $\omega > 0$.

For the scalar measured axis, the damping estimator uses its closed-form
non-negative least-squares solution $C=\max(0,\operatorname{mean}(\operatorname{Re}(Z)))$.
The imaginary-part estimator solves

$$\min_{M,K \geq 0}\left\|\omega\,\operatorname{Im}(Z_{\mathrm{meas}})-
\left(\omega^2 M-K\right)\right\|_2^2.$$

The implementation enumerates the unconstrained solution, each coordinate boundary,
and the origin, selecting the least-residual feasible candidate.  This is the exact
active-set solution for the two-parameter convex problem; it does not independently
clip an unconstrained mass or stiffness estimate.  The estimator preserves passivity,
but its synthetic tests are numerical checks rather than physical coefficient
identification.

## FRF Magnitude and Phase Agreement with Quantified Uncertainty

Given measured impedance $Z_{\text{meas}}(\omega)$ with standard deviation $\sigma_{|Z|}(\omega)$ and candidate model $Z_{\text{model}}(\omega)$:

1. **Relative Magnitude Error**:
   $$\epsilon_{\text{mag}}(\omega) = \frac{\big| |Z_{\text{model}}(\omega)| - |Z_{\text{meas}}(\omega)| \big|}{\max(|Z_{\text{meas}}(\omega)|, \text{floor})}$$
   Antiresonance and near-zero bins use an explicit floor ($\ge 10^{-6}$) to avoid singular division.
2. **Phase Error**:
   $$\Delta\theta(\omega) = \big| \text{atan2}(\sin(\theta_{\text{model}} - \theta_{\text{meas}}), \cos(\theta_{\text{model}} - \theta_{\text{meas}})) \big|$$
   wrapped to $[-\pi, \pi]$ rather than raw phase subtraction.
3. **Uncertainty Coverage**:
   $$z(\omega) = \frac{|Z_{\text{model}}(\omega) - Z_{\text{meas}}(\omega)|}{\max(\sigma_{|Z|}(\omega), 10^{-9})}$$
   Assessing whether the model predictions fall within the declared $k\sigma$ confidence interval (default $k=2.0$).

## Physical Band & Strain Qualification

- Frequency evaluations are strictly bounded within `[f_min, f_max]` (e.g. 10–500 Hz for translation/rotation $x_h, z_h$, 10–100 Hz for rotation $y_h$). Uncalibrated extrapolation outside this band is refused.
- Operating beam strain:
  $$\epsilon_{\max} = \max_s |\kappa(s)| r_{\text{outer}} + |\epsilon_{\text{axial}}| \le \epsilon_{\text{limit}}$$
  refuses operating points exceeding declared linear elastic limits (default 0.005 / 0.5%).

## Consumer Integration

- `measured_grip_to_boundary(dataset)` and `passive_impedance_to_boundary(grip, axis=0)` convert the identified passive parameters ($m_{\text{eff}}, c_g, k_g$) into `shared.python.golf_club.impact_coupling.GripBoundary`.
- This seamlessly drives `simulate_coupled_impact` and Club Tester heavy-hit workflows directly from measured boundary datasets.
