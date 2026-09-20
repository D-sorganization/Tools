# Non-Spherical Oblique Contact Mechanics & Moving Center of Pressure (IA-T4 #5073)

This technical reference documents the implementation of non-spherical oblique contact
mechanics, moving Center of Pressure (COP) kinematics, dynamic gear-effect torque generation,
high-frequency face/hosel modes, and multi-channel energy balance conservation for Issue #5073.

## 1. 3D Curved Face & Non-Spherical Geometry

Golf club heads feature dual-curvature face profiles:
- **Bulge radius** $R_{\text{bulge}}$: horizontal radius of curvature (heel-to-toe).
- **Roll radius** $R_{\text{roll}}$: vertical radius of curvature (crown-to-sole).

In local face material coordinates $(x, y, z)$:
$$z(x, y) = -\frac{1}{2}\left(\frac{x^2}{R_{\text{bulge}}} + \frac{y^2}{R_{\text{roll}}}\right)$$

Outward unit normal field:
$$n(x, y) = \frac{(x / R_{\text{bulge}}, y / R_{\text{roll}}, 1)}{\sqrt{(x / R_{\text{bulge}})^2 + (y / R_{\text{roll}})^2 + 1}}$$

The contact projection (`project_cop`) identifies the instantaneous closest point on the face
satisfying collinearity $(p_{\text{ball}} - p_{\text{cop}}) \parallel n(p_{\text{cop}})$ via quadratic
Newton convergence in $\le 3$ iterations to $< 10^{-13}$ m.

## 2. Moving Center of Pressure (COP) Kinematics

As the ball compresses and rolls/slips during oblique impact, the Center of Pressure $p_{\text{cop}}(t)$
migrates across the curved face.
- **Migration Velocity vs Material Velocity**:
  Force power is work-conjugate to material velocities $v_{\text{mat}}(p_{\text{cop}})$, NOT geometric
  migration rate $\dot{p}_{\text{cop}}$:
  $$P_{\text{mech}} = F_{\text{contact}} \cdot (v_{\text{mat, ball}}(p_{\text{cop}}) - v_{\text{mat, face}}(p_{\text{cop}}))$$
- **Dynamic Lever Arm & Physical Gear Effect**:
  The lever arm from the clubhead center of mass to the moving COP:
  $$r_{\text{cop}}(t) = p_{\text{cop}}(t) - p_{\text{com}}(t)$$
  Produces instantaneous dynamic torque:
  $$\tau_{\text{head}}(t) = r_{\text{cop}}(t) \times F_{\text{face}}(t)$$
  - Toe strikes ($x > 0$) generate opening torque ($\tau_y > 0$).
  - Heel strikes ($x < 0$) generate closing torque ($\tau_y < 0$).
  - High strikes ($y > 0$) generate loft-increasing torque ($\tau_x < 0$).
  - Low strikes ($y < 0$) generate delofting torque ($\tau_x > 0$).

## 3. High-Frequency Face & Hosel Modes

Flexible clubhead structural dynamics are captured via modal coordinates $\eta_m$:
- **Trampoline Mode** (~4500 Hz): Thin face plate membrane mode providing dynamic compliance.
- **Hosel Bending Mode** (~1800 Hz): Out-of-plane bending compliance at the shaft-head junction.
- **Hosel Torsion Mode** (~2200 Hz): In-plane torsional twisting at the hosel junction.

Modal equation:
$$M_m \ddot{\eta}_m + C_m \dot{\eta}_m + K_m \eta_m = Q_m$$
where generalized forces $Q_m = \Phi_m(p_{\text{cop}}) F_n$.
The modal deflection $u_{\text{modal}} = \sum_m \Phi_m(p_{\text{cop}}) \eta_m$ feeds back directly into
the effective contact gap:
$$g_{\text{eff}} = g_{\text{rigid}} - u_{\text{modal}}$$

## 4. Multi-Channel Energy Balance

The coupled system strictly balances power:
$$P_{\text{mech}} = P_{\text{modal\_transfer}} + P_{\text{normal\_contact}} + P_{\text{tangential\_contact}}$$
where:
- $P_{\text{modal\_transfer}} = F_n \cdot v_{\text{modal}}$
- $P_{\text{normal\_contact}} = F_n \cdot \dot{g}_{\text{eff}}$
- $P_{\text{tangential\_contact}} = F_t \cdot v_t$

The instantaneous power residual $\epsilon_{\text{power}} \equiv 0$ within numerical roundoff ($< 10^{-10}$ W).
