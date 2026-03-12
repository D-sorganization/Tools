"""
LaTeX-quality math popup for the Pendulum Simulator.

Displays rendered mathematical equations for:
- Mass matrix derivation and physical interpretation
- Full equations of motion for double, triple, and golfer model
- Coriolis, gravity, friction, and joint limit terms
- Energy conservation and Lagrangian derivation
- Golfer 8-DOF Baumgarte constrained system (KKT formulation)

Uses QTextBrowser with rich HTML + CSS for professional-quality rendering.
If matplotlib is available, equations are rendered as PNG via mathtext
for crisp display.

Design by Contract
------------------
- show_equations_popup(parent, topic) is the single entry point.
- topic must be one of the EquationTopic enum values.
- The popup is non-modal so the user can keep it open alongside the sim.

DRY
---
HTML template and styling are defined once. Topic content is pluggable.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QDialog, QWidget

logger = logging.getLogger(__name__)


class EquationTopic(Enum):
    """Available equation topics."""

    MASS_MATRIX = "mass_matrix"
    EQUATIONS_OF_MOTION = "equations_of_motion"


# ---------------------------------------------------------------------------
# Shared CSS for all equation popups
# ---------------------------------------------------------------------------

_CSS = """
body {
    background: #1a1a28;
    color: #c0c0d8;
    font-family: 'Segoe UI', 'DejaVu Sans', Arial, sans-serif;
    font-size: 14px;
    line-height: 1.7;
    padding: 20px 28px;
    max-width: 800px;
}
h1 { color: #6fa8dc; font-size: 22px; border-bottom: 2px solid #3a5a8c; padding-bottom: 8px; margin-top: 28px; }
h2 { color: #6fa8dc; font-size: 18px; margin-top: 24px; border-bottom: 1px solid #303050; padding-bottom: 4px; }
h3 { color: #7db8ec; font-size: 15px; margin-top: 18px; }
.eq {
    background: #12121e;
    border: 1px solid #303050;
    border-radius: 6px;
    padding: 14px 20px;
    margin: 12px 0;
    font-family: 'Cambria Math', 'STIX Two Math', 'Latin Modern Math', Georgia, serif;
    font-size: 16px;
    color: #a0e0a0;
    overflow-x: auto;
    line-height: 2.0;
}
.eq-inline {
    font-family: 'Cambria Math', 'STIX Two Math', Georgia, serif;
    color: #a0e0a0;
    font-size: 15px;
}
table.params {
    border-collapse: collapse;
    margin: 10px 0;
    width: 100%;
}
table.params td {
    padding: 6px 12px;
    border-bottom: 1px solid #303050;
    vertical-align: top;
}
table.params td:first-child {
    font-family: 'Cambria Math', Georgia, serif;
    color: #a0e0a0;
    white-space: nowrap;
    width: 120px;
}
.note {
    background: #1e1e32;
    border-left: 3px solid #6fa8dc;
    padding: 10px 16px;
    margin: 12px 0;
    font-style: italic;
}
.matrix {
    font-family: 'Cambria Math', Georgia, serif;
    font-size: 15px;
    white-space: pre;
    line-height: 1.8;
}
ul { padding-left: 22px; }
li { margin-bottom: 6px; }
"""

# ---------------------------------------------------------------------------
# Mass Matrix content
# ---------------------------------------------------------------------------

_MASS_MATRIX_HTML = f"""
<html><head><style>{_CSS}</style></head><body>

<h1>Mass (Inertia) Matrix — Derivation &amp; Interpretation</h1>

<h2>1. What Is the Mass Matrix?</h2>
<p>
The <b>mass matrix</b> <span class="eq-inline">M(q)</span> is the
configuration-dependent inertia tensor of the mechanism in generalized
coordinates.  It maps joint accelerations to the generalized forces
required to produce them:
</p>
<div class="eq">
τ = M(q) · q̈ + C(q, q̇) + G(q)
</div>
<p>
Physically, <span class="eq-inline">M(q)</span> tells you "how heavy does
the system feel" when you try to accelerate a particular joint.
It is always <b>symmetric</b> (M = Mᵀ) and <b>positive definite</b>
(all eigenvalues > 0), guaranteeing a unique solution for
<span class="eq-inline">q̈ = M⁻¹(τ − C − G)</span>.
</p>

<h2>2. Double Pendulum (2R) Mass Matrix</h2>
<p>
For a two-segment pendulum with generalized coordinates
<span class="eq-inline">q = [θ₁, φ]ᵀ</span> where θ₁ is the absolute
shoulder angle and φ is the wrist angle relative to the arm:
</p>
<div class="eq">
<span class="matrix">
     ┌                                                            ┐
M =  │  (m₁+mₑ)L₁² + mₑL₂² + 2mₑL₁L₂cos φ    mₑL₂² + mₑL₁L₂cos φ  │
     │  mₑL₂² + mₑL₁L₂cos φ                     mₑL₂²                  │
     └                                                            ┘
</span>
</div>
<p>where <span class="eq-inline">mₑ = m₂ + m<sub>club</sub></span>
is the effective distal mass (shaft + clubhead).</p>

<h3>2.1 Parameter Definitions</h3>
<table class="params">
<tr><td>m₁</td><td>Mass of segment 1 (arms), kg</td></tr>
<tr><td>m₂</td><td>Mass of segment 2 (shaft), kg</td></tr>
<tr><td>m<sub>club</sub></td><td>Clubhead point mass at tip of shaft, kg</td></tr>
<tr><td>mₑ</td><td>Effective mass: m₂ + m<sub>club</sub></td></tr>
<tr><td>L₁</td><td>Length of segment 1, m</td></tr>
<tr><td>L₂</td><td>Length of segment 2, m</td></tr>
<tr><td>φ</td><td>Relative wrist angle, rad</td></tr>
</table>

<h3>2.2 Physical Interpretation of Each Entry</h3>
<ul>
<li><b>M₁₁</b> — Effective inertia at the shoulder when the wrist is locked.
Includes contributions from both segments plus the coupling term
<span class="eq-inline">2mₑL₁L₂cos φ</span> that depends on the relative
configuration.</li>
<li><b>M₂₂</b> — Effective inertia at the wrist when the shoulder is locked.
This is simply <span class="eq-inline">mₑL₂²</span>, the inertia of the
distal segment about the wrist.</li>
<li><b>M₁₂ = M₂₁</b> — <b>Inertial coupling</b>. This is the key term for
energy transfer in golf swings. When joint 1 accelerates, the off-diagonal
term creates a "free" torque at joint 2 (and vice versa).  This is how the
proximal-to-distal kinetic chain works.</li>
</ul>

<div class="note">
<b>Key insight:</b> The coupling term is proportional to
<span class="eq-inline">cos φ</span>. Maximum coupling occurs when the
segments are aligned (φ = 0). Coupling vanishes at φ = ±90°.
In a golf swing, the delayed wrist release keeps the segments aligned
for maximum coupling transfer at impact.
</div>

<h3>2.3 Derivation Sketch</h3>
<p>Start from the kinetic energy in Cartesian coordinates, then transform
to generalized coordinates using the Jacobian:</p>
<div class="eq">
T = ½ q̇ᵀ M(q) q̇ = ½ Σᵢ mᵢ vᵢᵀvᵢ
</div>
<p>The velocity of each point mass is obtained via the geometric Jacobian
<span class="eq-inline">vᵢ = Jᵢ(q) q̇</span>, so:</p>
<div class="eq">
M(q) = Σᵢ mᵢ Jᵢᵀ Jᵢ
</div>

<h2>3. Triple Pendulum (3R) Mass Matrix</h2>
<p>
For three segments with coordinates <span class="eq-inline">q = [θ₁, φ₁, φ₂]ᵀ</span>,
the mass matrix is 3×3 symmetric:
</p>
<div class="eq">
<span class="matrix">
      ┌                    ┐
M  =  │  M₁₁   M₁₂   M₁₃ │
      │  M₁₂   M₂₂   M₂₃ │
      │  M₁₃   M₂₃   M₃₃ │
      └                    ┘
</span>
</div>
<p>Each entry follows the same pattern as the 2R case but with
additional terms for the third segment.  The coupling structure means
that accelerating the hub (joint 1) creates inertial torques at
both the arm (joint 2) and club (joint 3) simultaneously.</p>

<div class="note">
In the golf context: hub = sternum-to-shoulder (~0.15 m),
arm = shoulder-to-wrist (~0.60 m), club = wrist-to-clubhead (~1.10 m).
The short first segment means M₁₁ is dominated by the coupled terms.
</div>

<h2>4. Golfer Model (8-DOF) Mass Matrix</h2>
<p>
The golfer upper-body model uses 8 generalized coordinates for a closed
kinematic loop (left arm + right arm + shared club).  The mass matrix is
8×8 but has block structure reflecting the kinematic tree:
</p>
<div class="eq">
<span class="matrix">
      ┌                                              ┐
M  =  │  M<sub>hub</sub>    M<sub>hub,R</sub>   M<sub>hub,L</sub>   M<sub>hub,club</sub>  │
      │  M<sub>hub,R</sub>ᵀ  M<sub>R</sub>      0           M<sub>R,club</sub>   │
      │  M<sub>hub,L</sub>ᵀ  0           M<sub>L</sub>      M<sub>L,club</sub>   │
      │  M<sub>hub,club</sub>ᵀ M<sub>R,club</sub>ᵀ M<sub>L,club</sub>ᵀ M<sub>club</sub>     │
      └                                              ┘
</span>
</div>
<p>
The closed loop introduces 4 holonomic constraints
<span class="eq-inline">Φ(q) = 0</span> that enforce the two hands
meeting at the grip.  These are handled via <b>Baumgarte stabilization</b>
(see Equations of Motion section).
</p>

</body></html>
"""

# ---------------------------------------------------------------------------
# Equations of Motion content
# ---------------------------------------------------------------------------

_EOM_HTML = f"""
<html><head><style>{_CSS}</style></head><body>

<h1>Equations of Motion — Complete Derivation</h1>

<h2>1. Lagrangian Formulation</h2>
<p>
The equations of motion are derived from the Euler-Lagrange equations.
The Lagrangian is <span class="eq-inline">L = T − V</span> where T is
kinetic energy and V is potential energy.  For a driven system with
dissipation:
</p>
<div class="eq">
d/dt (∂L/∂q̇ᵢ) − ∂L/∂qᵢ = τᵢ + τ<sub>friction,i</sub> + τ<sub>limits,i</sub>
</div>
<p>This yields the standard manipulator equation:</p>
<div class="eq">
M(q) · q̈ = τ<sub>drive</sub> + τ<sub>friction</sub> + τ<sub>limits</sub> − C(q, q̇) − G(q)
</div>

<h2>2. Individual Terms</h2>

<h3>2.1 Mass Matrix M(q)</h3>
<p>See the Mass Matrix tab for full derivation. Configuration-dependent,
symmetric positive definite. Computed analytically for each model.</p>

<h3>2.2 Coriolis &amp; Centrifugal Vector C(q, q̇)</h3>
<p>For the 2R pendulum:</p>
<div class="eq">
<span class="matrix">
      ┌                                        ┐
C  =  │  −h (2 θ̇₁ φ̇ + φ̇²)           │
      │   h θ̇₁²                           │
      └                                        ┘

where  h = mₑ L₁ L₂ sin φ
</span>
</div>
<p>
<b>Physical meaning:</b> These terms arise from the velocity-dependent
"fictitious forces" in the rotating reference frames.  The first
component contains both the Coriolis term (∝ θ̇₁ φ̇) and the centrifugal
term (∝ φ̇²).
</p>

<h3>2.3 Gravity Vector G(q)</h3>
<div class="eq">
<span class="matrix">
      ┌                                                      ┐
G  =  │  (m₁ + mₑ) g L₁ sin θ₁  +  mₑ g L₂ sin(θ₁ + φ)  │
      │  mₑ g L₂ sin(θ₁ + φ)                                │
      └                                                      ┘
</span>
</div>
<p>Gravity acts through the COM of each segment.  The absolute angle
of segment 2 is <span class="eq-inline">θ₁ + φ</span> (relative
coordinate convention).</p>

<h3>2.4 Driving Torque τ<sub>drive</sub></h3>
<p>User-specified as polynomial functions of time:</p>
<div class="eq">
τᵢ(t) = c₀ + c₁t + c₂t² + c₃t³ + ...
</div>
<p>Subject to saturation limits (torque clamping):</p>
<div class="eq">
τᵢ,clamped = clip(τᵢ, −τ<sub>max,i</sub>, +τ<sub>max,i</sub>)
</div>
<div class="note">
The absolute-value clamp ensures symmetric limits (±τ<sub>max</sub>)
even if the user enters a negative limit value.  This prevents the
common error of only limiting positive torque.
</div>

<h3>2.5 Friction Torque τ<sub>friction</sub></h3>
<div class="eq">
τ<sub>friction,i</sub> = −bᵢ q̇ᵢ  −  μᵢ sign(q̇ᵢ)
</div>
<table class="params">
<tr><td>bᵢ</td><td>Viscous damping coefficient (N·m·s/rad)</td></tr>
<tr><td>μᵢ</td><td>Coulomb friction magnitude (N·m)</td></tr>
</table>

<h3>2.6 Joint Limit Penalty τ<sub>limits</sub></h3>
<p>Smooth Hermite smoothstep barrier at joint angle limits:</p>
<div class="eq">
τ<sub>lim</sub>(q) = s(d) · [ K·d + B·max(0, −q̇) ]

where d = q<sub>min</sub> − q  (penetration depth)
      s(x) = 3x² − 2x³  (smoothstep blend, x ∈ [0,1])
</div>
<table class="params">
<tr><td>K</td><td>Penalty stiffness (N·m/rad)</td></tr>
<tr><td>B</td><td>Penalty damping (N·m·s/rad)</td></tr>
</table>

<h2>3. Numerical Integration</h2>
<p>The ODE system is solved using adaptive Runge-Kutta methods:</p>
<table class="params">
<tr><td>Double</td><td>RK45 (Dormand-Prince, 5th order), rtol=1e-8, atol=1e-10</td></tr>
<tr><td>Triple</td><td>DOP853 (8th order), rtol=1e-6, atol=1e-8</td></tr>
<tr><td>Golfer</td><td>RK45 with constraint projection at each step</td></tr>
</table>

<h2>4. Energy Conservation</h2>
<div class="eq">
E = T + V = ½ q̇ᵀ M(q) q̇  +  V(q)

dE/dt = q̇ᵀ τ<sub>drive</sub>  +  q̇ᵀ τ<sub>friction</sub>
</div>
<p>
When τ<sub>drive</sub> = 0 and friction = 0, total energy is conserved
(E = const).  This is used as a sanity check for the integrator.
</p>

<h2>5. Golfer Model — Constrained Dynamics (KKT System)</h2>
<p>
The 8-DOF golfer model has 4 holonomic constraints
<span class="eq-inline">Φ(q) = 0</span> enforcing that both hands
grip the same club.  The constrained EOM are:
</p>
<div class="eq">
<span class="matrix">
┌  M    Φ<sub>q</sub>ᵀ ┐ ┌ q̈ ┐   ┌ τ − C − G                                ┐
│            │ │     │ = │                                            │
└  Φ<sub>q</sub>   0   ┘ └ λ  ┘   └ −γ(q,q̇) − 2α Φ̇(q,q̇) − β² Φ(q) ┘
</span>
</div>

<h3>5.1 Constraint Equations</h3>
<p>The 4 constraints enforce that the right-hand endpoint and left-hand
endpoint coincide at the club grip:</p>
<div class="eq">
Φ(q) = p<sub>R,wrist</sub>(q) − p<sub>L,wrist</sub>(q) = 0   (2 eqs: x, y)
Φ(q) = p<sub>R,wrist</sub>(q) − p<sub>club,base</sub>(q) = 0  (2 eqs: x, y)
</div>

<h3>5.2 Baumgarte Stabilization</h3>
<p>To prevent numerical constraint drift, the acceleration-level
constraint is augmented with position and velocity feedback:</p>
<div class="eq">
Φ<sub>q</sub> q̈ = −γ(q,q̇) − 2α Φ̇ − β² Φ
</div>
<table class="params">
<tr><td>α</td><td>Velocity feedback gain (typical: 5–20)</td></tr>
<tr><td>β</td><td>Position feedback gain (typical: 5–20)</td></tr>
<tr><td>Φ<sub>q</sub></td><td>Constraint Jacobian (4×8 matrix)</td></tr>
<tr><td>γ</td><td>Constraint bias: γ = −Φ̇<sub>q</sub> q̇ (velocity-level RHS)</td></tr>
<tr><td>λ</td><td>Lagrange multipliers (constraint forces)</td></tr>
</table>

<div class="note">
<b>Why Baumgarte?</b> Direct constraint enforcement at the acceleration
level allows constraint violations to accumulate (drift).  Baumgarte
feedback drives violations to zero exponentially, like a PD controller
on the constraint error.  Higher α, β = faster stabilization but can
cause stiffness.
</div>

<h2>6. Impulse, Work, and Power</h2>

<h3>6.1 Angular Power</h3>
<div class="eq">
P<sub>angular,i</sub>(t) = τᵢ(t) · ω<sub>i</sub>(t)
</div>
<p>Power delivered by joint torque τ at angular velocity ω.
Positive = energy flowing from proximal to distal.</p>

<h3>6.2 Linear Power</h3>
<div class="eq">
P<sub>linear,i</sub>(t) = F<sub>i</sub>(t) · v<sub>i</sub>(t)
</div>
<p>Power delivered by net joint force at the joint's linear velocity.</p>

<h3>6.3 Angular Impulse</h3>
<div class="eq">
J<sub>angular,i</sub>(t) = ∫₀ᵗ τᵢ(s) ds
</div>

<h3>6.4 Angular Work</h3>
<div class="eq">
W<sub>angular,i</sub>(t) = ∫₀ᵗ τᵢ(s) · ωᵢ(s) ds = ∫₀ᵗ P<sub>angular,i</sub>(s) ds
</div>

<h3>6.5 Linear Impulse</h3>
<div class="eq">
J<sub>linear,i</sub>(t) = ∫₀ᵗ F<sub>i</sub>(s) ds
</div>

<h3>6.6 Joint Moments</h3>
<p>At each joint, three moment quantities are computed (proximal on distal):</p>
<div class="eq">
M<sub>applied</sub> = τ<sub>joint</sub>                    (motor/muscle torque)
M<sub>force</sub>   = r × F<sub>net</sub>                  (moment of net force)
M<sub>total</sub>   = M<sub>applied</sub> + M<sub>force</sub>  (total moment)
</div>
<p>where <span class="eq-inline">r</span> is the position vector from the
joint to the distal segment's center of mass.</p>

</body></html>
"""

# ---------------------------------------------------------------------------
# Topic registry
# ---------------------------------------------------------------------------

_TOPICS = {
    EquationTopic.MASS_MATRIX: ("Mass Matrix — Derivation", _MASS_MATRIX_HTML),
    EquationTopic.EQUATIONS_OF_MOTION: (
        "Equations of Motion — Full Reference",
        _EOM_HTML,
    ),
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def show_equations_popup(parent: QWidget | None, topic: EquationTopic) -> QDialog:
    """Show a non-modal equations popup.

    Pre: topic is a valid EquationTopic.
    Post: returns the QDialog instance (caller may discard).
    """
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QDialog, QTextBrowser, QVBoxLayout

    assert topic in _TOPICS, f"Unknown topic: {topic}"
    title, html = _TOPICS[topic]

    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.setMinimumSize(720, 600)
    dlg.setStyleSheet("QDialog { background: #1a1a28; }")

    layout = QVBoxLayout(dlg)
    layout.setContentsMargins(0, 0, 0, 0)

    browser = QTextBrowser()
    browser.setOpenExternalLinks(True)
    browser.setHtml(html)
    browser.setStyleSheet("QTextBrowser { background: #1a1a28; border: none; }")
    layout.addWidget(browser)

    dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    dlg.show()
    logger.info("Opened equations popup: %s", title)
    return dlg
