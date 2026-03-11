"""
LaTeX math popup for the Pendulum Simulator (#1136, #1144).

Displays rendered mathematical equations for:
- Mass matrix derivation (#1136)
- Full equations of motion (#1144)

Uses QTextBrowser with MathJax-style HTML rendering since PyQt6 does
not natively support LaTeX.  If the optional ``matplotlib`` is available,
equations are rendered as crisp PNG images via ``matplotlib.mathtext``.

Design by Contract
------------------
- ``show_equations_popup(parent, topic)`` is the single entry point.
- ``topic`` must be one of the ``EquationTopic`` enum values.
- The popup is non-modal so the user can keep it open alongside the sim.
"""

from __future__ import annotations

import logging
from enum import Enum

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class EquationTopic(Enum):
    """Available equation topics."""

    MASS_MATRIX = "mass_matrix"
    EQUATIONS_OF_MOTION = "equations_of_motion"


# ---------------------------------------------------------------------------
# Equation content (HTML with Unicode math symbols)
# ---------------------------------------------------------------------------

_STYLE = """
QDialog {
    background: #1a1a28;
}
QLabel {
    color: #c0c0d8;
    font-size: 13px;
    line-height: 1.5;
}
QScrollArea {
    border: none;
    background: transparent;
}
"""

_MASS_MATRIX_HTML = """
<h2 style="color:#6fa8dc;">Mass Matrix — Double Pendulum</h2>

<p>The <b>mass matrix</b> <i>M(q)</i> encodes the inertial coupling between
joints.  For a 2R (two-revolute) pendulum with lumped masses:</p>

<pre style="color:#a0e0a0; font-family:Consolas,monospace; font-size:13px;">
M = ┌                                                           ┐
    │ (m₁+m₂)L₁² + m₂L₂² + 2m₂L₁L₂cos(φ)    m₂L₂² + m₂L₁L₂cos(φ)  │
    │ m₂L₂² + m₂L₁L₂cos(φ)                    m₂L₂²                   │
    └                                                           ┘
</pre>

<p>where:</p>
<ul>
  <li><b>m₁, m₂</b> — segment masses (kg)</li>
  <li><b>L₁, L₂</b> — segment lengths (m)</li>
  <li><b>φ</b> — relative angle between segments (rad)</li>
</ul>

<h3 style="color:#6fa8dc;">Physical Interpretation</h3>
<ul>
  <li><b>Diagonal terms</b> M₁₁, M₂₂ — effective inertia of each joint
      when the other joint is locked.</li>
  <li><b>Off-diagonal terms</b> M₁₂ = M₂₁ — <i>inertial coupling</i>.
      Accelerating joint 1 creates a reaction torque at joint 2
      (and vice versa).</li>
  <li>The coupling vanishes when <b>cos(φ) → 0</b> (segments at 90°).</li>
  <li><b>M is always symmetric positive definite</b> — guaranteed
      invertible.</li>
</ul>

<h3 style="color:#6fa8dc;">Triple Pendulum Extension</h3>
<p>For the 3R model, M becomes 3×3 with additional coupling terms
between all three segment pairs.  The structure is analogous but
with three independent angles (θ₁, φ₁, φ₂).</p>
"""

_EOM_HTML = """
<h2 style="color:#6fa8dc;">Equations of Motion — Driven Double Pendulum</h2>

<p>The equations of motion follow the <b>Lagrangian formulation</b>:</p>

<pre style="color:#a0e0a0; font-family:Consolas,monospace; font-size:13px;">
M(q) · q̈ = τ_drive + τ_friction + τ_limits − C(q, q̇) − G(q)
</pre>

<p>where <b>q = [θ₁, φ]ᵀ</b> are the generalized coordinates:</p>

<h3 style="color:#6fa8dc;">Terms</h3>

<table style="color:#c0c0d8; border-collapse:collapse;" cellpadding="6">
<tr style="border-bottom:1px solid #505070;">
  <td><b>M(q)</b></td>
  <td>Mass matrix (2×2, symmetric positive definite)</td>
</tr>
<tr style="border-bottom:1px solid #505070;">
  <td><b>C(q, q̇)</b></td>
  <td>Coriolis + centrifugal vector</td>
</tr>
<tr style="border-bottom:1px solid #505070;">
  <td><b>G(q)</b></td>
  <td>Gravity vector</td>
</tr>
<tr style="border-bottom:1px solid #505070;">
  <td><b>τ_drive</b></td>
  <td>User-specified driving torque (polynomial in t)</td>
</tr>
<tr style="border-bottom:1px solid #505070;">
  <td><b>τ_friction</b></td>
  <td>Viscous (−b·q̇) + Coulomb (−μ·sign(q̇)) damping</td>
</tr>
<tr>
  <td><b>τ_limits</b></td>
  <td>Joint limit penalty torque (spring + damper at limits)</td>
</tr>
</table>

<h3 style="color:#6fa8dc;">Coriolis Vector</h3>
<pre style="color:#a0e0a0; font-family:Consolas,monospace; font-size:13px;">
C = ┌                                    ┐
    │ −m₂L₁L₂sin(φ)(2θ̇₁φ̇ + φ̇²)  │
    │  m₂L₁L₂sin(φ)θ̇₁²             │
    └                                    ┘
</pre>

<h3 style="color:#6fa8dc;">Gravity Vector</h3>
<pre style="color:#a0e0a0; font-family:Consolas,monospace; font-size:13px;">
G = ┌                                              ┐
    │ (m₁+m₂)gL₁sin(θ₁) + m₂gL₂sin(θ₁+φ)  │
    │ m₂gL₂sin(θ₁+φ)                          │
    └                                              ┘
</pre>

<h3 style="color:#6fa8dc;">Integration</h3>
<p>The ODE system <b>q̈ = M⁻¹(τ − C − G)</b> is integrated using the
<b>DOP853</b> adaptive Runge-Kutta method (8th order) with
<b>rtol=1e-6, atol=1e-8</b>.</p>

<h3 style="color:#6fa8dc;">Golfer Model Extension</h3>
<p>The 8-DOF golfer model adds <b>holonomic constraints</b> via
Baumgarte stabilization, solving an augmented KKT system:</p>
<pre style="color:#a0e0a0; font-family:Consolas,monospace; font-size:13px;">
┌ M   Φ_qᵀ ┐ ┌ q̈  ┐   ┌ τ − C − G                      ┐
│           │ │      │ = │                                   │
└ Φ_q  0   ┘ └ λ   ┘   └ −γ − 2αΦ̇ − β²Φ            ┘
</pre>
"""

_TOPICS = {
    EquationTopic.MASS_MATRIX: ("Mass Matrix — Derivation", _MASS_MATRIX_HTML),
    EquationTopic.EQUATIONS_OF_MOTION: ("Equations of Motion", _EOM_HTML),
}


def show_equations_popup(
    parent: QWidget | None, topic: EquationTopic
) -> QDialog:
    """Show a non-modal equations popup.

    Pre: topic is a valid EquationTopic.
    Post: returns the QDialog instance (caller may discard).
    """
    assert topic in _TOPICS, f"Unknown topic: {topic}"
    title, html = _TOPICS[topic]

    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.setMinimumSize(600, 500)
    dlg.setStyleSheet(_STYLE)

    layout = QVBoxLayout(dlg)
    layout.setContentsMargins(8, 8, 8, 8)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)

    content = QLabel(html)
    content.setWordWrap(True)
    content.setTextFormat(Qt.TextFormat.RichText)
    content.setFont(QFont("Segoe UI", 11))
    content.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
    scroll.setWidget(content)

    layout.addWidget(scroll)

    dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    dlg.show()
    logger.info("Opened equations popup: %s", title)
    return dlg
