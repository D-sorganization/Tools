/**
 * Sectioned full-model derivations (#4120 V4) — TypeScript mirror of
 * `src/rate_of_closure/derivation_models.py` (+ the per-domain content
 * modules). Section keys and conditional rendering are parity-tested
 * against the Python side; KaTeX-flavoured LaTeX.
 */

import { derivationSteps, type DerivationStep } from "./derivation";
import type { ImpactScenario } from "./impact";

export interface DerivationConfig {
  /** Registry key of the active flight model. */
  flightModel: string;
  /** "manual" | "double_pendulum" | "triple_pendulum". */
  swingSource: string;
  /** Whether the gear-effect step is included (session default: on). */
  gearEffect: boolean;
  /** Live (yaw, side, forward) swing-plane tilts in degrees. */
  planeTiltsDeg: [number, number, number];
}

export const DEFAULT_DERIVATION_CONFIG: DerivationConfig = {
  flightModel: "waterloo_penner",
  swingSource: "manual",
  gearEffect: true,
  planeTiltsDeg: [0, -45, 0],
};

export interface DerivationSection {
  key: string;
  title: string;
  intro: string;
  steps: DerivationStep[];
}

// Vendored constants (mirror swing_sim.impact.constants).
const DRIVER_COR = 0.83;
const GOLF_BALL_MASS_G = 45.93;
const DRIVER_MASS_G = 200;
const DRIVER_MOI_G_CM2 = 450;
const DRIVER_CG_DEPTH_MM = 35;

/** Coefficient metadata for the literature models (registry mirror). */
const FLIGHT_MODEL_META: Record<
  string,
  { name: string; description: string; reference: string; law: string; values: string }
> = {
  waterloo_penner: {
    name: "Waterloo/Penner",
    description: "Waterloo quadratic Cd with Penner spin-ratio lift fit",
    reference: "Penner (2003); McPhee et al. (Waterloo)",
    law:
      "C_d = c_{d0} + c_{d1} s + c_{d2} s^2,\\qquad " +
      "C_l = \\min(C_{l,max},\\ c_{l1}\\,s^{c_{l2}}),\\qquad " +
      "s = \\frac{R\\,\\omega}{v}",
    values:
      "c_{d} = (0.21,\\ 0.05,\\ 0.02),\\ c_{l} = (0.70,\\ 0.645),\\ " +
      "\\text{spin ratio } s \\text{ drives both}",
  },
  macdonald_hanzely: {
    name: "MacDonald-Hanzely",
    description: "Analytic model with exponential spin decay",
    reference: "MacDonald & Hanzely (1991)",
    law: "C_d = const,\\qquad C_l \\propto s,\\qquad \\omega(t) = \\omega_0\\,e^{-t/\\tau}",
    values: "\\text{exponential spin decay sets late lift}",
  },
  nathan: {
    name: "Nathan",
    description: "Constant Cd/Cl model with spin decay",
    reference: "Nathan et al. (2018)",
    law: "C_d = const,\\qquad C_l = const,\\qquad \\omega(t) = \\omega_0\\,e^{-\\lambda t}",
    values: "C_d = 0.22,\\ C_l = 0.24,\\ \\lambda = 0.03\\ \\text{s}^{-1}",
  },
  ballantyne: {
    name: "Ballantyne",
    description: "Constant Cd/Cl model for steady spin",
    reference: "Ballantyne et al. (2012)",
    law: "C_d = const,\\qquad C_l = const,\\qquad \\omega(t) = \\omega_0\\,e^{-\\lambda t}",
    values: "C_d = 0.20,\\ C_l = 0.18,\\ \\lambda = 0.02\\ \\text{s}^{-1}",
  },
  jcole: {
    name: "J. Cole",
    description: "Constant Cd/Cl model with moderate decay",
    reference: "Cole (2016)",
    law: "C_d = const,\\qquad C_l = const,\\qquad \\omega(t) = \\omega_0\\,e^{-\\lambda t}",
    values: "C_d = 0.23,\\ C_l = 0.22,\\ \\lambda = 0.04\\ \\text{s}^{-1}",
  },
  rospie_dl: {
    name: "Rospie DL",
    description: "Constant Cd/Cl model tuned for driver launch",
    reference: "Rospie & Layton (2014)",
    law: "C_d = const,\\qquad C_l = const,\\qquad \\omega(t) = \\omega_0\\,e^{-\\lambda t}",
    values: "C_d = 0.21,\\ C_l = 0.19,\\ \\lambda = 0.03\\ \\text{s}^{-1}",
  },
  charry_l3: {
    name: "Charry L3",
    description: "Constant Cd/Cl model with higher drag",
    reference: "Charry et al. (2017)",
    law: "C_d = const,\\qquad C_l = const,\\qquad \\omega(t) = \\omega_0\\,e^{-\\lambda t}",
    values: "C_d = 0.24,\\ C_l = 0.21,\\ \\lambda = 0.05\\ \\text{s}^{-1}",
  },
};

function impactSteps(
  scenario: ImpactScenario,
  gearEffect: boolean,
): DerivationStep[] {
  const toe = scenario.impactOffsetToeMm;
  const high = scenario.impactOffsetHighMm;
  const offset = Math.hypot(toe, high);
  const steps: DerivationStep[] = [
    {
      title: "Impulse-Momentum Exchange With COR",
      latex:
        "J = \\frac{(1 + e)\\,v_{rel}}{\\frac{1}{m_{ball}} + " +
        "\\frac{1}{m_{eff}}},\\qquad v_{ball} = \\frac{J}{m_{ball}}",
      values:
        `e = ${DRIVER_COR.toFixed(2)}\\ \\text{(driver COR cap)},\\ ` +
        `m_{ball} = ${GOLF_BALL_MASS_G.toFixed(1)}\\ \\text{g},\\ ` +
        `m_{club} = ${DRIVER_MASS_G.toFixed(0)}\\ \\text{g}`,
      narrative:
        "The ball leaves with the momentum delivered by one normal " +
        "impulse J over the ~450 µs contact. The coefficient of " +
        "restitution e scales the separation speed; the club side " +
        "enters through its effective mass, not its full mass " +
        "(swing_sim.impact.models rigid-body COR model).",
    },
    {
      title: "Effective Mass — the MOI-Tensor Triple Product",
      latex:
        "\\frac{1}{m_{eff}} = \\frac{1}{m_{club}} + " +
        "(\\vec{r} \\times \\hat{n})^T I^{-1} (\\vec{r} \\times \\hat{n})",
      values:
        `|\\vec{r}| = ${offset.toFixed(1)}\\ \\text{mm}\\ ` +
        `\\text{(toe ${toe >= 0 ? "+" : ""}${toe.toFixed(0)}, high ` +
        `${high >= 0 ? "+" : ""}${high.toFixed(0)})},\\ ` +
        `I_{scalar} = ${DRIVER_MOI_G_CM2}\\ \\text{g}\\,\\text{cm}^2` +
        "\\ \\text{fallback: } \\frac{1}{m} + \\frac{|\\vec{r}|^2}{I}",
      narrative:
        "An off-center strike spends part of the impulse twisting the " +
        "head: the exact club-side denominator is the triple product " +
        "(r × n)ᵀ I⁻¹ (r × n) with the full 3×3 MOI tensor. A diagonal " +
        "tensor I·eye(3) reproduces the scalar fallback 1/m + |r|²/I " +
        "exactly because r lies in the face plane " +
        "(derivation in swing_sim.impact.models docstring).",
    },
    {
      title: "Friction Spin — the 2/7 Rolling Cap",
      latex:
        "J_f = \\min\\!\\left(\\mu J,\\ \\tfrac{2}{7}\\,m_{ball}\\," +
        "v_t\\right),\\qquad \\tfrac{2}{7} = \\frac{1}{1 + \\frac{5}{2}}",
      values:
        "I_{sphere} = \\tfrac{2}{5} m R^2 \\Rightarrow " +
        "J_f\\left(\\tfrac{1}{m} + \\tfrac{R^2}{I}\\right) = v_t " +
        "\\Rightarrow J_f = \\tfrac{2}{7}\\,m\\,v_t",
      narrative:
        "Friction converts the tangential approach speed into spin only " +
        "until the contact point stops sliding (rolling without slip). " +
        "For a uniform solid sphere that caps the friction impulse at " +
        "(2/7)·m·v_t — beyond it no more spin is available (Cross 2002; " +
        "SPHERE_ROLLING_CAP_FACTOR in swing_sim.impact.models). The " +
        "physical spin axis is t × n — the sign fix documented in the " +
        "port.",
    },
    {
      title: "Spin Loft and the D-Plane",
      latex:
        "\\text{spin loft} = \\arccos(\\hat{v} \\cdot \\hat{n})," +
        "\\qquad \\hat{a}_{spin} = \\widehat{\\hat{v} \\times \\hat{n}}",
      values:
        "\\hat{v} = (\\cos AoA \\cos path,\\ \\sin AoA,\\ " +
        "\\cos AoA \\sin path),\\ \\hat{n} = " +
        "(\\cos loft \\cos face,\\ \\sin loft,\\ \\cos loft \\sin face)",
      narrative:
        "The D-plane is spanned by the club-path vector and the " +
        "delivered face normal: the ball launches close to the normal " +
        "and spins about the plane's normal v̂ × n̂, so the " +
        "face-minus-path difference tilts the spin axis " +
        "(swing_sim.impact.delivery; Jorgensen; TrackMan D-plane " +
        "literature).",
    },
  ];
  if (gearEffect) {
    steps.push({
      title: "Gear Effect — Head Recoil Times CG Depth",
      latex:
        "\\Delta\\vec{\\omega}_{head} = I^{-1}\\left(\\vec{r} \\times " +
        "(-J\\hat{n})\\right),\\qquad \\vec{v}_{surf} = " +
        "\\tfrac{1}{2}\\,\\Delta\\vec{\\omega}_{head} \\times \\vec{r}",
      values:
        "\\vec{r} = \\vec{r}_{plane} + d\\,\\hat{n},\\quad " +
        `d = ${DRIVER_CG_DEPTH_MM}\\ \\text{mm (driver CG depth)}`,
      narrative:
        "The off-center impulse makes the head recoil in rotation; " +
        "because the CG sits a depth d behind the face, the rotating " +
        "face sweeps tangentially under the ball (time-averaged at half " +
        "the final recoil). Friction gears the ball against that moving " +
        "surface — toe hits gain draw spin, high hits lose backspin — " +
        "capped by the same 2/7 rolling limit " +
        "(swing_sim.impact.gear_effect derivation).",
    });
  }
  return steps;
}

function flightSteps(flightModel: string): DerivationStep[] {
  const meta =
    FLIGHT_MODEL_META[flightModel] ?? FLIGHT_MODEL_META.waterloo_penner;
  return [
    {
      title: "Equations of Motion — Drag, Lift, Gravity",
      latex:
        "m\\dot{\\vec{v}} = -\\tfrac{1}{2}\\rho A C_d " +
        "|\\vec{v}|\\,\\vec{v} + \\tfrac{1}{2}\\rho A C_l " +
        "|\\vec{v}|^2\\,(\\hat{\\omega} \\times \\hat{v}) + m\\vec{g}",
      values:
        "A = \\pi R^2,\\ \\rho \\approx 1.225\\ \\text{kg/m}^3,\\ " +
        "\\vec{g} = (0,\\ -9.81,\\ 0)\\ \\text{m/s}^2",
      narrative:
        "Three forces act in flight: drag opposing the velocity, the " +
        "Magnus lift perpendicular to both the spin axis and the " +
        "velocity (backspin lifts, tilted spin curves the shot " +
        "sideways), and gravity. The trajectory integrates this ODE to " +
        "a terminal ground event (swing_sim.flight.models base loop).",
    },
    {
      title: `Active Model Coefficient Law — ${meta.name}`,
      latex: meta.law,
      values: meta.values,
      narrative:
        `${meta.description}. The literature flight models differ ` +
        "mainly in how the drag and lift coefficients depend on the " +
        `spin ratio s = Rω/v. Citation: ${meta.reference} ` +
        "(swing_sim.flight registry metadata).",
    },
    {
      title: "Spin Decay and the Terminal Ground Event",
      latex:
        "\\omega(t) = \\omega_0\\,e^{-\\lambda t},\\qquad " +
        "y(t^*) = 0 \\Rightarrow \\text{carry},\\ \\text{apex},\\ " +
        "\\text{landing angle}",
      values:
        "\\text{carry} = x(t^*),\\ \\text{lateral} = z(t^*),\\ " +
        "\\text{landing} = \\arctan\\!\\frac{-v_y(t^*)}" +
        "{\\sqrt{v_x^2 + v_z^2}}",
      narrative:
        "Aerodynamic torque bleeds spin during flight (modeled as an " +
        "exponential decay in the spin-decay model families), reducing " +
        "late lift. Integration stops at the ground event; the reported " +
        "carry, apex, flight time, landing angle, and lateral offset " +
        "are read off that terminal state (swing_sim.flight metrics).",
    },
  ];
}

/** In-plane gravity from the three tilt angles (reference.py mirror). */
export function inPlaneGravity(
  yawDeg: number,
  sideDeg: number,
  fwdDeg: number,
  g = 9.81,
): [number, number] {
  const rad = Math.PI / 180;
  const [cy, sy] = [Math.cos(yawDeg * rad), Math.sin(yawDeg * rad)];
  const [cs, ss] = [Math.cos(sideDeg * rad), Math.sin(sideDeg * rad)];
  const [cf, sf] = [Math.cos(fwdDeg * rad), Math.sin(fwdDeg * rad)];
  // R = Rz(yaw) Rx(side) Ry(fwd); world gravity (0, 0, -g) dotted with
  // columns 0 (in-plane horizontal) and 2 (in-plane up).
  const col0 = [cy * cf + -sy * ss * -sf, sy * cf + cy * ss * -sf, cs * -sf];
  const col2 = [cy * sf - sy * ss * cf, sy * sf + cy * ss * cf, cs * cf];
  return [-g * col0[2], -g * col2[2]];
}

function swingSteps(
  swingSource: string,
  tilts: [number, number, number],
): DerivationStep[] {
  const [yaw, side, fwd] = tilts;
  const [gx, gy] = inPlaneGravity(yaw, side, fwd);
  const steps: DerivationStep[] = [
    {
      title: "Lagrangian Equations of Motion",
      latex:
        "M(\\theta)\\,\\ddot{\\theta} + C(\\theta, \\dot{\\theta}) + " +
        "G(\\theta) + D\\,\\dot{\\theta} = 0",
      values:
        "\\theta = (\\theta_1, \\theta_2):\\ \\text{arm and club links " +
        "in the inclined swing plane}",
      narrative:
        "The double-pendulum swing model treats arms and club as two " +
        "rigid links in an inclined plane. Its equations of motion " +
        "follow from the Lagrangian: a configuration-dependent mass " +
        "matrix, velocity-dependent Coriolis terms, gravity, and " +
        "viscous damping (swing_sim.reference / rust swing-core).",
    },
    {
      title: "Mass Matrix",
      latex:
        "M = \\begin{bmatrix} I_1 + I_2 + m_2 l_1^2 + " +
        "2 m_2 l_1 l_{c2} \\cos\\theta_2 & " +
        "I_2 + m_2 l_1 l_{c2} \\cos\\theta_2 \\\\ " +
        "I_2 + m_2 l_1 l_{c2} \\cos\\theta_2 & I_2 \\end{bmatrix}",
      values:
        "\\det M > 0\\ \\text{enforced (singular mass matrix rejected " +
        "by contract)}",
      narrative:
        "The symmetric 2×2 inertia matrix couples the links through the " +
        "wrist angle θ₂: a straighter wrist (cos θ₂ → 1) maximizes the " +
        "coupling. The integrator inverts M each step, with a " +
        "determinant contract guarding singular configurations " +
        "(swing_sim.reference.mass_matrix).",
    },
    {
      title: "Coriolis and Centripetal Terms",
      latex:
        "h = -m_2 l_1 l_{c2} \\sin\\theta_2:\\quad " +
        "C_1 = h\\,(2\\dot{\\theta}_1\\dot{\\theta}_2 + " +
        "\\dot{\\theta}_2^2),\\quad C_2 = -h\\,\\dot{\\theta}_1^2",
      values:
        "\\text{late release: } \\dot{\\theta}_1^2 \\text{ slings the " +
        "club through } C_2",
      narrative:
        "The velocity-dependent forces of the rotating links: the " +
        "centripetal −h·ω₁² term is the physics of the release — arm " +
        "rotation slings the club link outward without any wrist torque " +
        "(swing_sim.reference.coriolis_vector).",
    },
    {
      title: "Plane-Tilt Gravity Projection",
      latex:
        "R = R_z(yaw)\\,R_x(side)\\,R_y(fwd),\\qquad " +
        "\\vec{g}_{plane} = \\left(\\vec{g}_{world} \\cdot " +
        "\\hat{e}_1,\\ \\vec{g}_{world} \\cdot \\hat{e}_3\\right)",
      values:
        `(${yaw.toFixed(0)}^\\circ,\\ ${side.toFixed(0)}^\\circ,\\ ` +
        `${fwd.toFixed(0)}^\\circ) \\Rightarrow \\vec{g}_{plane} = ` +
        `(${gx >= 0 ? "+" : ""}${gx.toFixed(2)},\\ ` +
        `${gy >= 0 ? "+" : ""}${gy.toFixed(2)})\\ \\text{m/s}^2`,
      narrative:
        "The swing plane is oriented by yaw, side tilt, and forward " +
        "tilt; world gravity is projected onto the plane's in-plane " +
        "axes and the EOM consumes that 2-vector directly — a steeper " +
        "plane feels more in-plane gravity. The numbers substitute the " +
        "live tilts (swing_sim.reference.in_plane_gravity).",
    },
  ];
  if (swingSource === "triple_pendulum") {
    steps.push({
      title: "Triple-Pendulum Extension",
      latex:
        "M(\\theta)\\,\\ddot{\\theta} + C(\\theta, \\dot{\\theta}) + " +
        "G(\\theta) = 0,\\qquad \\theta \\in \\mathbb{R}^3",
      values:
        "\\text{links: torso} \\to \\text{arms} \\to \\text{club " +
        "(planar, same formalism)}",
      narrative:
        "The triple pendulum adds a torso link ahead of arms and club, " +
        "solved with the same mass-matrix formalism in the planar " +
        "frame — a 3×3 M(θ) assembled from the link inertias and " +
        "solved each step (rate_of_closure.simulation.sources).",
    });
  }
  return steps;
}

/** Every derivation section active under `config` (Python parity). */
export function derivationSections(
  scenario: ImpactScenario,
  config: DerivationConfig = DEFAULT_DERIVATION_CONFIG,
): DerivationSection[] {
  const sections: DerivationSection[] = [
    {
      key: "closure",
      title: "Closure Chain — Impact-Point Kinematics",
      intro:
        "The original derivation: from the frame conventions to the " +
        "reported impact-point deviations and closure metrics, with " +
        "the live scenario substituted.",
      steps: derivationSteps(scenario),
    },
    {
      key: "impact",
      title: "Impact Model — Impulse-Momentum With COR",
      intro:
        "How ball speed and spin come out of the strike: the " +
        "rigid-body impulse solve of swing_sim.impact, including the " +
        "MOI-tensor effective mass and the friction spin cap.",
      steps: impactSteps(scenario, config.gearEffect),
    },
    {
      key: "flight",
      title: "Ball Flight — Aerodynamic Integration",
      intro:
        "The trajectory ODE and the selected literature model's " +
        "coefficient law with its citation.",
      steps: flightSteps(config.flightModel),
    },
  ];
  if (
    config.swingSource === "double_pendulum" ||
    config.swingSource === "triple_pendulum"
  ) {
    sections.push({
      key: "swing",
      title: "Swing Model — Pendulum Dynamics",
      intro:
        "The pendulum swing source generating the delivery: Lagrangian " +
        "equations of motion in the tilted swing plane, with the live " +
        "plane tilts substituted.",
      steps: swingSteps(config.swingSource, config.planeTiltsDeg),
    });
  }
  return sections;
}
