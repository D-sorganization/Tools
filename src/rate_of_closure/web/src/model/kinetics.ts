/**
 * Swing kinetics for the web clone (#4125 H2) — TypeScript mirror of
 * `rate_of_closure/simulation/kinetics.py`.
 *
 * Per-sample inverse dynamics over the double-pendulum swing: net /
 * gravity / damping joint torques, joint powers (τ·ω), and joint
 * reaction-force magnitudes from Newton–Euler on each segment. The
 * math mirrors the Python module operation-for-operation and is
 * parity-pinned against the pytest-generated
 * `__fixtures__/kinetics_parity.json`.
 *
 * Sign convention (shared): positive torque acts counter-clockwise
 * about the swing-plane normal — the direction of increasing joint
 * angle. Forces are the proximal-on-distal reaction, gravity included.
 *
 * The 3D playback overlay is NOT mirrored here — deferred to the P7
 * WASM pass with the rest of the pose-level web scene (SPEC.md H2
 * deviation note); the plots section is the web presentation.
 */

import {
  golfDefaultParams,
  inPlaneGravity,
  simulatePendulum,
  type PendulumParams,
  type PendulumState,
  type SimulationInput,
} from "./simulation";

/** Clubhead point mass [kg] — shared golf-default head mass. */
export const CLUBHEAD_MASS_KG = 0.2;

/** Joint order (proximal to distal), matching the Python series. */
export const KINETIC_JOINT_NAMES = ["shoulder", "wrist"] as const;

export interface KineticsSeriesTs {
  tS: number[];
  /** Net (intersegmental) torque per joint, M·α + C. */
  shoulderTorqueNm: number[];
  wristTorqueNm: number[];
  /** Gravity torque per joint, -G. */
  shoulderGravityTorqueNm: number[];
  wristGravityTorqueNm: number[];
  /** Viscous damping torque per joint, -D. */
  shoulderDampingTorqueNm: number[];
  wristDampingTorqueNm: number[];
  /** Joint power, τ_net · ω (sums to dKE/dt). */
  shoulderPowerW: number[];
  wristPowerW: number[];
  /** Reaction-force magnitudes (Newton–Euler per segment). */
  shoulderForceN: number[];
  wristForceN: number[];
  /** Point-mass clubhead force estimate at the club tip. */
  clubheadForceN: number[];
}

interface EomTerms {
  c1: number;
  c2: number;
  g1: number;
  g2: number;
  d1: number;
  d2: number;
}

function eomTerms(
  p: PendulumParams,
  y: PendulumState,
  gInplane: [number, number],
): EomTerms {
  const [th1, th2, w1, w2] = y;
  const h = -p.m2 * p.l1 * p.lc2 * Math.sin(th2);
  const c1 = h * (2.0 * w1 * w2 + w2 * w2);
  const c2 = -h * w1 * w1;
  const [gx, gy] = gInplane;
  const t12 = th1 + th2;
  const a1 = p.m1 * p.lc1 + p.m2 * p.l1;
  const a2 = p.m2 * p.lc2;
  const g1 =
    -a1 * (gx * Math.cos(th1) + gy * Math.sin(th1)) -
    a2 * (gx * Math.cos(t12) + gy * Math.sin(t12));
  const g2 = -a2 * (gx * Math.cos(t12) + gy * Math.sin(t12));
  return { c1, c2, g1, g2, d1: p.d1 * y[2], d2: p.d2 * y[3] };
}

/** Central-difference gradient matching `np.gradient` (one-sided ends). */
export function gradient(values: number[], dt: number): number[] {
  const n = values.length;
  const out = new Array<number>(n);
  if (n === 1) {
    out[0] = 0;
    return out;
  }
  out[0] = (values[1] - values[0]) / dt;
  out[n - 1] = (values[n - 1] - values[n - 2]) / dt;
  for (let i = 1; i < n - 1; i += 1) {
    out[i] = (values[i + 1] - values[i - 1]) / (2.0 * dt);
  }
  return out;
}

/**
 * Kinetics over a sampled joint-state trajectory (uniform grid).
 * Mirrors `kinetics.inverse_dynamics` + `_reaction_forces`.
 */
export function computeKinetics(
  p: PendulumParams,
  states: PendulumState[],
  gInplane: [number, number],
  dt: number,
  clubheadMassKg: number = CLUBHEAD_MASS_KG,
): KineticsSeriesTs {
  if (states.length < 3) throw new Error("kinetics needs at least 3 samples");
  if (!(dt > 0)) throw new Error("dt must be > 0");
  const n = states.length;
  const alpha1 = gradient(states.map((s) => s[2]), dt);
  const alpha2 = gradient(states.map((s) => s[3]), dt);
  const [gx, gy] = gInplane;

  const out: KineticsSeriesTs = {
    tS: states.map((_, i) => i * dt),
    shoulderTorqueNm: [],
    wristTorqueNm: [],
    shoulderGravityTorqueNm: [],
    wristGravityTorqueNm: [],
    shoulderDampingTorqueNm: [],
    wristDampingTorqueNm: [],
    shoulderPowerW: [],
    wristPowerW: [],
    shoulderForceN: [],
    wristForceN: [],
    clubheadForceN: [],
  };
  for (let i = 0; i < n; i += 1) {
    const y = states[i];
    const [th1, th2, w1, w2] = y;
    const { c1, c2, g1, g2, d1, d2 } = eomTerms(p, y, gInplane);
    const cos2 = Math.cos(th2);
    const m11 =
      p.i1 + p.i2 + p.m2 * p.l1 * p.l1 + 2.0 * p.m2 * p.l1 * p.lc2 * cos2;
    const m12 = p.i2 + p.m2 * p.l1 * p.lc2 * cos2;
    const net1 = m11 * alpha1[i] + m12 * alpha2[i] + c1;
    const net2 = m12 * alpha1[i] + p.i2 * alpha2[i] + c2;
    out.shoulderTorqueNm.push(net1);
    out.wristTorqueNm.push(net2);
    out.shoulderGravityTorqueNm.push(-g1);
    out.wristGravityTorqueNm.push(-g2);
    out.shoulderDampingTorqueNm.push(-d1);
    out.wristDampingTorqueNm.push(-d2);
    out.shoulderPowerW.push(net1 * w1);
    out.wristPowerW.push(net2 * w2);

    // Newton–Euler reaction forces (in-plane 2-vectors, magnitudes).
    const phi12 = th1 + th2;
    const wd1 = alpha1[i];
    const wd12 = alpha1[i] + alpha2[i];
    const w12 = w1 + w2;
    const accAt = (
      r: number,
      phi: number,
      phid: number,
      phidd: number,
    ): [number, number] => [
      r * (phidd * Math.cos(phi) - phid * phid * Math.sin(phi)),
      r * (phidd * Math.sin(phi) + phid * phid * Math.cos(phi)),
    ];
    const aArm = accAt(p.lc1, th1, w1, wd1);
    const aElbow = accAt(p.l1, th1, w1, wd1);
    const aClubRel = accAt(p.lc2, phi12, w12, wd12);
    const aTipRel = accAt(p.l2, phi12, w12, wd12);
    const fWrist: [number, number] = [
      p.m2 * (aElbow[0] + aClubRel[0] - gx),
      p.m2 * (aElbow[1] + aClubRel[1] - gy),
    ];
    const fShoulder: [number, number] = [
      p.m1 * (aArm[0] - gx) + fWrist[0],
      p.m1 * (aArm[1] - gy) + fWrist[1],
    ];
    const fHead: [number, number] = [
      clubheadMassKg * (aElbow[0] + aTipRel[0] - gx),
      clubheadMassKg * (aElbow[1] + aTipRel[1] - gy),
    ];
    out.shoulderForceN.push(Math.hypot(fShoulder[0], fShoulder[1]));
    out.wristForceN.push(Math.hypot(fWrist[0], fWrist[1]));
    out.clubheadForceN.push(Math.hypot(fHead[0], fHead[1]));
  }
  return out;
}

const rad = (d: number): number => (d * Math.PI) / 180.0;

let cacheKey = "";
let cacheValue: KineticsSeriesTs | null = null;

/**
 * Kinetics for a simulation input, or null for sources without joint
 * states (manual). Re-simulates the deterministic pendulum trajectory
 * (same grid as `swingSamples`) and memoizes on the input fields that
 * affect it.
 */
export function kineticsForInput(
  input: SimulationInput,
): KineticsSeriesTs | null {
  if (input.sourceKind !== "double_pendulum") return null;
  const key = JSON.stringify([
    input.planeYawDeg,
    input.planeSideTiltDeg,
    input.planeForwardTiltDeg,
    input.swingDurationS,
  ]);
  if (key === cacheKey && cacheValue) return cacheValue;
  const p = golfDefaultParams();
  const g = inPlaneGravity(
    rad(input.planeYawDeg),
    rad(input.planeSideTiltDeg),
    rad(input.planeForwardTiltDeg),
  );
  const dt = 1e-3;
  const nSteps = Math.round(input.swingDurationS / dt);
  const states = simulatePendulum(p, [-Math.PI / 2, 0, 0, 0], g, dt, nSteps);
  cacheKey = key;
  cacheValue = computeKinetics(p, states, g, dt);
  return cacheValue;
}
