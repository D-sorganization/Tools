/**
 * Putting vertical — TypeScript mirror of
 * `shared/python/swing_sim/putting` (epic #4125, H3).
 *
 * Same derivations, same constants, same fixed-step RK4 (dt = 2 ms),
 * so the vitest parity suite pins the Python reference putt
 * value-for-value (`tests/rate_of_closure/test_putting.py`).
 *
 * Physics summary (full derivations in the Python docstrings):
 * - Impact: 1-D COR impulse along the lofted face normal plus the 2/7
 *   rolling-cap tangential transfer -> launch speed, angle, backspin.
 * - Skid: sliding friction decelerates the ball and spins it up until
 *   v = omega r (pure roll at (5 v0 + 2 omega0 r) / 7).
 * - Green speed: the USGA stimpmeter (36 in ramp, 20 deg release,
 *   ~1.83 m/s release speed) inverts to mu_r = v^2 / (2 g S).
 * - Capture: the ball must fall half its diameter while crossing the
 *   hole mouth -> v_capture = R sqrt(g / 2r) ~= 0.82 m/s.
 */

export const GRAVITY_M_S2 = 9.80665;
export const GOLF_BALL_MASS_KG = 0.04593;
export const GOLF_BALL_RADIUS_M = 0.04267 / 2.0;
export const HOLE_RADIUS_M = 0.054;
export const DEFAULT_PUTTER_COR = 0.78;
export const DEFAULT_SLIDING_MU = 0.4;

const FOOT_M = 0.3048;
const ROLLING_CAP = 2.0 / 7.0;
const DT_S = 0.002;
const STOP_SPEED_MPS = 0.005;
const MAX_TIME_S = 60.0;

/** Stimpmeter release speed [m/s] — USGA ramp geometry derivation. */
export const STIMP_RELEASE_SPEED_MPS = Math.sqrt(
  (2.0 * GRAVITY_M_S2 * 0.762 * Math.sin((20.0 * Math.PI) / 180.0)) /
    (1.0 + 2.0 / 5.0 / (0.87 * 0.87)),
);

export interface PutterSpec {
  name: string;
  headMassKg: number;
  loftDeg: number;
  cor: number;
}

/** H3-local minimal putters (H1 club-library reconciliation note). */
export const MINIMAL_PUTTERS: PutterSpec[] = [
  { name: "Blade Putter", headMassKg: 0.35, loftDeg: 3.0, cor: DEFAULT_PUTTER_COR },
  { name: "Mallet Putter", headMassKg: 0.36, loftDeg: 3.0, cor: DEFAULT_PUTTER_COR },
];

export interface PuttLaunch {
  ballSpeedMps: number;
  launchAngleDeg: number;
  horizontalSpeedMps: number;
  /** Topspin positive; a struck putt starts negative (backspin). */
  spinRadS: number;
  effectiveLoftDeg: number;
}

/** Putter-ball impact (COR impulse + 2/7 tangential cap). */
export function strike(
  putter: PutterSpec,
  clubheadSpeedMps: number,
  shaftLeanDeg = 0.0,
): PuttLaunch {
  if (!(clubheadSpeedMps > 0 && clubheadSpeedMps <= 10)) {
    throw new Error("clubheadSpeedMps must be in (0, 10]");
  }
  const effectiveLoftDeg = putter.loftDeg + shaftLeanDeg;
  if (effectiveLoftDeg < -2 || effectiveLoftDeg > 15) {
    throw new Error("effective loft must stay in [-2, 15] deg");
  }
  const delta = (effectiveLoftDeg * Math.PI) / 180.0;
  const massRatio = putter.headMassKg / (putter.headMassKg + GOLF_BALL_MASS_KG);
  const transfer = (1.0 + putter.cor) * massRatio;
  const vNormal = transfer * clubheadSpeedMps * Math.cos(delta);
  const uTangential = clubheadSpeedMps * Math.sin(delta);
  const vTangential = ROLLING_CAP * uTangential;
  const spinRadS = (-(1.0 - ROLLING_CAP) * uTangential) / GOLF_BALL_RADIUS_M;
  const horizontal =
    vNormal * Math.cos(delta) - vTangential * Math.sin(delta);
  const vertical = vNormal * Math.sin(delta) + vTangential * Math.cos(delta);
  return {
    ballSpeedMps: Math.hypot(horizontal, vertical),
    launchAngleDeg: (Math.atan2(vertical, horizontal) * 180.0) / Math.PI,
    horizontalSpeedMps: horizontal,
    spinRadS,
    effectiveLoftDeg,
  };
}

/** Pendulum backstroke proxy: v = A sqrt(g / L). */
export function clubheadSpeedFromBackstroke(
  backstrokeM: number,
  putterLengthM = 0.889,
): number {
  if (!(backstrokeM > 0 && backstrokeM <= 1.5)) {
    throw new Error("backstrokeM must be in (0, 1.5]");
  }
  return backstrokeM * Math.sqrt(GRAVITY_M_S2 / putterLengthM);
}

/** Stimp [ft] -> rolling-resistance coefficient. */
export function stimpToRollingMu(stimpFt: number): number {
  if (!(stimpFt >= 3 && stimpFt <= 16)) {
    throw new Error("stimpFt must be in [3, 16]");
  }
  return (
    (STIMP_RELEASE_SPEED_MPS * STIMP_RELEASE_SPEED_MPS) /
    (2.0 * GRAVITY_M_S2 * stimpFt * FOOT_M)
  );
}

/** Geometric lip-capture bound: R sqrt(g / 2r) ~= 0.82 m/s. */
export function captureSpeedMps(): number {
  return HOLE_RADIUS_M * Math.sqrt(GRAVITY_M_S2 / (2.0 * GOLF_BALL_RADIUS_M));
}

export interface GreenConditions {
  stimpFt: number;
  gradePercent: number;
  /** Downhill direction, CCW from the putt line [deg]. */
  aspectDeg: number;
  muSlide?: number;
}

export interface PuttResult {
  pathXM: number[];
  pathYM: number[];
  speedsMps: number[];
  timesS: number[];
  skidEndIndex: number;
  skidDistanceM: number;
  totalDistanceM: number;
  timeS: number;
  breakM: number;
  holed: boolean;
  speedAtHoleMps: number | null;
  marginMps: number | null;
  missDistanceM: number | null;
}

type State = [number, number, number, number, number];

function derivative(
  state: State,
  sliding: boolean,
  muSlide: number,
  muRoll: number,
  gPar: [number, number],
): State {
  const [, , vx, vy] = state;
  const speed = Math.hypot(vx, vy);
  if (speed <= 0) return [0, 0, gPar[0], gPar[1], 0];
  const mu = sliding ? muSlide : muRoll;
  return [
    vx,
    vy,
    (-mu * GRAVITY_M_S2 * vx) / speed + gPar[0],
    (-mu * GRAVITY_M_S2 * vy) / speed + gPar[1],
    sliding ? 2.5 * muSlide * GRAVITY_M_S2 : 0,
  ];
}

function rk4Step(
  state: State,
  sliding: boolean,
  muSlide: number,
  muRoll: number,
  gPar: [number, number],
): State {
  const k1 = derivative(state, sliding, muSlide, muRoll, gPar);
  const mid1 = state.map((s, i) => s + 0.5 * DT_S * k1[i]) as State;
  const k2 = derivative(mid1, sliding, muSlide, muRoll, gPar);
  const mid2 = state.map((s, i) => s + 0.5 * DT_S * k2[i]) as State;
  const k3 = derivative(mid2, sliding, muSlide, muRoll, gPar);
  const end = state.map((s, i) => s + DT_S * k3[i]) as State;
  const k4 = derivative(end, sliding, muSlide, muRoll, gPar);
  return state.map(
    (s, i) => s + (DT_S / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]),
  ) as State;
}

/** Integrate one putt on a uniform sloped green (Python parity). */
export function simulatePutt(
  launch: PuttLaunch,
  green: GreenConditions,
  holeDistanceM: number,
): PuttResult {
  if (!(holeDistanceM >= 0.1 && holeDistanceM <= 40)) {
    throw new Error("holeDistanceM must be in [0.1, 40]");
  }
  if (!(launch.horizontalSpeedMps > 0)) {
    throw new Error("putt must start moving");
  }
  const muSlide = green.muSlide ?? DEFAULT_SLIDING_MU;
  const muRoll = stimpToRollingMu(green.stimpFt);
  const aspect = (green.aspectDeg * Math.PI) / 180.0;
  const grade = green.gradePercent / 100.0;
  const gPar: [number, number] = [
    GRAVITY_M_S2 * grade * Math.cos(aspect),
    GRAVITY_M_S2 * grade * Math.sin(aspect),
  ];
  const vCapture = captureSpeedMps();

  let state: State = [
    0,
    0,
    launch.horizontalSpeedMps,
    0,
    launch.spinRadS * GOLF_BALL_RADIUS_M,
  ];
  let sliding = state[4] < state[2];
  const xs = [0];
  const ys = [0];
  const speeds = [launch.horizontalSpeedMps];
  const times = [0];
  let distance = 0;
  let skidDistance = 0;
  let skidEndIndex = sliding ? -1 : 0;
  let holed = false;
  let speedAtHole: number | null = null;
  let time = 0;

  while (time < MAX_TIME_S) {
    const prev = state;
    state = rk4Step(state, sliding, muSlide, muRoll, gPar);
    time += DT_S;
    const step = Math.hypot(state[0] - prev[0], state[1] - prev[1]);
    distance += step;
    const speed = Math.hypot(state[2], state[3]);
    if (sliding) {
      skidDistance += step;
      if (state[4] >= speed) {
        sliding = false;
        skidEndIndex = xs.length;
      }
    }
    xs.push(state[0]);
    ys.push(state[1]);
    speeds.push(speed);
    times.push(time);
    const toHole = Math.hypot(state[0] - holeDistanceM, state[1]);
    if (toHole <= HOLE_RADIUS_M) {
      if (speedAtHole === null) speedAtHole = speed;
      if (speed <= vCapture) {
        holed = true;
        break;
      }
    }
    if (speed <= STOP_SPEED_MPS) break;
  }

  if (skidEndIndex < 0) skidEndIndex = xs.length - 1;
  let missDistance: number | null = null;
  let margin: number | null = null;
  if (holed && speedAtHole !== null) {
    margin = vCapture - speedAtHole;
  } else {
    missDistance = Math.hypot(
      xs[xs.length - 1] - holeDistanceM,
      ys[ys.length - 1],
    );
  }
  return {
    pathXM: xs,
    pathYM: ys,
    speedsMps: speeds,
    timesS: times,
    skidEndIndex,
    skidDistanceM: skidDistance,
    totalDistanceM: distance,
    timeS: time,
    breakM: ys[ys.length - 1],
    holed,
    speedAtHoleMps: speedAtHole,
    marginMps: margin,
    missDistanceM: missDistance,
  };
}
