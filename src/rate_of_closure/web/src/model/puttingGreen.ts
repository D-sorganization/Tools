/**
 * Green surface heightfield, 2-D surface roll, and hole capture —
 * TypeScript mirror of `shared/python/swing_sim/putting/surface.py`
 * and the surface half of `putting/green.py` (#4800 P2).
 *
 * Same constants, same fixed-step RK4 (dt = 2 ms), same bilinear
 * heightfield and capture derivations, so the vitest parity suite
 * (`puttingGreen.test.ts`) pins the Python reference putts
 * value-for-value.
 *
 * Physics summary (full derivations in the Python docstrings):
 * - Surface: parametric plane (grade + aspect) or a regular square
 *   grid with bilinear interpolation; in-plane gravity is `-g grad h`
 *   (small-slope). Outside the grid hull the surface continues flat.
 * - Roll: skid (sliding friction, spin-up) then pure roll (stimp-
 *   derived rolling resistance), integrated with classic RK4.
 * - Capture: effective hole radius shrinking with approach speed,
 *   `R_eff(v) = R sqrt(1 - (v / v_capture)^2)` (Holmes, Am. J. Phys.
 *   59, 129-136, 1991; Penner, Can. J. Phys. 80, 83-96, 2002), with
 *   the geometric bound `v_capture = R sqrt(g / 2r) ~= 0.82 m/s`
 *   pinned as the limiting case. The legacy `simulatePutt` keeps the
 *   historic speed-threshold capture, bit-identical to pre-#4800.
 *
 * The versioned fail-closed JSON wire `swing_sim.green_surface/1`
 * lives in `puttingGreenWire.ts`.
 */

export const GRAVITY_M_S2 = 9.80665;
export const GOLF_BALL_RADIUS_M = 0.04267 / 2.0;
export const HOLE_RADIUS_M = 0.054;
export const DEFAULT_SLIDING_MU = 0.4;

const FOOT_M = 0.3048;
const DT_S = 0.002;
const STOP_SPEED_MPS = 0.005;
const MAX_TIME_S = 60.0;
const MAX_GRID_NODES = 2048;
const MAX_LOCAL_GRADE = 0.25;

/** Stimpmeter release speed [m/s] — USGA ramp geometry derivation. */
export const STIMP_RELEASE_SPEED_MPS = Math.sqrt(
  (2.0 * GRAVITY_M_S2 * 0.762 * Math.sin((20.0 * Math.PI) / 180.0)) /
    (1.0 + 2.0 / 5.0 / (0.87 * 0.87)),
);

export interface PuttLaunch {
  ballSpeedMps: number;
  launchAngleDeg: number;
  horizontalSpeedMps: number;
  /** Topspin positive; a struck putt starts negative (backspin). */
  spinRadS: number;
  effectiveLoftDeg: number;
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

/** Uniform-slope plane — the parametric heightfield. */
export interface PlanarGreenSurface {
  kind: "planar";
  gradePercent: number;
  aspectDeg: number;
}

/** Regular square-grid heightfield (rows index y, columns index x). */
export interface GridGreenSurface {
  kind: "grid";
  originM: [number, number];
  spacingM: number;
  heightsM: number[][];
}

export type GreenSurface = PlanarGreenSurface | GridGreenSurface;

/** Hole-capture models (see module docs). */
export type CaptureModel = "effective_radius" | "speed_threshold";

/** Stimp [ft] -> rolling-resistance coefficient. */
export function stimpToRollingMu(stimpFt: number): number {
  if (!Number.isFinite(stimpFt) || !(stimpFt >= 3 && stimpFt <= 16)) {
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

/** Effective capture radius R_eff(v) = R sqrt(1 - (v/vc)^2). */
export function effectiveHoleRadiusM(speedMps: number): number {
  if (!Number.isFinite(speedMps) || speedMps < 0) {
    throw new Error("speedMps must be finite and non-negative");
  }
  const ratio = speedMps / captureSpeedMps();
  if (ratio >= 1) return 0;
  return HOLE_RADIUS_M * Math.sqrt(1.0 - ratio * ratio);
}

/** Validated parametric plane surface. */
export function planarSurface(
  gradePercent: number,
  aspectDeg: number,
): PlanarGreenSurface {
  if (!Number.isFinite(gradePercent) || gradePercent < 0 || gradePercent > 10) {
    throw new Error("gradePercent must be in [0, 10]");
  }
  if (!Number.isFinite(aspectDeg) || aspectDeg < -360 || aspectDeg > 360) {
    throw new Error("aspectDeg must be in [-360, 360]");
  }
  return { kind: "planar", gradePercent, aspectDeg };
}

/** Validated grid-heightfield surface. */
export function gridSurface(
  originM: [number, number],
  spacingM: number,
  heightsM: number[][],
): GridGreenSurface {
  if (!Number.isFinite(originM[0]) || !Number.isFinite(originM[1])) {
    throw new Error("originM must be finite");
  }
  if (!Number.isFinite(spacingM) || spacingM < 0.01 || spacingM > 100) {
    throw new Error("spacingM must be in [0.01, 100]");
  }
  if (heightsM.length < 2 || heightsM.length > MAX_GRID_NODES) {
    throw new Error("need 2..2048 rows");
  }
  const width = heightsM[0].length;
  if (width < 2 || width > MAX_GRID_NODES) {
    throw new Error("need 2..2048 columns");
  }
  const limit = MAX_LOCAL_GRADE * spacingM;
  for (let j = 0; j < heightsM.length; j++) {
    const row = heightsM[j];
    if (row.length !== width) throw new Error("heightsM must be rectangular");
    for (let i = 0; i < width; i++) {
      if (typeof row[i] !== "number" || !Number.isFinite(row[i])) {
        throw new Error("heights must be finite numbers");
      }
      if (i + 1 < width && Math.abs(row[i + 1] - row[i]) > limit) {
        throw new Error("local grade exceeds 25 percent");
      }
      if (
        j + 1 < heightsM.length &&
        Math.abs(heightsM[j + 1][i] - row[i]) > limit
      ) {
        throw new Error("local grade exceeds 25 percent");
      }
    }
  }
  return { kind: "grid", originM: [originM[0], originM[1]], spacingM, heightsM };
}

function gridCell(
  surface: GridGreenSurface,
  xM: number,
  yM: number,
): [number, number, number, number] {
  const nx = surface.heightsM[0].length;
  const ny = surface.heightsM.length;
  let u = (xM - surface.originM[0]) / surface.spacingM;
  let v = (yM - surface.originM[1]) / surface.spacingM;
  u = Math.min(Math.max(u, 0), nx - 1);
  v = Math.min(Math.max(v, 0), ny - 1);
  const i = Math.min(Math.floor(u), nx - 2);
  const j = Math.min(Math.floor(v), ny - 2);
  return [i, j, u - i, v - j];
}

function gridInside(surface: GridGreenSurface, xM: number, yM: number): boolean {
  const nx = surface.heightsM[0].length;
  const ny = surface.heightsM.length;
  return (
    xM >= surface.originM[0] &&
    xM <= surface.originM[0] + (nx - 1) * surface.spacingM &&
    yM >= surface.originM[1] &&
    yM <= surface.originM[1] + (ny - 1) * surface.spacingM
  );
}

/** Surface elevation [m] (bilinear on grids, edge-clamped outside). */
export function surfaceHeightM(
  surface: GreenSurface,
  xM: number,
  yM: number,
): number {
  if (surface.kind === "planar") {
    const aspect = (surface.aspectDeg * Math.PI) / 180.0;
    const grade = surface.gradePercent / 100.0;
    return -grade * (xM * Math.cos(aspect) + yM * Math.sin(aspect));
  }
  const [i, j, tx, ty] = gridCell(surface, xM, yM);
  const h00 = surface.heightsM[j][i];
  const h10 = surface.heightsM[j][i + 1];
  const h01 = surface.heightsM[j + 1][i];
  const h11 = surface.heightsM[j + 1][i + 1];
  const top = h00 * (1.0 - tx) + h10 * tx;
  const bottom = h01 * (1.0 - tx) + h11 * tx;
  return top * (1.0 - ty) + bottom * ty;
}

/** In-plane gravity field `-g grad h`; planar gravity is precomputed
 * with the exact legacy expression so the planar limit stays
 * bit-identical to the pre-#4800 integrator. */
function gravityField(
  surface: GreenSurface,
): (xM: number, yM: number) => [number, number] {
  if (surface.kind === "planar") {
    const aspect = (surface.aspectDeg * Math.PI) / 180.0;
    const grade = surface.gradePercent / 100.0;
    const gPar: [number, number] = [
      GRAVITY_M_S2 * grade * Math.cos(aspect),
      GRAVITY_M_S2 * grade * Math.sin(aspect),
    ];
    return () => gPar;
  }
  return (xM: number, yM: number): [number, number] => {
    if (!gridInside(surface, xM, yM)) return [0, 0];
    const [i, j, tx, ty] = gridCell(surface, xM, yM);
    const h00 = surface.heightsM[j][i];
    const h10 = surface.heightsM[j][i + 1];
    const h01 = surface.heightsM[j + 1][i];
    const h11 = surface.heightsM[j + 1][i + 1];
    const dhdx = ((h10 - h00) * (1.0 - ty) + (h11 - h01) * ty) / surface.spacingM;
    const dhdy = ((h01 - h00) * (1.0 - tx) + (h11 - h10) * tx) / surface.spacingM;
    return [-GRAVITY_M_S2 * dhdx, -GRAVITY_M_S2 * dhdy];
  };
}

type State = [number, number, number, number, number];

function derivative(
  state: State,
  sliding: boolean,
  muSlide: number,
  muRoll: number,
  gravityAt: (xM: number, yM: number) => [number, number],
): State {
  const [x, y, vx, vy] = state;
  const [gx, gy] = gravityAt(x, y);
  const speed = Math.hypot(vx, vy);
  if (speed <= 0) return [0, 0, gx, gy, 0];
  const mu = sliding ? muSlide : muRoll;
  return [
    vx,
    vy,
    (-mu * GRAVITY_M_S2 * vx) / speed + gx,
    (-mu * GRAVITY_M_S2 * vy) / speed + gy,
    sliding ? 2.5 * muSlide * GRAVITY_M_S2 : 0,
  ];
}

function rk4Step(
  state: State,
  sliding: boolean,
  muSlide: number,
  muRoll: number,
  gravityAt: (xM: number, yM: number) => [number, number],
): State {
  const k1 = derivative(state, sliding, muSlide, muRoll, gravityAt);
  const mid1 = state.map((s, i) => s + 0.5 * DT_S * k1[i]) as State;
  const k2 = derivative(mid1, sliding, muSlide, muRoll, gravityAt);
  const mid2 = state.map((s, i) => s + 0.5 * DT_S * k2[i]) as State;
  const k3 = derivative(mid2, sliding, muSlide, muRoll, gravityAt);
  const end = state.map((s, i) => s + DT_S * k3[i]) as State;
  const k4 = derivative(end, sliding, muSlide, muRoll, gravityAt);
  return state.map(
    (s, i) => s + (DT_S / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]),
  ) as State;
}

function integrate(
  launch: PuttLaunch,
  gravityAt: (xM: number, yM: number) => [number, number],
  muSlide: number,
  muRoll: number,
  holeDistanceM: number,
  captured: (toHoleM: number, speedMps: number) => boolean,
): PuttResult {
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
    state = rk4Step(state, sliding, muSlide, muRoll, gravityAt);
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
      if (captured(toHole, speed)) {
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

function capturePredicate(
  model: CaptureModel,
): (toHoleM: number, speedMps: number) => boolean {
  if (model === "speed_threshold") {
    const vCapture = captureSpeedMps();
    return (_toHole, speed) => speed <= vCapture;
  }
  if (model === "effective_radius") {
    return (toHole, speed) => toHole <= effectiveHoleRadiusM(speed);
  }
  throw new Error(`unknown capture model: ${String(model)}`);
}

function requirePuttInputs(launch: PuttLaunch, holeDistanceM: number): void {
  if (!(holeDistanceM >= 0.1 && holeDistanceM <= 40)) {
    throw new Error("holeDistanceM must be in [0.1, 40]");
  }
  if (!(launch.horizontalSpeedMps > 0)) {
    throw new Error("putt must start moving");
  }
}

export interface SurfacePuttOptions {
  stimpFt: number;
  holeDistanceM: number;
  muSlide?: number;
  captureModel?: CaptureModel;
}

/** Integrate one putt on a green surface (planar or heightfield). */
export function simulatePuttOnSurface(
  launch: PuttLaunch,
  surface: GreenSurface,
  options: SurfacePuttOptions,
): PuttResult {
  if (surface.kind !== "planar" && surface.kind !== "grid") {
    throw new Error("surface must be a GreenSurface");
  }
  requirePuttInputs(launch, options.holeDistanceM);
  const muSlide =
    options.muSlide === undefined ? DEFAULT_SLIDING_MU : options.muSlide;
  if (!Number.isFinite(muSlide) || !(muSlide > 0 && muSlide <= 1.5)) {
    throw new Error("muSlide must be in (0, 1.5]");
  }
  const muRoll = stimpToRollingMu(options.stimpFt);
  return integrate(
    launch,
    gravityField(surface),
    muSlide,
    muRoll,
    options.holeDistanceM,
    capturePredicate(options.captureModel ?? "effective_radius"),
  );
}

/** Integrate one putt on the uniform sloped green (legacy planar API).
 *
 * Delegates to the surface integrator with a planar surface and the
 * historic speed-threshold capture — bit-identical to the pre-#4800
 * implementation (regression-gated by the parity pins). */
export function simulatePutt(
  launch: PuttLaunch,
  green: GreenConditions,
  holeDistanceM: number,
): PuttResult {
  const muSlide = green.muSlide === undefined ? DEFAULT_SLIDING_MU : green.muSlide;
  if (
    !Number.isFinite(green.gradePercent) ||
    !(green.gradePercent >= 0 && green.gradePercent <= 10)
  ) {
    throw new Error("gradePercent must be in [0, 10]");
  }
  if (
    !Number.isFinite(green.aspectDeg) ||
    !(green.aspectDeg >= -360 && green.aspectDeg <= 360)
  ) {
    throw new Error("aspectDeg must be in [-360, 360]");
  }
  return simulatePuttOnSurface(
    launch,
    planarSurface(green.gradePercent, green.aspectDeg),
    {
      stimpFt: green.stimpFt,
      holeDistanceM,
      muSlide,
      captureModel: "speed_threshold",
    },
  );
}
