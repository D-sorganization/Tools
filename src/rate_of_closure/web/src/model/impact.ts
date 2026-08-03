/**
 * Twist-based impact-point velocity model — TypeScript port.
 *
 * This mirrors `src/rate_of_closure/model.py` exactly; the vitest suite pins
 * the same numeric cases as the pytest suite, so the two implementations
 * cannot drift apart silently.
 *
 * Frame convention (right-handed golfer): +Y toward the target, +Z up,
 * +X toward the trail side. Negative path deviation = LEFT.
 */

export interface ImpactScenario {
  clubheadSpeedMph: number;
  omegaPlaneDps: number;
  omegaShaftDps: number;
  lieAngleDeg: number;
  comToFaceMm: number;
  impactOffsetToeMm: number;
  impactOffsetHighMm: number;
  contactDurationUs: number;
}

export interface ImpactResult {
  referenceSpeedMph: number;
  pointSpeedMph: number;
  speedDeltaMph: number;
  tangentialSpeedMph: number;
  pathDeviationDeg: number;
  aoaDeviationDeg: number;
  closureDuringContactDeg: number;
  loftGainDuringContactDeg: number;
  pointVelocityMps: [number, number, number];
  omegaDps: [number, number, number];
  shaftAxis: [number, number, number];
  planeNormal: [number, number, number];
}

export const MPH_PER_MPS = 1.0 / 0.44704;
const DEG = Math.PI / 180.0;

type Vec3 = [number, number, number];

/** Inclusive physical bounds per field, matching the Python model. */
export const BOUNDS: Record<keyof ImpactScenario, [number, number]> = {
  clubheadSpeedMph: [1.0, 250.0],
  omegaPlaneDps: [-20000.0, 20000.0],
  omegaShaftDps: [-20000.0, 20000.0],
  lieAngleDeg: [10.0, 90.0],
  comToFaceMm: [0.0, 150.0],
  impactOffsetToeMm: [-80.0, 80.0],
  impactOffsetHighMm: [-40.0, 40.0],
  contactDurationUs: [0.0, 2000.0],
};

export const DEFAULT_SCENARIO: ImpactScenario = {
  clubheadSpeedMph: 120.0,
  omegaPlaneDps: 2200.0,
  omegaShaftDps: 1700.0,
  lieAngleDeg: 58.0,
  comToFaceMm: 35.0,
  impactOffsetToeMm: 0.0,
  impactOffsetHighMm: 0.0,
  contactDurationUs: 450.0,
};

function cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function norm(v: Vec3): number {
  return Math.hypot(v[0], v[1], v[2]);
}

/** Validate a scenario against the physical bounds; throws RangeError. */
export function validateScenario(scenario: ImpactScenario): void {
  for (const key of Object.keys(BOUNDS) as (keyof ImpactScenario)[]) {
    const value = scenario[key];
    if (!Number.isFinite(value)) {
      throw new RangeError(`${key} must be finite`);
    }
    const [low, high] = BOUNDS[key];
    if (value < low || value > high) {
      throw new RangeError(`${key} must be within [${low}, ${high}]`);
    }
  }
}

/** Shaft axis and swing-plane normal for a given impact lie angle. */
export function frame(lieAngleDeg: number): { shaft: Vec3; normal: Vec3 } {
  const lie = lieAngleDeg * DEG;
  const shaft: Vec3 = [-Math.cos(lie), 0.0, Math.sin(lie)];
  const raw = cross([0.0, 1.0, 0.0], shaft);
  const n = norm(raw);
  return { shaft, normal: [raw[0] / n, raw[1] / n, raw[2] / n] };
}

/** Solve one scenario for the impact point's delivery deviation. */
export function solve(scenario: ImpactScenario): ImpactResult {
  validateScenario(scenario);
  const { shaft, normal } = frame(scenario.lieAngleDeg);
  const omega: Vec3 = [0, 1, 2].map(
    (i) =>
      scenario.omegaPlaneDps * DEG * normal[i] +
      scenario.omegaShaftDps * DEG * shaft[i],
  ) as Vec3;
  const lever: Vec3 = [
    scenario.impactOffsetToeMm / 1000.0,
    scenario.comToFaceMm / 1000.0,
    scenario.impactOffsetHighMm / 1000.0,
  ];
  const vRef: Vec3 = [0.0, scenario.clubheadSpeedMph / MPH_PER_MPS, 0.0];
  const tangential = cross(omega, lever);
  const vPoint: Vec3 = [
    vRef[0] + tangential[0],
    vRef[1] + tangential[1],
    vRef[2] + tangential[2],
  ];

  const refSpeed = norm(vRef);
  const pointSpeed = norm(vPoint);
  const contactS = scenario.contactDurationUs * 1e-6;

  return {
    referenceSpeedMph: refSpeed * MPH_PER_MPS,
    pointSpeedMph: pointSpeed * MPH_PER_MPS,
    speedDeltaMph: (pointSpeed - refSpeed) * MPH_PER_MPS,
    tangentialSpeedMph: norm(tangential) * MPH_PER_MPS,
    pathDeviationDeg: Math.atan2(vPoint[0], vPoint[1]) / DEG,
    aoaDeviationDeg:
      Math.atan2(vPoint[2], Math.hypot(vPoint[0], vPoint[1])) / DEG,
    closureDuringContactDeg: (omega[2] / DEG) * contactS,
    loftGainDuringContactDeg: (omega[0] / DEG) * contactS,
    pointVelocityMps: vPoint,
    omegaDps: [omega[0] / DEG, omega[1] / DEG, omega[2] / DEG],
    shaftAxis: shaft,
    planeNormal: normal,
  };
}
