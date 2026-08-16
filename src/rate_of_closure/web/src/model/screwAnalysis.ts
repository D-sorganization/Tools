/** Renderer-neutral instantaneous screw analysis in the app/world frame. */

export type Vec3 = [number, number, number];
export type Twist6 = [number, number, number, number, number, number];
export type MotionKind = "finite" | "translation" | "stationary";

export interface ScrewMotionTs {
  kind: MotionKind;
  referencePointM: Vec3;
  referenceVelocityMps: Vec3;
  angularVelocityRadS: Vec3;
  axisDirection: Vec3;
  axisPointM: Vec3 | null;
  pitchMPerRad: number | null;
  angularRateRadS: number;
  axialSpeedMps: number;
  radiusM: number | null;
  orbitalVelocityMps: Vec3;
  axialVelocityMps: Vec3;
  reconstructionResidualMps: number;
}

export interface MotionProjectionTs {
  direction: Vec3;
  totalMps: number;
  orbitalMps: number;
  axialMps: number;
}

export interface ScrewGlyphTs {
  axisLineM: [Vec3, Vec3];
  helixM: Vec3[];
  radiusLineM: [Vec3, Vec3];
  handedness: 1 | -1;
}

export interface JointMotionAtTs {
  jointIds: string[];
  axisPointsM: Vec3[];
  angularVelocityRadS: Vec3[];
  contributionVelocityMps: Vec3[];
  endpointVelocityMps: Vec3;
  reconstructionResidualMps: number;
}

const EPSILON = 1e-10;
const GLYPH_POINTS = 96;

const add = (first: Vec3, second: Vec3): Vec3 => [
  first[0] + second[0], first[1] + second[1], first[2] + second[2],
];
const subtract = (first: Vec3, second: Vec3): Vec3 => [
  first[0] - second[0], first[1] - second[1], first[2] - second[2],
];
const scale = (vector: Vec3, factor: number): Vec3 => [
  vector[0] * factor, vector[1] * factor, vector[2] * factor,
];
const dot = (first: Vec3, second: Vec3): number =>
  first[0] * second[0] + first[1] * second[1] + first[2] * second[2];
const cross = (first: Vec3, second: Vec3): Vec3 => [
  first[1] * second[2] - first[2] * second[1],
  first[2] * second[0] - first[0] * second[2],
  first[0] * second[1] - first[1] * second[0],
];
const norm = (vector: Vec3): number => Math.hypot(...vector);

function finiteVector(vector: Vec3, name: string): void {
  if (vector.length !== 3 || vector.some((value) => !Number.isFinite(value))) {
    throw new Error(`${name} must be a finite three-vector`);
  }
}

function finiteMotion(
  omega: Vec3,
  velocity: Vec3,
  reference: Vec3,
): ScrewMotionTs {
  const rateSquared = dot(omega, omega);
  const rate = Math.sqrt(rateSquared);
  const pitch = dot(omega, velocity) / rateSquared;
  const axisPoint = add(reference, scale(cross(omega, velocity), 1 / rateSquared));
  const axial = scale(omega, pitch);
  const orbital = subtract(velocity, axial);
  return {
    kind: "finite",
    referencePointM: [...reference],
    referenceVelocityMps: [...velocity],
    angularVelocityRadS: [...omega],
    axisDirection: scale(omega, 1 / rate),
    axisPointM: axisPoint,
    pitchMPerRad: pitch,
    angularRateRadS: rate,
    axialSpeedMps: pitch * rate,
    radiusM: norm(subtract(reference, axisPoint)),
    orbitalVelocityMps: orbital,
    axialVelocityMps: axial,
    reconstructionResidualMps: norm(subtract(add(orbital, axial), velocity)),
  };
}

/** Decompose [angular velocity, velocity at reference] into screw motion. */
export function analyzeTwist(twist: Twist6, referencePointM: Vec3): ScrewMotionTs {
  if (twist.length !== 6 || twist.some((value) => !Number.isFinite(value))) {
    throw new Error("twist must be a finite six-vector");
  }
  finiteVector(referencePointM, "referencePointM");
  const omega: Vec3 = [twist[0], twist[1], twist[2]];
  const velocity: Vec3 = [twist[3], twist[4], twist[5]];
  if (norm(omega) > EPSILON) return finiteMotion(omega, velocity, referencePointM);
  const speed = norm(velocity);
  const translating = speed > EPSILON;
  return {
    kind: translating ? "translation" : "stationary",
    referencePointM: [...referencePointM],
    referenceVelocityMps: [...velocity],
    angularVelocityRadS: omega,
    axisDirection: translating ? scale(velocity, 1 / speed) : [0, 0, 0],
    axisPointM: null,
    pitchMPerRad: null,
    angularRateRadS: 0,
    axialSpeedMps: translating ? speed : 0,
    radiusM: null,
    orbitalVelocityMps: [0, 0, 0],
    axialVelocityMps: [...velocity],
    reconstructionResidualMps: 0,
  };
}

const DEFAULT_DIRECTIONS = {
  target: [1, 0, 0] as Vec3,
  vertical: [0, 1, 0] as Vec3,
  lateral: [0, 0, 1] as Vec3,
};

/** Return signed total, orbital, and axial velocity projections. */
export function projectMotion(
  motion: ScrewMotionTs,
  directions: Record<string, Vec3> = DEFAULT_DIRECTIONS,
): Record<string, MotionProjectionTs> {
  return Object.fromEntries(Object.entries(directions).map(([name, rawDirection]) => {
    finiteVector(rawDirection, `direction[${name}]`);
    const magnitude = norm(rawDirection);
    if (magnitude <= EPSILON) throw new Error(`direction[${name}] must be nonzero`);
    const direction = scale(rawDirection, 1 / magnitude);
    return [name, {
      direction,
      totalMps: dot(motion.referenceVelocityMps, direction),
      orbitalMps: dot(motion.orbitalVelocityMps, direction),
      axialMps: dot(motion.axialVelocityMps, direction),
    }];
  }));
}

function orthogonalBasis(axis: Vec3): [Vec3, Vec3] {
  const index = axis.map(Math.abs).indexOf(Math.min(...axis.map(Math.abs)));
  const seeds: Vec3[] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const firstRaw = cross(axis, seeds[index]);
  const first = scale(firstRaw, 1 / norm(firstRaw));
  return [first, cross(axis, first)];
}

function dominantSign(vector: Vec3): 1 | -1 {
  const index = vector.map(Math.abs).indexOf(Math.max(...vector.map(Math.abs)));
  return vector[index] >= 0 ? 1 : -1;
}

/** Build bounded 3D axis/helix/radius geometry for a finite screw. */
export function buildScrewGlyph(
  motion: ScrewMotionTs,
  sceneExtentM: number,
): ScrewGlyphTs | null {
  if (!Number.isFinite(sceneExtentM) || sceneExtentM <= 0) {
    throw new Error("sceneExtentM must be finite and positive");
  }
  if (motion.kind !== "finite" || motion.axisPointM === null) return null;
  const growth = 0.55 + 0.35 * Math.tanh(motion.angularRateRadS / 10);
  const halfLength = sceneExtentM * growth;
  const axis = motion.axisDirection;
  const point = motion.axisPointM;
  const [first, second] = orthogonalBasis(axis);
  const handedness = dominantSign(motion.angularVelocityRadS);
  const radius = sceneExtentM * 0.055;
  const helixM = Array.from({ length: GLYPH_POINTS }, (_, index): Vec3 => {
    const fraction = index / (GLYPH_POINTS - 1);
    const phase = -2 * Math.PI + fraction * 4 * Math.PI;
    const axial = (-0.82 + fraction * 1.64) * halfLength;
    return add(add(
      add(point, scale(axis, axial)),
      scale(first, radius * Math.cos(phase)),
    ), scale(second, handedness * radius * Math.sin(phase)));
  });
  return {
    axisLineM: [subtract(point, scale(axis, halfLength)), add(point, scale(axis, halfLength))],
    helixM,
    radiusLineM: [point, motion.referencePointM],
    handedness,
  };
}

function derivativeAt(values: Vec3[], times: number[], index: number): Vec3 {
  const lower = index === 0 ? 0 : index - 1;
  const upper = index === values.length - 1 ? values.length - 1 : index + 1;
  const interval = times[upper] - times[lower];
  if (!(interval > 0)) throw new Error("times must be strictly increasing");
  return scale(subtract(values[upper], values[lower]), 1 / interval);
}

/** Reconstruct revolute-joint screw contributions at one sampled instant. */
export function jointMotionAt(
  times: number[],
  jointPositionsM: Vec3[][],
  jointIds: string[],
  rawIndex: number,
): JointMotionAtTs {
  if (times.length < 3 || jointPositionsM.length !== times.length) {
    throw new Error("joint motion requires at least three aligned samples");
  }
  if (jointPositionsM.some((row) => row.length !== jointIds.length + 1)) {
    throw new Error("each joint row must contain one more point than joint IDs");
  }
  const index = Math.max(0, Math.min(Math.round(rawIndex), times.length - 1));
  const segmentsByJoint = jointIds.map((_, jointIndex) =>
    jointPositionsM.map((row) => subtract(row[jointIndex + 1], row[jointIndex])));
  const absoluteOmega = segmentsByJoint.map((segments) => {
    const segment = segments[index];
    const rate = derivativeAt(segments, times, index);
    const lengthSquared = dot(segment, segment);
    if (lengthSquared <= EPSILON) throw new Error("joint segments must be nonzero");
    return scale(cross(segment, rate), 1 / lengthSquared);
  });
  const relativeOmega = absoluteOmega.map((omega, jointIndex) =>
    jointIndex === 0 ? omega : subtract(omega, absoluteOmega[jointIndex - 1]));
  const points = jointPositionsM[index];
  const endpoint = points[points.length - 1];
  const contributions = relativeOmega.map((omega, jointIndex) =>
    cross(omega, subtract(endpoint, points[jointIndex])));
  const endpointVelocity = derivativeAt(
    jointPositionsM.map((row) => row[row.length - 1]),
    times,
    index,
  );
  const reconstructed = contributions.reduce(add, [0, 0, 0]);
  return {
    jointIds: [...jointIds],
    axisPointsM: points.slice(0, -1),
    angularVelocityRadS: relativeOmega,
    contributionVelocityMps: contributions,
    endpointVelocityMps: endpointVelocity,
    reconstructionResidualMps: norm(subtract(reconstructed, endpointVelocity)),
  };
}
