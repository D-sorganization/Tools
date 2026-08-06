/** UI-neutral 3D targets in the app frame: downrange, elevation, right [m]. */

export type Vector3 = readonly [number, number, number];
export type TargetFrame = "app" | "flight";
export type SpatialTargetKind = "landing_area" | "aerial_waypoint";
export type ElevationSource = "course_surface" | "absolute";

export interface TargetPointTs {
  readonly appCoordinatesM: Vector3;
  readonly sourceFrame: TargetFrame;
}

export interface SphereToleranceTs {
  readonly kind: "sphere";
  readonly radiusM: number;
}

export interface BoxToleranceTs {
  readonly kind: "box";
  readonly halfExtentsM: Vector3;
}

export interface SurfaceCircleToleranceTs {
  readonly kind: "surface_circle";
  readonly radiusM: number;
}

export interface SurfaceCorridorToleranceTs {
  readonly kind: "surface_corridor";
  readonly halfLengthM: number;
  readonly halfWidthM: number;
}

export type AcceptanceGeometryTs =
  | SphereToleranceTs
  | BoxToleranceTs
  | SurfaceCircleToleranceTs
  | SurfaceCorridorToleranceTs;

export interface SpatialTargetTs {
  readonly label: string;
  readonly kind: SpatialTargetKind;
  readonly point: TargetPointTs;
  readonly tolerance: AcceptanceGeometryTs;
  readonly elevationSource: ElevationSource;
  readonly groundSource: string | null;
  readonly units: "m";
  readonly frame: "app";
}

export interface SpatialTargetInput {
  readonly label: string;
  readonly kind: SpatialTargetKind;
  readonly point: TargetPointTs;
  readonly tolerance: AcceptanceGeometryTs;
  readonly elevationSource: ElevationSource;
  readonly groundSource?: string | null;
  readonly units?: string;
  readonly frame?: string;
}

export interface SpatialTargetMissTs {
  readonly closestPointM: Vector3;
  /** Actual minus closest accepted point: positive means long, high, right. */
  readonly vectorM: Vector3;
  readonly distanceM: number;
  readonly accepted: boolean;
}

const ACCEPTANCE_ABS_TOL_M = 1e-12;

function finiteNumber(value: unknown, name: string): number {
  if (typeof value !== "number") throw new TypeError(`${name} must be a number`);
  if (!Number.isFinite(value)) throw new RangeError(`${name} must be finite`);
  return value;
}

function positiveNumber(value: unknown, name: string): number {
  const number = finiteNumber(value, name);
  if (number <= 0) throw new RangeError(`${name} must be finite and > 0`);
  return number;
}

function vector3(value: unknown, name: string): Vector3 {
  if (!Array.isArray(value) || value.length !== 3) {
    throw new TypeError(`${name} must contain exactly three coordinates`);
  }
  return Object.freeze([
    finiteNumber(value[0], name),
    finiteNumber(value[1], name),
    finiteNumber(value[2], name),
  ]);
}

function targetFrame(value: unknown, name = "frame"): TargetFrame {
  if (typeof value !== "string") throw new TypeError(`${name} must be a string`);
  if (value !== "app" && value !== "flight") {
    throw new RangeError(`${name} must be 'app' or 'flight'`);
  }
  return value;
}

function nonemptyText(value: unknown, name: string): string {
  if (typeof value !== "string") throw new TypeError(`${name} must be a string`);
  if (value.length === 0 || value.trim() !== value) {
    throw new RangeError(`${name} must be non-empty and trimmed`);
  }
  return value;
}

function canonicalTargetPoint(
  appCoordinatesM: unknown,
  sourceFrame: unknown,
): TargetPointTs {
  return Object.freeze({
    appCoordinatesM: vector3(appCoordinatesM, "appCoordinatesM"),
    sourceFrame: targetFrame(sourceFrame, "source_frame"),
  });
}

/** Convert source coordinates through the same app/flight mapping as Python. */
export function targetPointFromFrame(
  coordinatesM: unknown,
  sourceFrame: TargetFrame,
): TargetPointTs {
  const coordinates = vector3(coordinatesM, "coordinatesM");
  const validatedFrame = targetFrame(sourceFrame, "source_frame");
  const appCoordinates: Vector3 =
    validatedFrame === "app"
      ? coordinates
      : [coordinates[0], coordinates[2], -coordinates[1]];
  return canonicalTargetPoint(appCoordinates, validatedFrame);
}

/** Return a canonical point in the requested app or flight frame. */
export function targetPointInFrame(
  point: TargetPointTs,
  frame: TargetFrame,
): Vector3 {
  const validatedPoint = canonicalTargetPoint(point.appCoordinatesM, point.sourceFrame);
  const validatedFrame = targetFrame(frame);
  const [downrange, elevation, right] = validatedPoint.appCoordinatesM;
  return validatedFrame === "app"
    ? validatedPoint.appCoordinatesM
    : Object.freeze([downrange, -right, elevation]);
}

export function sphereTolerance(radiusM: number): SphereToleranceTs {
  return Object.freeze({ kind: "sphere", radiusM: positiveNumber(radiusM, "radiusM") });
}

export function boxTolerance(halfExtentsM: unknown): BoxToleranceTs {
  const extents = vector3(halfExtentsM, "halfExtentsM");
  if (extents.some((extent) => extent <= 0)) {
    throw new RangeError("halfExtentsM must be finite and > 0");
  }
  return Object.freeze({ kind: "box", halfExtentsM: extents });
}

export function surfaceCircleTolerance(radiusM: number): SurfaceCircleToleranceTs {
  return Object.freeze({
    kind: "surface_circle",
    radiusM: positiveNumber(radiusM, "radiusM"),
  });
}

export function surfaceCorridorTolerance(
  halfLengthM: number,
  halfWidthM: number,
): SurfaceCorridorToleranceTs {
  return Object.freeze({
    kind: "surface_corridor",
    halfLengthM: positiveNumber(halfLengthM, "halfLengthM"),
    halfWidthM: positiveNumber(halfWidthM, "halfWidthM"),
  });
}

function validatedTolerance(tolerance: AcceptanceGeometryTs): AcceptanceGeometryTs {
  switch (tolerance.kind) {
    case "sphere":
      return sphereTolerance(tolerance.radiusM);
    case "box":
      return boxTolerance(tolerance.halfExtentsM);
    case "surface_circle":
      return surfaceCircleTolerance(tolerance.radiusM);
    case "surface_corridor":
      return surfaceCorridorTolerance(tolerance.halfLengthM, tolerance.halfWidthM);
  }
}

function validateKindContract(target: SpatialTargetTs): void {
  const surface =
    target.tolerance.kind === "surface_circle" ||
    target.tolerance.kind === "surface_corridor";
  if (target.kind === "landing_area") {
    if (!surface) throw new RangeError("landing_area requires a surface tolerance");
    if (target.elevationSource !== "course_surface") {
      throw new RangeError("landing_area requires course_surface elevation");
    }
    if (target.groundSource === null) {
      throw new RangeError("landing_area requires groundSource");
    }
    nonemptyText(target.groundSource, "groundSource");
    return;
  }
  if (surface) throw new RangeError("aerial_waypoint requires a 3D volume tolerance");
  if (target.elevationSource !== "absolute") {
    throw new RangeError("aerial_waypoint requires absolute elevation");
  }
  if (target.groundSource !== null) {
    throw new RangeError("aerial_waypoint groundSource must be null");
  }
}

/** Validate and defensively freeze a target contract. */
export function createSpatialTarget(input: SpatialTargetInput): SpatialTargetTs {
  if (input.kind !== "landing_area" && input.kind !== "aerial_waypoint") {
    throw new RangeError(`unknown target kind ${String(input.kind)}`);
  }
  if (input.units !== undefined && input.units !== "m") {
    throw new RangeError("units must be 'm'");
  }
  if (input.frame !== undefined && input.frame !== "app") {
    throw new RangeError("frame must be 'app'");
  }
  const target: SpatialTargetTs = Object.freeze({
    label: nonemptyText(input.label, "label"),
    kind: input.kind,
    point: canonicalTargetPoint(input.point.appCoordinatesM, input.point.sourceFrame),
    tolerance: validatedTolerance(input.tolerance),
    elevationSource: input.elevationSource,
    groundSource: input.groundSource ?? null,
    units: "m",
    frame: "app",
  });
  validateKindContract(target);
  return target;
}

function closestSphere(actual: Vector3, center: Vector3, radiusM: number): Vector3 {
  const delta: Vector3 = [
    actual[0] - center[0],
    actual[1] - center[1],
    actual[2] - center[2],
  ];
  const distance = Math.hypot(...delta);
  if (distance <= radiusM) return actual;
  const scale = radiusM / distance;
  return Object.freeze([
    center[0] + scale * delta[0],
    center[1] + scale * delta[1],
    center[2] + scale * delta[2],
  ]);
}

function closestBox(actual: Vector3, center: Vector3, extents: Vector3): Vector3 {
  if (actual.every((value, index) => Math.abs(value - center[index]) <= extents[index])) {
    return actual;
  }
  return Object.freeze(
    actual.map((value, index) =>
      Math.min(Math.max(value, center[index] - extents[index]), center[index] + extents[index]),
    ) as [number, number, number],
  );
}

function closestSurfaceCircle(
  actual: Vector3,
  center: Vector3,
  radiusM: number,
): Vector3 {
  const deltaX = actual[0] - center[0];
  const deltaRight = actual[2] - center[2];
  const radialDistance = Math.hypot(deltaX, deltaRight);
  if (radialDistance <= radiusM) return Object.freeze([actual[0], center[1], actual[2]]);
  const scale = radiusM / radialDistance;
  return Object.freeze([
    center[0] + scale * deltaX,
    center[1],
    center[2] + scale * deltaRight,
  ]);
}

function closestPoint(target: SpatialTargetTs, actual: Vector3): Vector3 {
  const center = target.point.appCoordinatesM;
  const tolerance = target.tolerance;
  if (tolerance.kind === "sphere") return closestSphere(actual, center, tolerance.radiusM);
  if (tolerance.kind === "box") return closestBox(actual, center, tolerance.halfExtentsM);
  if (tolerance.kind === "surface_circle") {
    return closestSurfaceCircle(actual, center, tolerance.radiusM);
  }
  return Object.freeze([
    Math.min(Math.max(actual[0], center[0] - tolerance.halfLengthM), center[0] + tolerance.halfLengthM),
    center[1],
    Math.min(Math.max(actual[2], center[2] - tolerance.halfWidthM), center[2] + tolerance.halfWidthM),
  ]);
}

/** Compute actual-minus-closest signed miss in app-frame metres. */
export function spatialTargetMiss(
  targetInput: SpatialTargetTs,
  actualAppM: unknown,
): SpatialTargetMissTs {
  const target = createSpatialTarget(targetInput);
  const actual = vector3(actualAppM, "actualAppM");
  const closest = closestPoint(target, actual);
  let vector: Vector3 = Object.freeze([
    actual[0] - closest[0],
    actual[1] - closest[1],
    actual[2] - closest[2],
  ]);
  let distanceM = Math.hypot(...vector);
  const accepted = distanceM <= ACCEPTANCE_ABS_TOL_M;
  if (accepted) {
    vector = Object.freeze([0, 0, 0]);
    distanceM = 0;
  }
  return Object.freeze({ closestPointM: closest, vectorM: vector, distanceM, accepted });
}

/** Convert an observed point before computing the canonical app-frame miss. */
export function spatialTargetMissFromFrame(
  target: SpatialTargetTs,
  actualM: unknown,
  frame: TargetFrame,
): SpatialTargetMissTs {
  const point = targetPointFromFrame(actualM, frame);
  return spatialTargetMiss(target, point.appCoordinatesM);
}

/** Internal decoder seam: coordinates are already canonical app-frame values. */
export function targetPointFromCanonicalApp(
  coordinatesM: unknown,
  sourceFrame: unknown,
): TargetPointTs {
  return canonicalTargetPoint(coordinatesM, sourceFrame);
}
