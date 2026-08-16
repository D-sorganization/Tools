/** Deterministic version-1 persistence and legacy 2D migration for 3D targets. */

import {
  boxTolerance,
  createSpatialTarget,
  sphereTolerance,
  surfaceCircleTolerance,
  surfaceCorridorTolerance,
  targetPointFromCanonicalApp,
  type AcceptanceGeometryTs,
  type SpatialTargetTs,
} from "./spatialTarget";

export const SPATIAL_TARGET_SCHEMA = "swing_sim.spatial_target";
export const SPATIAL_TARGET_SCHEMA_VERSION = 1;
export const LEGACY_GROUND_SOURCE = "legacy.course_surface/default";

type JsonPrimitive = string | number | boolean | null;
type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };
type JsonRecord = Record<string, JsonValue>;

const CURRENT_FIELDS = [
  "schema",
  "schema_version",
  "units",
  "frame",
  "source_frame",
  "label",
  "kind",
  "position_m",
  "elevation_source",
  "ground_source",
  "tolerance",
] as const;

const LEGACY_SNAKE_FIELDS = [
  "kind",
  "distance_m",
  "radius_m",
  "lateral_m",
  "band_half_length_m",
  "half_width_m",
] as const;

const LEGACY_CAMEL_FIELDS = [
  "kind",
  "distanceM",
  "radiusM",
  "lateralM",
  "bandHalfLengthM",
  "halfWidthM",
] as const;

function record(value: unknown, name: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError(`${name} must be a mapping`);
  }
  return value as Record<string, unknown>;
}

function exactFields(
  value: Record<string, unknown>,
  allowed: readonly string[],
  name: string,
): void {
  const unknown = Object.keys(value).filter((key) => !allowed.includes(key));
  if (unknown.length > 0) throw new RangeError(`${name} has unknown fields: ${unknown.sort()}`);
  const missing = allowed.filter((key) => !(key in value));
  if (missing.length > 0) throw new RangeError(`${name} is missing fields: ${missing}`);
}

function stringValue(value: unknown, name: string): string {
  if (typeof value !== "string") throw new TypeError(`${name} must be a string`);
  return value;
}

function numberValue(value: unknown, name: string): number {
  if (typeof value !== "number") throw new TypeError(`${name} must be a number`);
  if (!Number.isFinite(value)) throw new RangeError(`${name} must be finite`);
  return value;
}

function toleranceToRecord(tolerance: AcceptanceGeometryTs): JsonRecord {
  switch (tolerance.kind) {
    case "sphere":
      return { kind: "sphere", radius_m: tolerance.radiusM };
    case "box":
      return {
        kind: "box",
        half_extents_m: {
          x: tolerance.halfExtentsM[0],
          elevation: tolerance.halfExtentsM[1],
          right: tolerance.halfExtentsM[2],
        },
      };
    case "surface_circle":
      return { kind: "surface_circle", radius_m: tolerance.radiusM };
    case "surface_corridor":
      return {
        kind: "surface_corridor",
        half_length_m: tolerance.halfLengthM,
        half_width_m: tolerance.halfWidthM,
      };
  }
}

function targetToRecord(targetInput: SpatialTargetTs): JsonRecord {
  const target = createSpatialTarget(targetInput);
  const [downrange, elevation, right] = target.point.appCoordinatesM;
  return {
    schema: SPATIAL_TARGET_SCHEMA,
    schema_version: SPATIAL_TARGET_SCHEMA_VERSION,
    units: target.units,
    frame: target.frame,
    source_frame: target.point.sourceFrame,
    label: target.label,
    kind: target.kind,
    position_m: { x: downrange, elevation, right },
    elevation_source: target.elevationSource,
    ground_source: target.groundSource,
    tolerance: toleranceToRecord(target.tolerance),
  };
}

function sortedJsonValue(value: JsonValue): JsonValue {
  if (Array.isArray(value)) return value.map(sortedJsonValue);
  if (value === null || typeof value !== "object") return value;
  return Object.fromEntries(
    Object.keys(value)
      .sort()
      .map((key) => [key, sortedJsonValue(value[key])]),
  );
}

/** Encode deterministic compact JSON with recursively sorted keys. */
export function spatialTargetToJson(target: SpatialTargetTs): string {
  return JSON.stringify(sortedJsonValue(targetToRecord(target)));
}

function parsePosition(value: unknown, sourceFrame: unknown) {
  const position = record(value, "position_m");
  exactFields(position, ["x", "elevation", "right"], "position_m");
  return targetPointFromCanonicalApp(
    [
      numberValue(position.x, "position_m.x"),
      numberValue(position.elevation, "position_m.elevation"),
      numberValue(position.right, "position_m.right"),
    ],
    sourceFrame,
  );
}

function parseHalfExtents(value: unknown) {
  const extents = record(value, "half_extents_m");
  exactFields(extents, ["x", "elevation", "right"], "half_extents_m");
  return boxTolerance([
    numberValue(extents.x, "half_extents_m.x"),
    numberValue(extents.elevation, "half_extents_m.elevation"),
    numberValue(extents.right, "half_extents_m.right"),
  ]);
}

function parseTolerance(value: unknown): AcceptanceGeometryTs {
  const tolerance = record(value, "tolerance");
  const kind = stringValue(tolerance.kind, "tolerance.kind");
  switch (kind) {
    case "sphere":
      exactFields(tolerance, ["kind", "radius_m"], "tolerance");
      return sphereTolerance(numberValue(tolerance.radius_m, "tolerance.radius_m"));
    case "box":
      exactFields(tolerance, ["kind", "half_extents_m"], "tolerance");
      return parseHalfExtents(tolerance.half_extents_m);
    case "surface_circle":
      exactFields(tolerance, ["kind", "radius_m"], "tolerance");
      return surfaceCircleTolerance(numberValue(tolerance.radius_m, "tolerance.radius_m"));
    case "surface_corridor":
      exactFields(tolerance, ["kind", "half_length_m", "half_width_m"], "tolerance");
      return surfaceCorridorTolerance(
        numberValue(tolerance.half_length_m, "tolerance.half_length_m"),
        numberValue(tolerance.half_width_m, "tolerance.half_width_m"),
      );
    default:
      throw new RangeError(`unknown tolerance kind ${kind}`);
  }
}

function parseCurrent(data: Record<string, unknown>): SpatialTargetTs {
  exactFields(data, CURRENT_FIELDS, "spatial target");
  if (data.schema !== SPATIAL_TARGET_SCHEMA) {
    throw new RangeError(`schema must be ${SPATIAL_TARGET_SCHEMA}`);
  }
  if (data.schema_version !== SPATIAL_TARGET_SCHEMA_VERSION) {
    throw new RangeError(`unsupported schema_version ${String(data.schema_version)}`);
  }
  if (data.units !== "m") throw new RangeError("units must be 'm'");
  if (data.frame !== "app") throw new RangeError("frame must be 'app'");
  const groundSource =
    data.ground_source === null
      ? null
      : stringValue(data.ground_source, "ground_source");
  return createSpatialTarget({
    label: stringValue(data.label, "label"),
    kind: stringValue(data.kind, "kind") as SpatialTargetTs["kind"],
    point: parsePosition(data.position_m, data.source_frame),
    tolerance: parseTolerance(data.tolerance),
    elevationSource: stringValue(
      data.elevation_source,
      "elevation_source",
    ) as SpatialTargetTs["elevationSource"],
    groundSource,
    units: "m",
    frame: "app",
  });
}

function legacyNumber(
  data: Record<string, unknown>,
  snakeName: string,
  camelName: string,
  fallback: number,
): number {
  const value = data[snakeName] ?? data[camelName] ?? fallback;
  return numberValue(value, snakeName);
}

function parseLegacy(data: Record<string, unknown>): SpatialTargetTs {
  const camelCase = Object.keys(data).some((key) => key.includes("M"));
  const allowed = camelCase ? LEGACY_CAMEL_FIELDS : LEGACY_SNAKE_FIELDS;
  const unknown = Object.keys(data).filter((key) => !allowed.includes(key as never));
  if (unknown.length > 0) throw new RangeError(`legacy target has unknown fields: ${unknown}`);
  const kind = stringValue(data.kind, "kind");
  if (kind !== "green" && kind !== "fairway") {
    throw new RangeError(`unknown legacy target kind ${kind}`);
  }
  const distanceM = legacyNumber(data, "distance_m", "distanceM", 230);
  const lateralM = kind === "green" ? legacyNumber(data, "lateral_m", "lateralM", 0) : 0;
  const tolerance =
    kind === "green"
      ? surfaceCircleTolerance(legacyNumber(data, "radius_m", "radiusM", 10))
      : surfaceCorridorTolerance(
          legacyNumber(data, "band_half_length_m", "bandHalfLengthM", 15),
          legacyNumber(data, "half_width_m", "halfWidthM", 16),
        );
  return createSpatialTarget({
    label: `Migrated ${kind[0].toUpperCase()}${kind.slice(1)} Target`,
    kind: "landing_area",
    point: targetPointFromCanonicalApp([distanceM, 0, lateralM], "app"),
    tolerance,
    elevationSource: "course_surface",
    groundSource: LEGACY_GROUND_SOURCE,
  });
}

/** Decode version 1 or migrate an explicit unversioned 2D target. */
export function spatialTargetFromJson(text: string): SpatialTargetTs {
  if (typeof text !== "string") throw new TypeError("target JSON must be text");
  let decoded: unknown;
  try {
    decoded = JSON.parse(text);
  } catch {
    throw new RangeError("target must contain valid JSON");
  }
  const data = record(decoded, "spatial target");
  return "schema_version" in data ? parseCurrent(data) : parseLegacy(data);
}
