/** Shared fail-closed validators for flight-to-ground wire records. */

import {
  GROUND_TARGET_FRAME,
  type GroundCalibration,
  type GroundContactState,
  type GroundProvenance,
  type GroundSurfaceProfile,
  type GroundVec3,
} from "./flightGroundTypes";
import { hasUnpairedSurrogate } from "./unicodeScalar";

export type JsonRecord = Record<string, unknown>;
export const MIN_CANONICAL_POSITIVE = 1e-11;
const UNIT_TOLERANCE = 1e-9;

export const record = (value: unknown, name: string): JsonRecord => {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError(name + " must be an object");
  }
  return value as JsonRecord;
};

export const exact = (value: JsonRecord, keys: readonly string[], name: string): void => {
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  if (actual.length !== expected.length || actual.some((key, index) => key !== expected[index])) {
    throw new RangeError(name + " fields do not match v1 schema");
  }
};

export const text = (value: unknown, name: string): string => {
  if (typeof value !== "string" || value.length === 0) throw new TypeError(name + " must be nonempty");
  if (value !== value.replace(/^[\s\v\f]+|[\s\v\f]+$/g, "")) {
    throw new RangeError(name + " must not have leading or trailing whitespace");
  }
  if (hasUnpairedSurrogate(value)) {
    throw new RangeError(name + " must not contain unpaired surrogate code points");
  }
  return value;
};

export const finiteRaw = (value: unknown, name: string): number => {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new TypeError(name + " must be finite");
  if (Number.isInteger(value) && !Number.isSafeInteger(value)) {
    throw new RangeError(name + " must lie within the cross-runtime safe range");
  }
  return value;
};

export const canonicalNumber = (value: unknown, name: string): number => {
  const raw = finiteRaw(value, name);
  const normalized = Number(raw.toFixed(11));
  return Object.is(normalized, -0) ? 0 : normalized;
};

export const nonnegative = (value: unknown, name: string): number => {
  const raw = finiteRaw(value, name);
  if (raw < 0) throw new RangeError(name + " must be nonnegative");
  return canonicalNumber(raw, name);
};

export const positive = (value: unknown, name: string): number => {
  const raw = finiteRaw(value, name);
  if (raw < MIN_CANONICAL_POSITIVE) {
    throw new RangeError(name + " must be at least " + MIN_CANONICAL_POSITIVE);
  }
  return canonicalNumber(raw, name);
};

export const bounded = (value: unknown, name: string, upper = 1): number => {
  const raw = finiteRaw(value, name);
  if (raw < 0 || raw > upper) throw new RangeError(name + " must lie within [0, " + upper + "]");
  return canonicalNumber(raw, name);
};

export const integer = (value: unknown, name: string, minimum = 0): number => {
  const raw = finiteRaw(value, name);
  if (!Number.isInteger(raw)) throw new TypeError(name + " must be an integer");
  if (raw < minimum || !Number.isSafeInteger(raw)) {
    throw new RangeError(name + " must lie within the cross-runtime safe range");
  }
  return raw;
};

export const boolean = (value: unknown, name: string): boolean => {
  if (typeof value !== "boolean") throw new TypeError(name + " must be a boolean");
  return value;
};

export const array = (value: unknown, name: string): readonly unknown[] => {
  if (!Array.isArray(value)) throw new TypeError(name + " must be an array");
  return value;
};

export const oneOf = <T extends string>(
  value: unknown,
  values: readonly T[],
  name: string,
): T => {
  const parsed = text(value, name);
  if (!values.includes(parsed as T)) throw new RangeError("invalid " + name + ": " + parsed);
  return parsed as T;
};

export const vector = (value: unknown, name: string): GroundVec3 => {
  const values = array(value, name);
  if (values.length !== 3) throw new RangeError(name + " must contain three components");
  return Object.freeze(values.map((item) => canonicalNumber(item, name))) as unknown as GroundVec3;
};

export function parseProvenance(value: unknown): GroundProvenance {
  const item = record(value, "provenance");
  exact(item, ["input_sha256", "producer", "producer_version", "source_revision"], "provenance");
  const input = text(item.input_sha256, "input_sha256").toLowerCase();
  if (!/^[0-9a-f]{64}$/.test(input)) throw new RangeError("input_sha256 must be 64 lowercase hexadecimal characters");
  return Object.freeze({
    producer: text(item.producer, "producer"),
    producer_version: text(item.producer_version, "producer_version"),
    source_revision: text(item.source_revision, "source_revision"),
    input_sha256: input,
  });
}

export function parseCalibration(value: unknown): GroundCalibration {
  const item = record(value, "calibration");
  exact(item, ["calibration_id", "confidence", "kind", "source"], "calibration");
  return Object.freeze({
    calibration_id: text(item.calibration_id, "calibration_id"),
    kind: oneOf(item.kind, ["measured", "literature", "estimated", "unvalidated"] as const, "calibration kind"),
    source: text(item.source, "calibration source"),
    confidence: bounded(item.confidence, "confidence"),
  });
}

export function parseContactState(value: unknown): GroundContactState {
  const item = record(value, "contact state");
  exact(item, ["angular_velocity_rad_s", "frame", "position_m", "time_s", "velocity_m_s"], "contact state");
  return Object.freeze({
    time_s: nonnegative(item.time_s, "time_s"),
    frame: oneOf(item.frame, [GROUND_TARGET_FRAME] as const, "frame"),
    position_m: vector(item.position_m, "position_m"),
    velocity_m_s: vector(item.velocity_m_s, "velocity_m_s"),
    angular_velocity_rad_s: vector(item.angular_velocity_rad_s, "angular_velocity_rad_s"),
  });
}

const SURFACE_KEYS = [
  "compressibility_fraction", "compression_damping_fraction", "firmness_pa", "frame",
  "grass_height_m", "hardness_fraction", "height_m", "kinetic_friction",
  "moisture_fraction", "normal_restitution", "normal_unit", "provider_id",
  "provider_version", "rolling_resistance", "static_friction", "surface_id",
  "surface_velocity_m_s", "turf_density_kg_m3",
] as const;

export function parseSurface(value: unknown): GroundSurfaceProfile {
  const item = record(value, "surface");
  exact(item, SURFACE_KEYS, "surface");
  const normal = vector(item.normal_unit, "normal_unit");
  const velocity = vector(item.surface_velocity_m_s, "surface_velocity_m_s");
  if (Math.abs(Math.hypot(...normal) - 1) > UNIT_TOLERANCE || normal[1] <= 0) {
    throw new RangeError("normal_unit must be an upward unit vector");
  }
  if (Math.abs(dot(velocity, normal)) > UNIT_TOLERANCE) {
    throw new RangeError("v1 surface_velocity_m_s must be tangential to the plane");
  }
  const staticRaw = finiteRaw(item.static_friction, "static_friction");
  const kineticRaw = finiteRaw(item.kinetic_friction, "kinetic_friction");
  if (kineticRaw > staticRaw) throw new RangeError("kinetic_friction must not exceed static_friction");
  const staticFriction = bounded(staticRaw, "static_friction", 5);
  const kineticFriction = bounded(kineticRaw, "kinetic_friction", 5);
  return Object.freeze({
    surface_id: text(item.surface_id, "surface_id"), provider_id: text(item.provider_id, "provider_id"),
    provider_version: text(item.provider_version, "provider_version"),
    frame: oneOf(item.frame, [GROUND_TARGET_FRAME] as const, "frame"),
    height_m: canonicalNumber(item.height_m, "height_m"), normal_unit: normal,
    surface_velocity_m_s: velocity, normal_restitution: bounded(item.normal_restitution, "normal_restitution"),
    static_friction: staticFriction, kinetic_friction: kineticFriction,
    rolling_resistance: bounded(item.rolling_resistance, "rolling_resistance"),
    firmness_pa: positive(item.firmness_pa, "firmness_pa"),
    hardness_fraction: bounded(item.hardness_fraction, "hardness_fraction"),
    grass_height_m: nonnegative(item.grass_height_m, "grass_height_m"),
    compressibility_fraction: bounded(item.compressibility_fraction, "compressibility_fraction"),
    compression_damping_fraction: bounded(item.compression_damping_fraction, "compression_damping_fraction"),
    turf_density_kg_m3: nonnegative(item.turf_density_kg_m3, "turf_density_kg_m3"),
    moisture_fraction: bounded(item.moisture_fraction, "moisture_fraction"),
  });
}

export const dot = (left: GroundVec3, right: GroundVec3): number =>
  left[0] * right[0] + left[1] * right[1] + left[2] * right[2];

export const groundSignedGapM = (
  state: GroundContactState,
  surface: GroundSurfaceProfile,
  ballRadiusM: number,
): number => dot([
  state.position_m[0],
  state.position_m[1] - surface.height_m,
  state.position_m[2],
], surface.normal_unit) - ballRadiusM;

export const relativeNormalSpeedMps = (
  state: GroundContactState,
  surface: GroundSurfaceProfile,
): number => dot([
  state.velocity_m_s[0] - surface.surface_velocity_m_s[0],
  state.velocity_m_s[1] - surface.surface_velocity_m_s[1],
  state.velocity_m_s[2] - surface.surface_velocity_m_s[2],
], surface.normal_unit);
