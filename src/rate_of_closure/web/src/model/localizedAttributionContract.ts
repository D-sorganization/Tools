/** Shared strict bounds, target meanings, and numerical policy. */

export type RecordValue = Record<string, unknown>;

export const MAX_TEXT_LENGTH = 256;
export const MAX_SOURCES = 32;
export const MAX_TARGETS = 64;
export const MAX_PAIRS = 4096;
export const MAX_OBSERVATIONS = 131072;

interface TargetDefinition {
  kind: "state" | "impact" | "shot";
  unit: string;
  convention: string;
  coordinateFrame: string | null;
}

const state = (unit = "m"): TargetDefinition => ({
  kind: "state", unit, convention: "app-frame-cartesian-v1",
  coordinateFrame: "app_frame:x_target,y_up,z_right",
});
const impact = (unit: string): TargetDefinition => ({
  kind: "impact", unit, convention: "rate-of-closure-impact-v1",
  coordinateFrame: null,
});
const shot = (unit: string): TargetDefinition => ({
  kind: "shot", unit, convention: "rate-of-closure-flight-v1",
  coordinateFrame: null,
});

export const TARGET_REGISTRY: Readonly<Record<string, TargetDefinition>> = {
  position_x_m: state(), position_y_m: state(), position_z_m: state(),
  impact_time_s: impact("s"), clubhead_speed_mps: impact("m/s"),
  spin_loft_deg: impact("deg"), face_to_path_deg: impact("deg"),
  spin_axis_tilt_deg: impact("deg"), ball_speed_mph: shot("mph"),
  launch_angle_deg: shot("deg"), launch_azimuth_deg: shot("deg"),
  spin_rpm: shot("rpm"), carry_m: shot("m"), lateral_m: shot("m"),
  max_height_m: shot("m"), flight_time_s: shot("s"),
  landing_angle_deg: shot("deg"),
};

export const record = (
  value: unknown, fields: readonly string[], label: string,
): RecordValue => {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  const result = value as RecordValue;
  if (Object.keys(result).sort().join("|") !== [...fields].sort().join("|")) {
    throw new Error(`${label} has invalid fields`);
  }
  return result;
};

export const stable = (value: unknown, label: string): string => {
  const hasControl = typeof value === "string" && [...value].some(
    (character) => character.charCodeAt(0) < 32,
  );
  if (typeof value !== "string" || value.length === 0 || value.trim() !== value ||
      [...value].length > MAX_TEXT_LENGTH || hasControl || /^[=+\-@]/u.test(value)) {
    throw new Error(`${label} must be a stable safe ID within the length cap`);
  }
  return value;
};

export const finite = (value: unknown, label: string): number => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${label} must be finite`);
  }
  return value;
};

export const nullableFinite = (value: unknown, label: string): number | null =>
  value === null ? null : finite(value, label);

export const index = (value: unknown, label: string): number => {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) {
    throw new Error(`${label} must be a nonnegative safe integer`);
  }
  return value;
};

export const boundedArray = (
  value: unknown, maximum: number, label: string,
): readonly unknown[] => {
  if (!Array.isArray(value) || value.length > maximum) {
    throw new Error(`${label} must be an array within its resource cap`);
  }
  return value;
};

export const requireAuthorityShape = (
  sources: number, targets: number, pairs: number, observations: number,
): void => {
  if (sources > MAX_SOURCES || targets > MAX_TARGETS || pairs > MAX_PAIRS ||
      observations > MAX_OBSERVATIONS || pairs * targets > MAX_OBSERVATIONS) {
    throw new Error("authority arrays or pair-target matrix exceed resource cap");
  }
};

export const responseMatches = (actual: number, expected: number): boolean => {
  const tolerance = 4 * Number.EPSILON * Math.max(1, Math.abs(expected));
  return Number.isFinite(actual) && Number.isFinite(expected) &&
    Number.isFinite(tolerance) && Math.abs(actual - expected) <= tolerance;
};

export const requireNonzeroIntervention = (baseline: number, perturbed: number): void => {
  if (baseline === perturbed) throw new Error("source intervention delta must be nonzero");
};

export const deepFreeze = <T>(value: T): T => {
  if (typeof value === "object" && value !== null && !Object.isFrozen(value)) {
    Object.values(value as Record<string, unknown>).forEach(deepFreeze);
    Object.freeze(value);
  }
  return value;
};
