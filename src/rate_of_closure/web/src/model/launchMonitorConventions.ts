/** Immutable, source-backed launch-monitor convention contracts. */

import type { Vec3 } from "./impactPhysics";

export const CONVENTION_IDS = [
  "app_native", "trackman_comparable", "foresight_comparable",
] as const;
export type ConventionId = typeof CONVENTION_IDS[number];

export const PARAMETER_IDS = [
  "club_speed", "club_path", "attack_angle", "face_angle", "dynamic_loft",
  "face_to_path", "spin_loft", "launch_direction",
] as const;
export type ParameterId = typeof PARAMETER_IDS[number];

export const REFERENCE_POINTS = [
  "tracked_head_reference", "geometric_center", "face_center", "impact_location",
  "mixed_club_delivery", "ball_center",
] as const;
type ReferencePoint = typeof REFERENCE_POINTS[number];
export const EVENT_TIMES = [
  "inspection_event", "just_before_first_contact", "impact",
  "maximum_compression", "just_after_separation",
] as const;
type EventTime = typeof EVENT_TIMES[number];
type SignRule = "unspecified" | "nonnegative" | "positive_right" | "positive_up";
type QuantityStatus = "derived" | "modeled" | "measured_comparable";
type AvailabilityRule = "always" | "nonzero_club_travel" | "face_geometry" | "collision_complete";

export const COMPARABILITY_REASON = {
  parameter: "parameter", referencePoint: "reference_point", eventTime: "event_time",
  frame: "frame", geometry: "geometry", signRule: "sign_rule", unit: "unit",
  availability: "availability",
} as const;
type ComparabilityReason = typeof COMPARABILITY_REASON[keyof typeof COMPARABILITY_REASON];

export interface ParameterDefinitionTs {
  readonly conventionId: ConventionId;
  readonly parameterId: ParameterId;
  readonly label: string;
  readonly sourceUrl: string;
  readonly retrievedOn: string;
  readonly referencePoint: ReferencePoint;
  readonly eventTime: EventTime;
  readonly frameId: string;
  readonly geometryContract: string;
  readonly signRule: SignRule;
  readonly unit: string;
  readonly quantityStatus: QuantityStatus;
  readonly availability: AvailabilityRule;
}

export interface ConventionRegistryTs {
  readonly schemaVersion: "launch-monitor-conventions/v1";
  readonly definitions: readonly ParameterDefinitionTs[];
  definition(convention: string, parameter: string): ParameterDefinitionTs;
}

interface Identity { label: string; unit: string; signRule: SignRule; geometryContract: string }
interface Policy {
  referencePoint: ReferencePoint; eventTime: EventTime; quantityStatus: QuantityStatus;
  availability: AvailabilityRule; sourceUrl: string;
  signRule?: SignRule;
}

const FRAME_ID = "target_frame:x_target,y_up,z_right";
const RETRIEVED_ON = "2026-08-05";
const APP_SOURCE = "https://github.com/D-sorganization/Tools/blob/main/docs/specs/D_PLANE_GEOMETRY.md";
const TRACKMAN_CLUB = "https://www.trackman.com/blog/golf/club-data-definitions";
const TRACKMAN_PARAMETERS = "https://www.trackman.com/blog/golf/40-trackman-parameters";
const FORESIGHT_CLUB = "https://help.foresightsports.com/hc/en-us/articles/47214673873811-Club-Head-Data-Measurements-Definitions";
const FORESIGHT_BALL = "https://help.foresightsports.com/hc/en-us/articles/47144162581523-Ball-Launch-Data-Measurements-Ball-Flight-Results";

const IDENTITIES: Record<ParameterId, Identity> = {
  club_speed: { label: "Club Speed", unit: "m/s", signRule: "nonnegative", geometryContract: "magnitude(club_velocity)" },
  club_path: { label: "Club Path", unit: "deg", signRule: "positive_right", geometryContract: "heading(club_velocity)" },
  attack_angle: { label: "Attack Angle", unit: "deg", signRule: "positive_up", geometryContract: "elevation(club_velocity)" },
  face_angle: { label: "Face Angle", unit: "deg", signRule: "positive_right", geometryContract: "heading(face_normal)" },
  dynamic_loft: { label: "Dynamic Loft", unit: "deg", signRule: "positive_up", geometryContract: "elevation(face_normal)" },
  face_to_path: { label: "Face to Path", unit: "deg", signRule: "positive_right", geometryContract: "wrapped(face_angle-club_path)" },
  spin_loft: { label: "Spin Loft", unit: "deg", signRule: "nonnegative", geometryContract: "angle_3d(club_velocity,face_normal)" },
  launch_direction: { label: "Launch Direction", unit: "deg", signRule: "positive_right", geometryContract: "heading(initial_ball_velocity)" },
};

const club = (
  referencePoint: ReferencePoint, eventTime: EventTime,
  quantityStatus: QuantityStatus, sourceUrl: string,
): Policy => ({
  referencePoint, eventTime, quantityStatus, sourceUrl,
  availability: "nonzero_club_travel",
});
const face = (
  referencePoint: ReferencePoint, eventTime: EventTime,
  quantityStatus: QuantityStatus, sourceUrl: string,
): Policy => ({ referencePoint, eventTime, quantityStatus, sourceUrl, availability: "face_geometry" });
const launch = (quantityStatus: QuantityStatus, sourceUrl: string): Policy => ({
  referencePoint: "ball_center", eventTime: "just_after_separation", quantityStatus,
  sourceUrl, availability: "collision_complete",
});

const APP_POLICIES: Record<ParameterId, Policy> = {
  club_speed: club("tracked_head_reference", "inspection_event", "derived", APP_SOURCE),
  club_path: club("tracked_head_reference", "inspection_event", "derived", APP_SOURCE),
  attack_angle: club("tracked_head_reference", "inspection_event", "derived", APP_SOURCE),
  face_angle: face("face_center", "inspection_event", "derived", APP_SOURCE),
  dynamic_loft: face("face_center", "inspection_event", "derived", APP_SOURCE),
  face_to_path: face("mixed_club_delivery", "inspection_event", "derived", APP_SOURCE),
  spin_loft: face("mixed_club_delivery", "inspection_event", "derived", APP_SOURCE),
  launch_direction: launch("modeled", APP_SOURCE),
};
const TRACKMAN_POLICIES: Record<ParameterId, Policy> = {
  club_speed: club("geometric_center", "just_before_first_contact", "measured_comparable", TRACKMAN_PARAMETERS),
  club_path: club("geometric_center", "maximum_compression", "measured_comparable", TRACKMAN_CLUB),
  attack_angle: club("geometric_center", "maximum_compression", "measured_comparable", TRACKMAN_CLUB),
  face_angle: face("impact_location", "maximum_compression", "measured_comparable", TRACKMAN_CLUB),
  dynamic_loft: face("impact_location", "maximum_compression", "measured_comparable", TRACKMAN_CLUB),
  face_to_path: face("mixed_club_delivery", "maximum_compression", "derived", TRACKMAN_PARAMETERS),
  spin_loft: face("mixed_club_delivery", "maximum_compression", "derived", TRACKMAN_PARAMETERS),
  launch_direction: launch("measured_comparable", TRACKMAN_PARAMETERS),
};
const FORESIGHT_POLICIES: Record<ParameterId, Policy> = {
  club_speed: club("face_center", "just_before_first_contact", "measured_comparable", FORESIGHT_CLUB),
  club_path: club("face_center", "impact", "measured_comparable", FORESIGHT_CLUB),
  attack_angle: club("face_center", "impact", "measured_comparable", FORESIGHT_CLUB),
  face_angle: face("impact_location", "impact", "measured_comparable", FORESIGHT_CLUB),
  dynamic_loft: face("impact_location", "impact", "measured_comparable", FORESIGHT_CLUB),
  face_to_path: face("mixed_club_delivery", "impact", "derived", FORESIGHT_CLUB),
  spin_loft: face("mixed_club_delivery", "impact", "derived", FORESIGHT_CLUB),
  launch_direction: {
    ...launch("measured_comparable", FORESIGHT_BALL),
    signRule: "unspecified",
  },
};
const POLICIES: Record<ConventionId, Record<ParameterId, Policy>> = {
  app_native: APP_POLICIES,
  trackman_comparable: TRACKMAN_POLICIES,
  foresight_comparable: FORESIGHT_POLICIES,
};

const buildRegistry = (): ConventionRegistryTs => {
  const definitions = CONVENTION_IDS.flatMap((conventionId) =>
    PARAMETER_IDS.map((parameterId) => Object.freeze({
      conventionId, parameterId, ...IDENTITIES[parameterId], ...POLICIES[conventionId][parameterId],
      retrievedOn: RETRIEVED_ON, frameId: FRAME_ID,
    }))).sort((left, right) => `${left.conventionId}.${left.parameterId}`
      .localeCompare(`${right.conventionId}.${right.parameterId}`));
  const frozen = Object.freeze(definitions);
  return Object.freeze({
    schemaVersion: "launch-monitor-conventions/v1" as const,
    definitions: frozen,
    definition: (convention: string, parameter: string) => {
      const found = frozen.find((item) =>
        item.conventionId === convention && item.parameterId === parameter);
      if (!found) throw new RangeError(`unknown convention parameter: ${convention}.${parameter}`);
      return found;
    },
  });
};

let cachedRegistry: ConventionRegistryTs | null = null;
export const conventionRegistry = (): ConventionRegistryTs => {
  cachedRegistry ??= buildRegistry();
  return cachedRegistry;
};

export function compareDefinitions(first: ParameterDefinitionTs, second: ParameterDefinitionTs) {
  const checks: Array<[ComparabilityReason, unknown, unknown]> = [
    [COMPARABILITY_REASON.parameter, first.parameterId, second.parameterId],
    [COMPARABILITY_REASON.referencePoint, first.referencePoint, second.referencePoint],
    [COMPARABILITY_REASON.eventTime, first.eventTime, second.eventTime],
    [COMPARABILITY_REASON.frame, first.frameId, second.frameId],
    [COMPARABILITY_REASON.geometry, first.geometryContract, second.geometryContract],
    [COMPARABILITY_REASON.signRule, first.signRule, second.signRule],
    [COMPARABILITY_REASON.unit, first.unit, second.unit],
    [COMPARABILITY_REASON.availability, first.availability, second.availability],
  ];
  const reasons = checks.filter(([, left, right]) => left !== right).map(([reason]) => reason);
  return Object.freeze({ comparable: reasons.length === 0, reasons: Object.freeze(reasons) });
}

const vector = (value: Vec3, name: string): Vec3 => {
  if (value.length !== 3 || value.some((component) => !Number.isFinite(component))) {
    throw new RangeError(`${name} must contain three finite components`);
  }
  return value;
};
export function shiftPointVelocity(reference: Vec3, angular: Vec3, offset: Vec3): Vec3 {
  const [vx, vy, vz] = vector(reference, "reference");
  const [wx, wy, wz] = vector(angular, "angular");
  const [rx, ry, rz] = vector(offset, "offset");
  return [vx + wy * rz - wz * ry, vy + wz * rx - wx * rz, vz + wx * ry - wy * rx];
}

export function transformVector(value: Vec3, rotation: [Vec3, Vec3, Vec3]): Vec3 {
  const input = vector(value, "value");
  const rows = rotation.map((row) => vector(row, "rotation row"));
  const dot = (a: Vec3, b: Vec3) => a.reduce((sum, item, index) => sum + item * b[index], 0);
  const validRows = rows.every((row, index) => rows.every((other, otherIndex) =>
    Math.abs(dot(row, other) - (index === otherIndex ? 1 : 0)) <= 1e-10));
  const determinant = rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
    - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
    + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0]);
  if (!validRows || Math.abs(determinant - 1) > 1e-10) {
    throw new RangeError("rotation must be a proper orthonormal matrix");
  }
  return rows.map((row) => dot(row, input)) as Vec3;
}

const toWire = (definition: ParameterDefinitionTs) => ({
  convention_id: definition.conventionId, parameter_id: definition.parameterId,
  label: definition.label, source_url: definition.sourceUrl, retrieved_on: definition.retrievedOn,
  reference_point: definition.referencePoint, event_time: definition.eventTime,
  frame_id: definition.frameId, geometry_contract: definition.geometryContract,
  sign_rule: definition.signRule, unit: definition.unit,
  quantity_status: definition.quantityStatus, availability: definition.availability,
});
const stable = (value: unknown): unknown => Array.isArray(value) ? value.map(stable)
  : value && typeof value === "object" ? Object.fromEntries(Object.entries(value)
    .sort(([first], [second]) => first.localeCompare(second))
    .map(([key, item]) => [key, stable(item)])) : value;
export const stableConventionJson = (registry: ConventionRegistryTs): string => JSON.stringify(stable({
  definitions: registry.definitions.map(toWire), schema_version: registry.schemaVersion,
}));

export function migrateConventionRegistry(payload: unknown): ConventionRegistryTs {
  if (!payload || typeof payload !== "object") throw new RangeError("registry must be an object");
  const raw = payload as { schema_version?: unknown; definitions?: unknown };
  if (raw.schema_version !== "launch-monitor-conventions/v1" &&
      raw.schema_version !== "launch-monitor-conventions/v0") {
    throw new RangeError("unsupported convention registry schema");
  }
  if (!Array.isArray(raw.definitions)) throw new RangeError("definitions must be an array");
  const canonical = JSON.parse(stableConventionJson(conventionRegistry()));
  const migrated = raw.definitions.map((item) => {
    if (!item || typeof item !== "object") throw new RangeError("definition must be an object");
    const record = { ...(item as Record<string, unknown>) };
    if ("vendor" in record) { record.convention_id = record.vendor; delete record.vendor; }
    return record;
  });
  if (JSON.stringify(stable(migrated)) !== JSON.stringify(stable(canonical.definitions))) {
    throw new RangeError("definition fields or values do not match the supported v1 catalog");
  }
  return conventionRegistry();
}
