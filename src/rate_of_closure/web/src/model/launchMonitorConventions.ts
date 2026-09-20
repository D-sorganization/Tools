/** Immutable, source-backed launch-monitor convention contracts. */

import type { Vec3 } from "./impactPhysics";
import {
  APP_POLICIES,
  FORESIGHT_POLICIES,
  FRAME_ID,
  GROUP_LABELS,
  IDENTITIES,
  PARAMETER_GROUP_MAP,
  RETRIEVED_ON,
  TRACKMAN_POLICIES,
} from "./launchMonitorConventionsCatalog";

export const CONVENTION_IDS = [
  "app_native", "trackman_comparable", "foresight_comparable",
] as const;
export type ConventionId = typeof CONVENTION_IDS[number];

export const PARAMETER_GROUPS = [
  "club_delivery", "face_orientation", "ball_launch", "ball_spin", "ball_flight",
] as const;
export type ParameterGroup = typeof PARAMETER_GROUPS[number];

export const PARAMETER_IDS = [
  "club_speed", "club_path", "attack_angle", "dynamic_lie", "closure_rate", "swing_direction", "low_point",
  "face_angle", "dynamic_loft", "face_to_path", "spin_loft", "impact_offset", "impact_height",
  "ball_speed", "launch_angle", "launch_direction", "smash_factor",
  "total_spin", "spin_axis", "back_spin", "side_spin",
  "apex_height", "carry_distance", "total_distance", "carry_offline", "curve", "flight_time", "landing_angle",
] as const;
export type ParameterId = typeof PARAMETER_IDS[number];

export const REFERENCE_POINTS = [
  "tracked_head_reference", "geometric_center", "face_center", "impact_location",
  "mixed_club_delivery", "ball_center",
] as const;
export type ReferencePoint = typeof REFERENCE_POINTS[number];

export const EVENT_TIMES = [
  "inspection_event", "just_before_first_contact", "impact",
  "maximum_compression", "just_after_separation",
  "apex", "landing", "flight_duration",
] as const;
export type EventTime = typeof EVENT_TIMES[number];

export type SignRule = "unspecified" | "nonnegative" | "positive_right" | "positive_up";
export type QuantityStatus =
  | "derived" | "modeled" | "measured_comparable"
  | "approximated" | "flight_model_dependent" | "unavailable";
export type AvailabilityRule =
  | "always" | "nonzero_club_travel" | "face_geometry"
  | "collision_complete" | "trajectory_complete" | "unavailable";

export const COMPARABILITY_REASON = {
  parameter: "parameter", referencePoint: "reference_point", eventTime: "event_time",
  frame: "frame", geometry: "geometry", signRule: "sign_rule", unit: "unit",
  availability: "availability",
} as const;
export type ComparabilityReason = typeof COMPARABILITY_REASON[keyof typeof COMPARABILITY_REASON];

export interface ParameterDefinitionTs {
  readonly conventionId: ConventionId;
  readonly parameterId: ParameterId;
  readonly group: ParameterGroup;
  readonly label: string;
  readonly definition: string;
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

export const parameterGroup = (parameterId: ParameterId): ParameterGroup => PARAMETER_GROUP_MAP[parameterId];
export const parameterGroupLabel = (group: ParameterGroup): string => GROUP_LABELS[group];

const buildRegistry = (): ConventionRegistryTs => {
  const policies = {
    app_native: APP_POLICIES,
    trackman_comparable: TRACKMAN_POLICIES,
    foresight_comparable: FORESIGHT_POLICIES,
  };
  const definitions = CONVENTION_IDS.flatMap((conventionId) =>
    PARAMETER_IDS.map((parameterId) => Object.freeze({
      conventionId,
      parameterId,
      group: PARAMETER_GROUP_MAP[parameterId],
      ...IDENTITIES[parameterId],
      ...policies[conventionId][parameterId],
      retrievedOn: RETRIEVED_ON,
      frameId: FRAME_ID,
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
