/** Strict parser/serializer for regional ground execution evidence; no physics. */

import {
  canonicalGroundJson,
  parseFlightToGroundResult,
} from "./flightGroundContract";
import type {
  FlightToGroundResult,
  GroundProvenance,
  GroundVec3,
} from "./flightGroundTypes";
import {
  array,
  exact,
  integer,
  nonnegative,
  oneOf,
  parseProvenance,
  record,
  text,
  vector,
} from "./flightGroundValidation";
import { sha256Text } from "./sha256";
import { parseUniqueJson } from "./strictJson";
import {
  parseGroundRegionalMaterialPlanRequest,
  type GroundRegionalMaterialPlanRequest,
} from "./groundRegionalPlan";

export const GROUND_REGIONAL_EXECUTION_VERSION =
  "ground-regional-execution-result/v1" as const;
export const GROUND_REGIONAL_EXECUTION_LIMITATIONS = Object.freeze([
  "coplanar_static_surfaces_only",
  "material_changes_only_no_geometry_or_velocity_discontinuities",
] as const);
export const MAX_GROUND_REGIONAL_EXECUTION_WIRE_BYTES = 8_388_608;
export const GROUND_REGIONAL_EXECUTOR_ID = "tools-ground-regional-executor" as const;
export const GROUND_REGIONAL_EXECUTOR_VERSION = "1.0.0" as const;

export type RegionalGroundExecutionStatus =
  | "complete" | "partial" | "cancelled" | "failed";
export type RegionalGroundExecutionFailureReason =
  | "cancelled" | "step_limit" | "surface_transition_limit"
  | "unsupported_surface" | "numerical_failure" | "composition_failure";

export interface RegionalGroundTransition {
  readonly event_sequence: number;
  readonly time_s: number;
  readonly position_m: GroundVec3;
  readonly from_region_id: string | null;
  readonly to_region_id: string | null;
  readonly from_surface_id: string;
  readonly to_surface_id: string;
}

export interface GroundRegionalExecutionResult {
  readonly request_id: string;
  readonly surface_id: string;
  readonly plan_id: string;
  readonly ground_request_sha256: string;
  readonly regional_plan_sha256: string;
  readonly regional_plan: GroundRegionalMaterialPlanRequest;
  readonly status: RegionalGroundExecutionStatus;
  readonly failure_reason: RegionalGroundExecutionFailureReason | null;
  readonly ground_result: FlightToGroundResult | null;
  readonly plan_provenance: GroundProvenance;
  readonly executor_provenance: GroundProvenance;
  readonly model_id: string;
  readonly model_version: string;
  readonly transitions: readonly RegionalGroundTransition[];
  readonly limitations: typeof GROUND_REGIONAL_EXECUTION_LIMITATIONS;
  readonly unit_system: "SI";
  readonly schema_version: typeof GROUND_REGIONAL_EXECUTION_VERSION;
  readonly execution_input_sha256: string;
}

const RESULT_KEYS = [
  "executor_provenance", "failure_reason", "ground_request_sha256",
  "ground_result", "limitations", "model_id", "model_version", "plan_id",
  "plan_provenance", "regional_plan_sha256", "request_id", "schema_version",
  "regional_plan", "status", "surface_id", "transitions", "unit_system",
] as const;
const TRANSITION_KEYS = [
  "event_sequence", "from_region_id", "from_surface_id", "position_m", "time_s",
  "to_region_id", "to_surface_id",
] as const;
const STATUSES = ["complete", "partial", "cancelled", "failed"] as const;
const FAILURE_REASONS = [
  "cancelled", "step_limit", "surface_transition_limit", "unsupported_surface",
  "numerical_failure", "composition_failure",
] as const;

const digest = (value: unknown, name: string): string => {
  const parsed = text(value, name);
  if (!/^[0-9a-f]{64}$/.test(parsed)) {
    throw new RangeError(name + " must be 64 lowercase hexadecimal characters");
  }
  return parsed;
};

const nullableText = (value: unknown, name: string): string | null =>
  value === null ? null : text(value, name);

const limitations = (
  value: unknown,
): typeof GROUND_REGIONAL_EXECUTION_LIMITATIONS => {
  const values = array(value, "limitations");
  if (values.length !== GROUND_REGIONAL_EXECUTION_LIMITATIONS.length ||
    values.some((item, index) =>
      item !== GROUND_REGIONAL_EXECUTION_LIMITATIONS[index])) {
    throw new RangeError("limitations must declare the complete v1 qualification");
  }
  return GROUND_REGIONAL_EXECUTION_LIMITATIONS;
};

const transition = (value: unknown): RegionalGroundTransition => {
  const item = record(value, "regional transition");
  exact(item, TRANSITION_KEYS, "regional transition");
  const fromRegion = nullableText(item.from_region_id, "from_region_id");
  const toRegion = nullableText(item.to_region_id, "to_region_id");
  if (fromRegion === toRegion) {
    throw new RangeError("regional transition must change active regions");
  }
  const fromSurface = text(item.from_surface_id, "from_surface_id");
  const toSurface = text(item.to_surface_id, "to_surface_id");
  if (fromSurface === toSurface) {
    throw new RangeError("transition surface identities must differ");
  }
  return Object.freeze({
    event_sequence: integer(item.event_sequence, "event_sequence"),
    time_s: nonnegative(item.time_s, "time_s"),
    position_m: vector(item.position_m, "position_m"),
    from_region_id: fromRegion,
    to_region_id: toRegion,
    from_surface_id: fromSurface,
    to_surface_id: toSurface,
  });
};

const inputDigest = (ground: string, plan: string): string => sha256Text(
  canonicalGroundJson({
    ground_request_sha256: ground,
    regional_plan_sha256: plan,
  }),
);

type PlanSelection = readonly [string | null, string];
const BOUNDARY_TOLERANCE_M = 1e-9;

const selectionAt = (
  plan: GroundRegionalMaterialPlanRequest,
  coordinate: number,
): PlanSelection => {
  const matches = plan.regions.filter((region) =>
    region.lower_coordinate_m < coordinate && coordinate < region.upper_coordinate_m);
  if (matches.length === 0) return [null, plan.base_surface.surface_id];
  const selected = matches.reduce((left, right) =>
    right.precedence > left.precedence ? right : left);
  return [selected.region_id, selected.surface.surface_id];
};

const boundarySides = (
  plan: GroundRegionalMaterialPlanRequest,
  coordinate: number,
): readonly [PlanSelection, PlanSelection] => {
  const regionBounds = plan.regions.flatMap((region) =>
    [region.lower_coordinate_m, region.upper_coordinate_m]);
  const matching = regionBounds.filter((bound) =>
    Math.abs(bound - coordinate) <= BOUNDARY_TOLERANCE_M);
  if (matching.length === 0) {
    throw new RangeError("transition crossing must lie on a regional plan boundary");
  }
  const boundary = matching.reduce((left, right) =>
    Math.abs(right - coordinate) < Math.abs(left - coordinate) ? right : left);
  const bounds = [...new Set([
    plan.lower_coordinate_m, plan.upper_coordinate_m, ...regionBounds,
  ])].sort((left, right) => left - right);
  const lower = Math.max(...bounds.filter((bound) => bound < boundary));
  const upper = Math.min(...bounds.filter((bound) => bound > boundary));
  return [
    selectionAt(plan, lower + (boundary - lower) / 2),
    selectionAt(plan, boundary + (upper - boundary) / 2),
  ];
};

const sameSelection = (left: PlanSelection, right: PlanSelection): boolean =>
  left[0] === right[0] && left[1] === right[1];

const validatePlanTransition = (
  plan: GroundRegionalMaterialPlanRequest,
  item: RegionalGroundTransition,
  axisVelocity: number,
): void => {
  const offset = item.position_m.map((value, axis) =>
    value - plan.axis_origin_m[axis]) as unknown as GroundVec3;
  const coordinate = offset.reduce((total, value, axis) =>
    total + value * plan.axis_unit[axis], 0);
  const [left, right] = boundarySides(plan, coordinate);
  const from: PlanSelection = [item.from_region_id, item.from_surface_id];
  const to: PlanSelection = [item.to_region_id, item.to_surface_id];
  let expected: readonly [PlanSelection, PlanSelection];
  if (axisVelocity > 0) {
    expected = [left, right];
  } else if (axisVelocity < 0) {
    expected = [right, left];
  } else {
    throw new RangeError("transition direction must be nonzero at a plan crossing");
  }
  if (sameSelection(left, right) ||
    !sameSelection(from, expected[0]) || !sameSelection(to, expected[1])) {
    throw new RangeError("transition identities must match the regional plan crossing");
  }
};

const validateStatus = (result: GroundRegionalExecutionResult): void => {
  const ground = result.ground_result;
  if (ground === null) {
    if (result.failure_reason === null) {
      throw new RangeError("null ground_result requires failure_reason");
    }
    if (result.status === "cancelled") {
      if (result.failure_reason !== "cancelled") {
        throw new RangeError("cancelled status requires cancelled failure_reason");
      }
      return;
    }
    if (result.status !== "failed" || result.failure_reason === "cancelled") {
      throw new RangeError("null ground_result requires failed or cancelled status");
    }
    return;
  }
  if (result.failure_reason !== null ||
    result.status !== ground.status ||
    (ground.status !== "complete" && ground.status !== "partial")) {
    throw new RangeError("execution status must match the embedded ground result");
  }
  if (ground.request_id !== result.request_id || ground.surface_id !== result.surface_id) {
    throw new RangeError("embedded ground result identities must match the envelope");
  }
  if (ground.model_id !== result.model_id || ground.model_version !== result.model_version) {
    throw new RangeError("embedded model identity must match the envelope");
  }
};

const validateLedger = (result: GroundRegionalExecutionResult): void => {
  result.transitions.slice(1).forEach((item, index) => {
    const previous = result.transitions[index];
    if (item.event_sequence <= previous.event_sequence || item.time_s < previous.time_s) {
      throw new RangeError("transition ledger must be strictly ordered");
    }
  });
  if (result.ground_result === null) {
    if (result.transitions.length > 0) {
      throw new RangeError("null ground_result cannot declare transitions");
    }
    return;
  }
  const events = result.ground_result.events.filter(
    ({ event_type }) => event_type === "surface_transition",
  );
  if (events.length !== result.transitions.length || events.some((event, index) => {
    const item = result.transitions[index];
    return event.sequence !== item.event_sequence || event.time_s !== item.time_s ||
      event.position_m.some((component, axis) => component !== item.position_m[axis]);
  })) throw new RangeError("transition ledger must match ground result events");
  events.forEach((event, index) => {
    const velocity = event.velocity_before_m_s.reduce((total, value, axis) =>
      total + value * result.regional_plan.axis_unit[axis], 0);
    validatePlanTransition(result.regional_plan, result.transitions[index], velocity);
  });
};

const validatePlanIdentity = (result: GroundRegionalExecutionResult): void => {
  const plan = result.regional_plan;
  if (sha256Text(canonicalGroundJson(plan)) !== result.regional_plan_sha256) {
    throw new RangeError("regional_plan_sha256 must match the embedded plan");
  }
  if (result.plan_id !== plan.request_id) {
    throw new RangeError("plan_id must match the embedded regional plan");
  }
  if (result.surface_id !== plan.base_surface.surface_id) {
    throw new RangeError("surface_id must match the regional plan base surface");
  }
  if (canonicalGroundJson(result.plan_provenance) !== canonicalGroundJson(plan.provenance)) {
    throw new RangeError("plan provenance must match the embedded regional plan");
  }
};

const executorEvidence = (
  value: unknown,
  groundDigest: string,
  planDigest: string,
): readonly [GroundProvenance, string] => {
  const executor = parseProvenance(value);
  const jointDigest = inputDigest(groundDigest, planDigest);
  if (executor.input_sha256 !== jointDigest) {
    throw new RangeError("executor provenance must match canonical execution inputs");
  }
  if (executor.producer !== GROUND_REGIONAL_EXECUTOR_ID) {
    throw new RangeError("executor producer must match the v1 authority");
  }
  if (executor.producer_version !== GROUND_REGIONAL_EXECUTOR_VERSION) {
    throw new RangeError("executor version must match the v1 authority");
  }
  return [executor, jointDigest];
};

/** Parse and deeply validate one immutable execution envelope. */
export const parseGroundRegionalExecutionResult = (
  value: unknown,
): GroundRegionalExecutionResult => {
  const item = record(value, "regional ground execution result");
  exact(item, RESULT_KEYS, "regional ground execution result");
  const groundDigest = digest(item.ground_request_sha256, "ground_request_sha256");
  const planDigest = digest(item.regional_plan_sha256, "regional_plan_sha256");
  const [executor, jointDigest] = executorEvidence(
    item.executor_provenance, groundDigest, planDigest,
  );
  const failure = item.failure_reason === null ? null : oneOf(
    item.failure_reason,
    FAILURE_REASONS,
    "failure_reason",
  );
  const parsed = {
    request_id: text(item.request_id, "request_id"),
    surface_id: text(item.surface_id, "surface_id"),
    plan_id: text(item.plan_id, "plan_id"),
    ground_request_sha256: groundDigest,
    regional_plan_sha256: planDigest,
    regional_plan: parseGroundRegionalMaterialPlanRequest(item.regional_plan),
    status: oneOf(item.status, STATUSES, "status"),
    failure_reason: failure,
    ground_result: item.ground_result === null
      ? null
      : parseFlightToGroundResult(item.ground_result),
    plan_provenance: parseProvenance(item.plan_provenance),
    executor_provenance: executor,
    model_id: text(item.model_id, "model_id"),
    model_version: text(item.model_version, "model_version"),
    transitions: Object.freeze(array(item.transitions, "transitions").map(transition)),
    limitations: limitations(item.limitations),
    unit_system: oneOf(item.unit_system, ["SI"] as const, "unit_system"),
    schema_version: oneOf(
      item.schema_version,
      [GROUND_REGIONAL_EXECUTION_VERSION] as const,
      "schema_version",
    ),
  } as GroundRegionalExecutionResult;
  Object.defineProperty(parsed, "execution_input_sha256", {
    value: jointDigest,
    enumerable: false,
  });
  validateStatus(parsed);
  validatePlanIdentity(parsed);
  validateLedger(parsed);
  return Object.freeze(parsed);
};

/** Parse bounded JSON with duplicate-key rejection. */
export const groundRegionalExecutionResultFromJson = (
  value: string,
): GroundRegionalExecutionResult => {
  if (typeof value !== "string") {
    throw new TypeError("regional ground execution JSON must be text");
  }
  if (new TextEncoder().encode(value).byteLength >
    MAX_GROUND_REGIONAL_EXECUTION_WIRE_BYTES) {
    throw new RangeError("regional ground execution exceeds maximum wire size");
  }
  return parseGroundRegionalExecutionResult(parseUniqueJson(value));
};

/** Serialize validated evidence with the shared canonical numeric policy. */
export const stableGroundRegionalExecutionJson = (
  value: GroundRegionalExecutionResult,
): string => canonicalGroundJson(value);
