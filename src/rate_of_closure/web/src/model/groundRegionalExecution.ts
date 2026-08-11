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

export const GROUND_REGIONAL_EXECUTION_VERSION =
  "ground-regional-execution-result/v1" as const;
export const GROUND_REGIONAL_EXECUTION_LIMITATIONS = Object.freeze([
  "coplanar_static_surfaces_only",
  "material_changes_only_no_geometry_or_velocity_discontinuities",
] as const);
export const MAX_GROUND_REGIONAL_EXECUTION_WIRE_BYTES = 8_388_608;

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
  "status", "surface_id", "transitions", "unit_system",
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
  return Object.freeze({
    event_sequence: integer(item.event_sequence, "event_sequence"),
    time_s: nonnegative(item.time_s, "time_s"),
    position_m: vector(item.position_m, "position_m"),
    from_region_id: fromRegion,
    to_region_id: toRegion,
    from_surface_id: text(item.from_surface_id, "from_surface_id"),
    to_surface_id: text(item.to_surface_id, "to_surface_id"),
  });
};

const inputDigest = (ground: string, plan: string): string => sha256Text(
  canonicalGroundJson({
    ground_request_sha256: ground,
    regional_plan_sha256: plan,
  }),
);

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
  if (result.ground_result === null) return;
  const events = result.ground_result.events.filter(
    ({ event_type }) => event_type === "surface_transition",
  );
  if (events.length !== result.transitions.length || events.some((event, index) => {
    const item = result.transitions[index];
    return event.sequence !== item.event_sequence || event.time_s !== item.time_s ||
      event.position_m.some((component, axis) => component !== item.position_m[axis]);
  })) throw new RangeError("transition ledger must match ground result events");
};

/** Parse and deeply validate one immutable execution envelope. */
export const parseGroundRegionalExecutionResult = (
  value: unknown,
): GroundRegionalExecutionResult => {
  const item = record(value, "regional ground execution result");
  exact(item, RESULT_KEYS, "regional ground execution result");
  const groundDigest = digest(item.ground_request_sha256, "ground_request_sha256");
  const planDigest = digest(item.regional_plan_sha256, "regional_plan_sha256");
  const executor = parseProvenance(item.executor_provenance);
  const jointDigest = inputDigest(groundDigest, planDigest);
  if (executor.input_sha256 !== jointDigest) {
    throw new RangeError("executor provenance must match canonical execution inputs");
  }
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
