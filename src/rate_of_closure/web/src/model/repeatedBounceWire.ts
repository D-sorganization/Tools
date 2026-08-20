/** Import-only repeated-bounce prefix evidence contract; no browser physics. */

import { canonicalGroundJson } from "./flightGroundContract";
import {
  parseGroundEvent,
  parseGroundTrajectoryPoint,
} from "./flightGroundResultContract";
import type {
  GroundContactState,
  GroundEvent,
  GroundFrame,
  GroundTrajectoryPoint,
  GroundVec3,
} from "./flightGroundTypes";
import {
  array,
  boolean,
  bounded,
  canonicalNumber,
  exact,
  nonnegative,
  oneOf,
  parseContactState,
  positive,
  record,
  text,
  vector,
} from "./flightGroundValidation";
import { parseUniqueJson } from "./strictJson";

export const REPEATED_BOUNCE_SCHEMA_VERSION =
  "ground-repeated-bounce-result/v1" as const;
export const MAX_REPEATED_BOUNCE_WIRE_BYTES = 1_048_576;
export const REPEATED_BOUNCE_EVIDENCE_ABSOLUTE_TOLERANCE = 1e-10;
export const REPEATED_BOUNCE_EVIDENCE_RELATIVE_TOLERANCE = 1e-10;
const ENERGY_ABSOLUTE_TOLERANCE_J = 1e-10;
const ENERGY_RELATIVE_TOLERANCE = 1e-10;

export type ImpactRegime = "sticking" | "sliding";
export type BounceTerminationReason =
  | "settled_to_skid" | "cancelled" | "time_limit" | "event_limit"
  | "no_recontact" | "numerical_failure";

export interface ImpactEnergyLedger {
  readonly kinetic_before_j: number;
  readonly kinetic_after_j: number;
  readonly boundary_work_j: number;
  readonly dissipation_j: number;
}

export interface ImpactImpulseResult {
  readonly state_before: GroundContactState;
  readonly state_after: GroundContactState;
  readonly regime: ImpactRegime;
  readonly normal_impulse_n_s: number;
  readonly tangential_impulse_n_s: GroundVec3;
  readonly total_impulse_n_s: GroundVec3;
  readonly contact_velocity_before_m_s: GroundVec3;
  readonly contact_velocity_after_m_s: GroundVec3;
  readonly effective_restitution: number;
  readonly friction_utilization: number;
  readonly energy: ImpactEnergyLedger;
}

export interface BounceAirSegment {
  readonly start_time_s: number;
  readonly end_time_s: number;
  readonly start_position_m: GroundVec3;
  readonly end_position_m: GroundVec3;
  readonly horizontal_distance_m: number;
  readonly completed_at_contact: boolean;
}

export interface BounceTermination {
  readonly reason: BounceTerminationReason;
  readonly time_s: number;
  readonly elapsed_time_s: number;
}

export interface RepeatedBounceResult {
  readonly schema_version: typeof REPEATED_BOUNCE_SCHEMA_VERSION;
  readonly unit_system: "SI";
  readonly request_id: string;
  readonly surface_id: string;
  readonly frame: GroundFrame;
  readonly model_id: string;
  readonly model_version: string;
  readonly request_fingerprint_sha256: string;
  readonly trajectory: readonly GroundTrajectoryPoint[];
  readonly events: readonly GroundEvent[];
  readonly impacts: readonly ImpactImpulseResult[];
  readonly airborne_segments: readonly BounceAirSegment[];
  readonly handoff_state: GroundContactState | null;
  readonly termination: BounceTermination;
  readonly warnings: readonly string[];
}

const RESULT_KEYS = [
  "airborne_segments", "events", "frame", "handoff_state", "impacts", "model_id",
  "model_version", "request_fingerprint_sha256", "request_id", "schema_version",
  "surface_id", "termination", "trajectory", "unit_system", "warnings",
] as const;
const IMPACT_KEYS = [
  "contact_velocity_after_m_s", "contact_velocity_before_m_s", "effective_restitution",
  "energy", "friction_utilization", "normal_impulse_n_s", "regime", "state_after",
  "state_before", "tangential_impulse_n_s", "total_impulse_n_s",
] as const;

const close = (left: number, right: number): boolean =>
  Math.abs(left - right) <= Math.max(
    REPEATED_BOUNCE_EVIDENCE_ABSOLUTE_TOLERANCE,
    REPEATED_BOUNCE_EVIDENCE_RELATIVE_TOLERANCE * Math.max(Math.abs(left), Math.abs(right)),
  );
const vectorsClose = (left: readonly number[], right: readonly number[]): boolean =>
  left.every((component, index) => close(component, right[index]));
const energyBalanceClose = (
  actual: number,
  expected: number,
  inputs: readonly number[],
): boolean => Math.abs(actual - expected) <= ENERGY_ABSOLUTE_TOLERANCE_J +
  ENERGY_RELATIVE_TOLERANCE * Math.max(...inputs.map(Math.abs), Math.abs(actual));

const parseEnergy = (value: unknown): ImpactEnergyLedger => {
  const item = record(value, "impact energy");
  exact(item, [
    "boundary_work_j", "dissipation_j", "kinetic_after_j", "kinetic_before_j",
  ], "impact energy");
  const energy = Object.freeze({
    kinetic_before_j: nonnegative(item.kinetic_before_j, "kinetic_before_j"),
    kinetic_after_j: nonnegative(item.kinetic_after_j, "kinetic_after_j"),
    boundary_work_j: canonicalNumber(item.boundary_work_j, "boundary_work_j"),
    dissipation_j: nonnegative(item.dissipation_j, "dissipation_j"),
  });
  const expectedDissipation = energy.kinetic_before_j + energy.boundary_work_j -
    energy.kinetic_after_j;
  if (!energyBalanceClose(energy.dissipation_j, expectedDissipation, [
    energy.kinetic_before_j,
    energy.kinetic_after_j,
    energy.boundary_work_j,
  ])) {
    throw new RangeError("impact dissipation must match the energy balance");
  }
  return energy;
};

const parseImpact = (value: unknown): ImpactImpulseResult => {
  const item = record(value, "impact result");
  exact(item, IMPACT_KEYS, "impact result");
  const before = parseContactState(item.state_before);
  const after = parseContactState(item.state_after);
  if (before.frame !== after.frame || before.time_s !== after.time_s) {
    throw new RangeError("impact state frame and time must match");
  }
  return Object.freeze({
    state_before: before,
    state_after: after,
    regime: oneOf(item.regime, ["sticking", "sliding"] as const, "impact regime"),
    normal_impulse_n_s: positive(item.normal_impulse_n_s, "normal_impulse_n_s"),
    tangential_impulse_n_s: vector(item.tangential_impulse_n_s, "tangential_impulse_n_s"),
    total_impulse_n_s: vector(item.total_impulse_n_s, "total_impulse_n_s"),
    contact_velocity_before_m_s: vector(
      item.contact_velocity_before_m_s, "contact_velocity_before_m_s",
    ),
    contact_velocity_after_m_s: vector(
      item.contact_velocity_after_m_s, "contact_velocity_after_m_s",
    ),
    effective_restitution: bounded(item.effective_restitution, "effective_restitution"),
    friction_utilization: bounded(item.friction_utilization, "friction_utilization"),
    energy: parseEnergy(item.energy),
  });
};

const parseAirSegment = (value: unknown): BounceAirSegment => {
  const item = record(value, "airborne segment");
  exact(item, [
    "completed_at_contact", "end_position_m", "end_time_s", "horizontal_distance_m",
    "start_position_m", "start_time_s",
  ], "airborne segment");
  const start = vector(item.start_position_m, "start_position_m");
  const end = vector(item.end_position_m, "end_position_m");
  const startTime = nonnegative(item.start_time_s, "start_time_s");
  const endTime = nonnegative(item.end_time_s, "end_time_s");
  const distance = nonnegative(item.horizontal_distance_m, "horizontal_distance_m");
  if (endTime <= startTime || !close(distance, Math.hypot(
    end[0] - start[0], end[2] - start[2],
  ))) throw new RangeError("airborne segment time or distance is inconsistent");
  return Object.freeze({
    start_time_s: startTime, end_time_s: endTime, start_position_m: start,
    end_position_m: end, horizontal_distance_m: distance,
    completed_at_contact: boolean(item.completed_at_contact, "completed_at_contact"),
  });
};

const parseTermination = (value: unknown): BounceTermination => {
  const item = record(value, "bounce termination");
  exact(item, ["elapsed_time_s", "reason", "time_s"], "bounce termination");
  return Object.freeze({
    reason: oneOf(item.reason, [
      "settled_to_skid", "cancelled", "time_limit", "event_limit", "no_recontact",
      "numerical_failure",
    ] as const, "bounce termination reason"),
    time_s: nonnegative(item.time_s, "termination time_s"),
    elapsed_time_s: nonnegative(item.elapsed_time_s, "elapsed_time_s"),
  });
};

const eventMatchesImpact = (event: GroundEvent, impact: ImpactImpulseResult): boolean =>
  close(event.time_s, impact.state_before.time_s) &&
  vectorsClose(event.position_m, impact.state_before.position_m) &&
  vectorsClose(event.velocity_before_m_s, impact.state_before.velocity_m_s) &&
  vectorsClose(event.velocity_after_m_s, impact.state_after.velocity_m_s) &&
  vectorsClose(event.angular_velocity_before_rad_s, impact.state_before.angular_velocity_rad_s) &&
  vectorsClose(event.angular_velocity_after_rad_s, impact.state_after.angular_velocity_rad_s);

const validateLedger = (result: RepeatedBounceResult): void => {
  if (result.events.length !== result.impacts.length) {
    throw new RangeError("each bounce event requires one impact result");
  }
  result.events.forEach((event, index) => {
    if (event.sequence !== index || event.frame !== result.frame) {
      throw new RangeError("bounce event sequence and frame must match result");
    }
    if ((index === 0 && event.event_type !== "first_contact") ||
      (index > 0 && event.event_type !== "bounce")) {
      throw new RangeError("bounce event types are inconsistent");
    }
    if (index > 0 && event.time_s < result.events[index - 1].time_s) {
      throw new RangeError("bounce event times must be nondecreasing");
    }
    if (!eventMatchesImpact(event, result.impacts[index])) {
      throw new RangeError("bounce event states must match their impact result");
    }
  });
};

const stateMatchesPoint = (
  state: GroundContactState,
  point: GroundTrajectoryPoint,
): boolean => close(state.time_s, point.time_s) && state.frame === point.frame &&
  vectorsClose(state.position_m, point.position_m) &&
  vectorsClose(state.velocity_m_s, point.velocity_m_s) &&
  vectorsClose(state.angular_velocity_rad_s, point.angular_velocity_rad_s);

const validateTrajectoryAndHandoff = (result: RepeatedBounceResult): void => {
  result.trajectory.forEach((point, index) => {
    if (point.frame !== result.frame || !["impact", "bounce", "skid"].includes(point.phase)) {
      throw new RangeError("bounce trajectory frame or phase is inconsistent");
    }
    if (index > 0 && point.time_s <= result.trajectory[index - 1].time_s) {
      throw new RangeError("bounce trajectory times must be strictly increasing");
    }
    if (point.phase === "skid" && index !== result.trajectory.length - 1) {
      throw new RangeError("only the terminal bounce point may be skid");
    }
  });
  if (result.events.length > 0 && result.trajectory.length === 0) {
    throw new RangeError("bounce events require trajectory evidence");
  }
  result.events.forEach((event, index) => {
    const impact = result.impacts[index];
    const aligned = result.trajectory.filter((point) => close(point.time_s, event.time_s));
    if (!impact || aligned.length !== 1 || !stateMatchesPoint(impact.state_after, aligned[0])) {
      throw new RangeError(
        "each bounce event requires one matching post-impact trajectory point",
      );
    }
    const expectedPhase = impact.effective_restitution === 0 ? "skid" :
      index === 0 ? "impact" : "bounce";
    if (aligned[0].phase !== expectedPhase) {
      throw new RangeError("event-aligned trajectory phase is inconsistent");
    }
  });
  if (result.termination.reason === "settled_to_skid" && result.handoff_state === null) {
    throw new RangeError("settled bounce prefix requires a handoff state");
  }
  if (result.handoff_state !== null) {
    const finalPoint = result.trajectory[result.trajectory.length - 1];
    const finalImpact = result.impacts[result.impacts.length - 1];
    if (result.termination.reason !== "settled_to_skid" || !finalPoint || !finalImpact ||
      finalImpact.effective_restitution !== 0 || finalPoint.phase !== "skid" ||
      !stateMatchesPoint(result.handoff_state, finalPoint)) {
      throw new RangeError("handoff state must match the settled terminal skid point");
    }
  }
};

const validateTermination = (result: RepeatedBounceResult): void => {
  const finalPoint = result.trajectory[result.trajectory.length - 1];
  if (finalPoint && !close(result.termination.time_s, finalPoint.time_s)) {
    throw new RangeError("termination time must match the final trajectory point");
  }
  const firstEvent = result.events[0];
  const expectedElapsed = firstEvent ? result.termination.time_s - firstEvent.time_s : 0;
  if (expectedElapsed < 0 || !close(result.termination.elapsed_time_s, expectedElapsed)) {
    throw new RangeError("termination elapsed time must match bounce chronology");
  }
};

const validateSegments = (result: RepeatedBounceResult): void => {
  const completed = result.airborne_segments.filter(({ completed_at_contact }) =>
    completed_at_contact).length;
  const partial = result.airborne_segments.length - completed;
  if (completed !== Math.max(0, result.events.length - 1) || partial > 1 ||
    (result.termination.reason === "time_limit" && partial !== 1)) {
    throw new RangeError("airborne segment ledger does not match bounce events");
  }
  result.airborne_segments.forEach((segment, index) => {
    const start = result.events[index];
    if (!start || !close(segment.start_time_s, start.time_s) ||
      !vectorsClose(segment.start_position_m, start.position_m)) {
      throw new RangeError("airborne segment start must match its impact event");
    }
    if (segment.completed_at_contact) {
      const end = result.events[index + 1];
      if (!end || !close(segment.end_time_s, end.time_s) ||
        !vectorsClose(segment.end_position_m, end.position_m)) {
        throw new RangeError("completed segment end must match contact event");
      }
    } else if (index !== result.airborne_segments.length - 1) {
      throw new RangeError("only the final airborne segment may be partial");
    } else {
      const finalPoint = result.trajectory[result.trajectory.length - 1];
      if (!close(segment.end_time_s, result.termination.time_s) || !finalPoint ||
        !vectorsClose(segment.end_position_m, finalPoint.position_m)) {
        throw new RangeError("partial segment end must match termination evidence");
      }
    }
  });
};

/** Parse and deeply validate exact immutable repeated-bounce evidence. */
export const parseRepeatedBounceResult = (value: unknown): RepeatedBounceResult => {
  const item = record(value, "repeated bounce result");
  exact(item, RESULT_KEYS, "repeated bounce result");
  const fingerprint = text(item.request_fingerprint_sha256, "request fingerprint");
  if (!/^[0-9a-f]{64}$/.test(fingerprint)) {
    throw new RangeError("request fingerprint must be 64 lowercase hexadecimal characters");
  }
  const result: RepeatedBounceResult = Object.freeze({
    schema_version: oneOf(item.schema_version, [REPEATED_BOUNCE_SCHEMA_VERSION] as const, "schema_version"),
    unit_system: oneOf(item.unit_system, ["SI"] as const, "unit_system"),
    request_id: text(item.request_id, "request_id"),
    surface_id: text(item.surface_id, "surface_id"),
    frame: oneOf(item.frame, ["target_frame:x_downrange,y_up,z_right"] as const, "frame"),
    model_id: text(item.model_id, "model_id"), model_version: text(item.model_version, "model_version"),
    request_fingerprint_sha256: fingerprint,
    trajectory: Object.freeze(array(item.trajectory, "trajectory").map(parseGroundTrajectoryPoint)),
    events: Object.freeze(array(item.events, "events").map(parseGroundEvent)),
    impacts: Object.freeze(array(item.impacts, "impacts").map(parseImpact)),
    airborne_segments: Object.freeze(array(item.airborne_segments, "airborne_segments").map(parseAirSegment)),
    handoff_state: item.handoff_state === null ? null : parseContactState(item.handoff_state),
    termination: parseTermination(item.termination),
    warnings: Object.freeze(array(item.warnings, "warnings").map((warning) => text(warning, "warning"))),
  });
  validateLedger(result);
  validateTrajectoryAndHandoff(result);
  validateTermination(result);
  validateSegments(result);
  return result;
};

/** Parse bounded strict JSON with duplicate-key rejection. */
export const repeatedBounceResultFromJson = (value: string): RepeatedBounceResult => {
  if (typeof value !== "string") throw new TypeError("repeated bounce JSON must be text");
  if (new TextEncoder().encode(value).byteLength > MAX_REPEATED_BOUNCE_WIRE_BYTES) {
    throw new RangeError("repeated bounce evidence exceeds maximum wire size");
  }
  return parseRepeatedBounceResult(parseUniqueJson(value));
};

/** Serialize validated imported evidence with the shared canonical numeric policy. */
export const stableRepeatedBounceResultJson = (value: unknown): string => {
  const text = canonicalGroundJson(parseRepeatedBounceResult(value));
  if (new TextEncoder().encode(text).byteLength > MAX_REPEATED_BOUNCE_WIRE_BYTES) {
    throw new RangeError("repeated bounce evidence exceeds maximum wire size");
  }
  return text;
};
