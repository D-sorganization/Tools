/** Strict result parser for flight-to-ground-result/v1. */

import {
  FLIGHT_TO_GROUND_RESULT_VERSION,
  GROUND_TARGET_FRAME,
  type FlightToGroundResult,
  type GroundEvent,
  type GroundEventType,
  type GroundPhase,
  type GroundSummary,
  type GroundTermination,
  type GroundTrajectoryPoint,
  type GroundUnavailableField,
  type GroundWarning,
} from "./flightGroundTypes";
import {
  array,
  boolean,
  canonicalNumber,
  exact,
  integer,
  nonnegative,
  oneOf,
  parseCalibration,
  parseContactState,
  parseProvenance,
  record,
  text,
  vector,
} from "./flightGroundValidation";

const POINT_KEYS = [
  "angular_velocity_rad_s", "frame", "phase", "position_m", "time_s", "velocity_m_s",
] as const;
const EVENT_KEYS = [
  "angular_velocity_after_rad_s", "angular_velocity_before_rad_s", "event_type",
  "frame", "position_m", "sequence", "time_s", "velocity_after_m_s",
  "velocity_before_m_s",
] as const;
const RESULT_KEYS = [
  "calibration", "events", "frame", "model_id", "model_version", "provenance",
  "request_id", "schema_version", "status", "summary", "surface_id", "termination",
  "trajectory", "unavailable_fields", "unit_system", "warnings",
] as const;

export const parseGroundTrajectoryPoint = (value: unknown): GroundTrajectoryPoint => {
  const item = record(value, "trajectory point");
  exact(item, POINT_KEYS, "trajectory point");
  const state = parseContactState({
    angular_velocity_rad_s: item.angular_velocity_rad_s,
    frame: item.frame,
    position_m: item.position_m,
    time_s: item.time_s,
    velocity_m_s: item.velocity_m_s,
  });
  const phase = oneOf(item.phase, ["impact", "bounce", "skid", "roll", "rest"] as const, "phase");
  if (phase === "rest" && [...state.velocity_m_s, ...state.angular_velocity_rad_s].some(
    (component) => Math.abs(component) > 1e-9,
  )) throw new RangeError("rest phase requires zero linear and angular velocity");
  return Object.freeze({ ...state, phase });
};

export const parseGroundEvent = (value: unknown): GroundEvent => {
  const item = record(value, "ground event");
  exact(item, EVENT_KEYS, "ground event");
  const event: GroundEvent = Object.freeze({
    sequence: integer(item.sequence, "event sequence"),
    event_type: oneOf(
      item.event_type,
      [
        "first_contact",
        "bounce",
        "skid_to_roll",
        "surface_transition",
        "rest",
        "left_surface",
      ] as const,
      "event_type",
    ),
    time_s: nonnegative(item.time_s, "event time_s"),
    frame: oneOf(item.frame, [GROUND_TARGET_FRAME] as const, "frame"),
    position_m: vector(item.position_m, "position_m"),
    velocity_before_m_s: vector(item.velocity_before_m_s, "velocity_before_m_s"),
    velocity_after_m_s: vector(item.velocity_after_m_s, "velocity_after_m_s"),
    angular_velocity_before_rad_s: vector(
      item.angular_velocity_before_rad_s,
      "angular_velocity_before_rad_s",
    ),
    angular_velocity_after_rad_s: vector(
      item.angular_velocity_after_rad_s,
      "angular_velocity_after_rad_s",
    ),
  });
  if (event.event_type === "rest" && [
    ...event.velocity_after_m_s,
    ...event.angular_velocity_after_rad_s,
  ].some((component) => Math.abs(component) > 1e-9)) {
    throw new RangeError("rest event requires zero output velocity and spin");
  }
  return event;
};

const parseSummary = (value: unknown): GroundSummary => {
  const item = record(value, "ground summary");
  exact(item, [
    "bounce_air_distance_m", "bounce_count", "carry_distance_m", "final_downrange_m",
    "final_offline_m", "roll_distance_m", "skid_distance_m",
    "surface_path_distance_m", "total_distance_m",
  ], "ground summary");
  return Object.freeze({
    carry_distance_m: nonnegative(item.carry_distance_m, "carry_distance_m"),
    bounce_air_distance_m: nonnegative(item.bounce_air_distance_m, "bounce_air_distance_m"),
    skid_distance_m: nonnegative(item.skid_distance_m, "skid_distance_m"),
    roll_distance_m: nonnegative(item.roll_distance_m, "roll_distance_m"),
    surface_path_distance_m: nonnegative(item.surface_path_distance_m, "surface_path_distance_m"),
    total_distance_m: nonnegative(item.total_distance_m, "total_distance_m"),
    final_downrange_m: canonicalNumber(item.final_downrange_m, "final_downrange_m"),
    final_offline_m: canonicalNumber(item.final_offline_m, "final_offline_m"),
    bounce_count: integer(item.bounce_count, "bounce_count"),
  });
};

const parseTermination = (value: unknown): GroundTermination => {
  const item = record(value, "termination");
  exact(item, ["completed", "reason", "time_s"], "termination");
  const reason = oneOf(item.reason, [
    "rest", "time_limit", "event_limit", "left_surface", "numerical_failure",
    "unavailable_input",
  ] as const, "termination reason");
  const completed = boolean(item.completed, "completed");
  if (completed !== ["rest", "left_surface"].includes(reason)) {
    throw new RangeError("completed does not match termination reason");
  }
  return Object.freeze({ reason, completed, time_s: nonnegative(item.time_s, "termination time_s") });
};

const parseWarning = (value: unknown): GroundWarning => {
  const item = record(value, "warning");
  exact(item, ["code", "message", "severity"], "warning");
  return Object.freeze({
    code: text(item.code, "warning code"),
    message: text(item.message, "warning message"),
    severity: oneOf(item.severity, ["info", "warning", "error"] as const, "warning severity"),
  });
};

const parseUnavailable = (value: unknown): GroundUnavailableField => {
  const item = record(value, "unavailable field");
  exact(item, ["field_id", "provenance", "reason"], "unavailable field");
  return Object.freeze({
    field_id: oneOf(item.field_id, [
      "terminal_angular_velocity_rad_s", "physical_contact_bracket", "surface_profile",
    ] as const, "unavailable field_id"),
    reason: oneOf(item.reason, [
      "source_does_not_propagate", "no_physical_contact", "unsupported_surface",
      "source_out_of_bounds",
    ] as const, "unavailable reason"),
    provenance: text(item.provenance, "unavailable field provenance"),
  });
};

const PHASE_TRANSITIONS: Readonly<Record<GroundPhase, readonly GroundPhase[]>> = {
  impact: ["impact", "bounce", "skid", "roll", "rest"],
  bounce: ["bounce", "skid", "roll", "rest"],
  skid: ["skid", "roll", "rest"],
  roll: ["roll", "rest"],
  rest: ["rest"],
};
const EVENT_TRANSITIONS: Readonly<Record<GroundEventType, readonly GroundEventType[]>> = {
  first_contact: ["bounce", "skid_to_roll", "surface_transition", "rest", "left_surface"],
  bounce: ["bounce", "skid_to_roll", "surface_transition", "rest", "left_surface"],
  skid_to_roll: ["surface_transition", "rest", "left_surface"],
  surface_transition: ["surface_transition", "skid_to_roll", "rest", "left_surface"],
  rest: [],
  left_surface: [],
};
const close = (left: number, right: number): boolean =>
  Math.abs(left - right) <= Math.max(1e-8, 1e-10 * Math.max(Math.abs(left), Math.abs(right)));
const vectorsClose = (left: readonly number[], right: readonly number[]): boolean =>
  left.every((value, index) => close(value, right[index]));

const validateOrdering = (
  points: readonly GroundTrajectoryPoint[],
  events: readonly GroundEvent[],
): void => {
  if (!points.length) throw new RangeError("trajectory must be nonempty");
  points.slice(1).forEach((point, index) => {
    const previous = points[index];
    if (point.time_s <= previous.time_s) throw new RangeError("trajectory times must increase");
    if (!PHASE_TRANSITIONS[previous.phase].includes(point.phase)) {
      throw new RangeError("invalid ground phase transition");
    }
  });
  events.forEach((event, index) => {
    if (event.sequence !== index) throw new RangeError("event sequence must be contiguous from zero");
    if (index && event.time_s < events[index - 1].time_s) throw new RangeError("event times must not decrease");
    if (index && !EVENT_TRANSITIONS[events[index - 1].event_type].includes(event.event_type)) {
      throw new RangeError("invalid ground event transition");
    }
  });
};

const validateSummary = (
  summary: GroundSummary,
  points: readonly GroundTrajectoryPoint[],
  events: readonly GroundEvent[],
): void => {
  const first = points[0].position_m;
  const final = points[points.length - 1].position_m;
  const expected = [Math.hypot(first[0], first[2]), final[0], final[2]];
  const actual = [summary.carry_distance_m, summary.final_downrange_m, summary.final_offline_m];
  if (!vectorsClose(actual, expected)) throw new RangeError("summary displacement metrics must match trajectory geometry");
  if (!close(summary.total_distance_m, Math.hypot(final[0], final[2]))) {
    throw new RangeError("summary total distance must match final horizontal position");
  }
  if (!close(summary.surface_path_distance_m, summary.skid_distance_m + summary.roll_distance_m)) {
    throw new RangeError("surface path must equal skid plus roll distance");
  }
  if (summary.bounce_count !== events.filter(({ event_type }) => event_type === "bounce").length) {
    throw new RangeError("summary bounce_count must match bounce events");
  }
};

const validateFirstContact = (
  point: GroundTrajectoryPoint,
  event: GroundEvent | undefined,
): void => {
  if (!event || event.event_type !== "first_contact" || point.phase !== "impact") {
    throw new RangeError("ground output must begin with first_contact impact");
  }
  if (!close(point.time_s, event.time_s) || !vectorsClose(point.position_m, event.position_m)
    || !vectorsClose(point.velocity_m_s, event.velocity_after_m_s)
    || !vectorsClose(point.angular_velocity_rad_s, event.angular_velocity_after_rad_s)) {
    throw new RangeError("first_contact must match the initial trajectory point");
  }
};

const STATUS_REASONS = {
  complete: ["rest", "left_surface"],
  partial: ["time_limit", "event_limit"],
  failed: ["numerical_failure"],
  unavailable: ["unavailable_input"],
} as const;

const validateTerminal = (result: FlightToGroundResult): void => {
  const point = result.trajectory[result.trajectory.length - 1];
  const event = result.events[result.events.length - 1];
  if (!close(result.termination.time_s, point.time_s)) {
    throw new RangeError("termination time_s must match the final trajectory point");
  }
  if (result.events.some(({ time_s }) =>
    time_s < result.trajectory[0].time_s || time_s > point.time_s)) {
    throw new RangeError("event times must lie within the trajectory interval");
  }
  if (result.status === "partial") {
    if (point.phase === "rest" || ["rest", "left_surface"].includes(event.event_type)) {
      throw new RangeError("partial result cannot contain a terminal state");
    }
    return;
  }
  const expected = result.termination.reason === "rest" ? "rest" : "left_surface";
  if (event.event_type !== expected || !close(event.time_s, result.termination.time_s)
    || !vectorsClose(event.position_m, point.position_m)
    || !vectorsClose(event.velocity_after_m_s, point.velocity_m_s)
    || !vectorsClose(event.angular_velocity_after_rad_s, point.angular_velocity_rad_s)) {
    throw new RangeError("completed termination must match the terminal state");
  }
  if ((expected === "rest") !== (point.phase === "rest")) {
    throw new RangeError("terminal event and final phase do not agree");
  }
};

function validateResult(result: FlightToGroundResult): void {
  if (!(STATUS_REASONS[result.status] as readonly string[]).includes(result.termination.reason)) {
    throw new RangeError("result status is incompatible with termination reason");
  }
  const hasUnavailable = result.unavailable_fields.length > 0;
  const unavailableIds = result.unavailable_fields.map(({ field_id }) => field_id);
  if (new Set(unavailableIds).size !== unavailableIds.length) {
    throw new RangeError("unavailable field IDs must be unique");
  }
  if ((result.status === "unavailable") !== hasUnavailable) {
    throw new RangeError("only unavailable results require unavailable_fields");
  }
  if (result.status === "failed" || result.status === "unavailable") {
    if (result.trajectory.length || result.events.length || result.summary || result.termination.completed) {
      throw new RangeError("failed or unavailable results cannot contain fabricated output");
    }
    return;
  }
  validateOrdering(result.trajectory, result.events);
  validateFirstContact(result.trajectory[0], result.events[0]);
  if (!result.summary) throw new RangeError("complete or partial results require a summary");
  validateSummary(result.summary, result.trajectory, result.events);
  validateTerminal(result);
}

/** Parse and freeze one exact flight-to-ground result record. */
export function parseFlightToGroundResultRecord(payload: unknown): FlightToGroundResult {
  const item = record(payload, "ground simulation result");
  exact(item, RESULT_KEYS, "ground simulation result");
  const result: FlightToGroundResult = Object.freeze({
    schema_version: oneOf(item.schema_version, [FLIGHT_TO_GROUND_RESULT_VERSION] as const, "schema_version"),
    request_id: text(item.request_id, "request_id"), unit_system: oneOf(item.unit_system, ["SI"] as const, "unit_system"),
    frame: oneOf(item.frame, [GROUND_TARGET_FRAME] as const, "frame"),
    surface_id: text(item.surface_id, "surface_id"), model_id: text(item.model_id, "model_id"),
    model_version: text(item.model_version, "model_version"),
    status: oneOf(item.status, ["complete", "partial", "failed", "unavailable"] as const, "status"),
    trajectory: Object.freeze(array(item.trajectory, "trajectory").map(parseGroundTrajectoryPoint)),
    events: Object.freeze(array(item.events, "events").map(parseGroundEvent)),
    summary: item.summary === null ? null : parseSummary(item.summary),
    termination: parseTermination(item.termination), calibration: parseCalibration(item.calibration),
    warnings: Object.freeze(array(item.warnings, "warnings").map(parseWarning)),
    unavailable_fields: Object.freeze(array(item.unavailable_fields, "unavailable_fields").map(parseUnavailable)),
    provenance: parseProvenance(item.provenance),
  });
  validateResult(result);
  return result;
}
