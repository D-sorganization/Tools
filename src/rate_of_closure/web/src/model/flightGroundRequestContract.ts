/** Strict request parser for flight-to-ground-request/v1. */

import {
  FLIGHT_TO_GROUND_REQUEST_VERSION,
  type FlightToGroundRequest,
} from "./flightGroundTypes";
import {
  exact,
  finiteRaw,
  groundSignedGapM,
  integer,
  oneOf,
  parseCalibration,
  parseContactState,
  parseProvenance,
  parseSurface,
  positive,
  record,
  relativeNormalSpeedMps,
  text,
} from "./flightGroundValidation";

const REQUEST_KEYS = [
  "ball_mass_kg", "ball_radius_m", "calibration", "first_penetrating_state",
  "last_separated_state", "max_events", "max_time_s", "output_interval_s",
  "provenance", "request_id", "rotational_inertia_factor", "schema_version",
  "surface", "unit_system",
] as const;

const validateBracket = (request: FlightToGroundRequest): void => {
  const separated = request.last_separated_state;
  const penetrating = request.first_penetrating_state;
  if (penetrating.time_s <= separated.time_s) {
    throw new RangeError("contact bracket times must be strictly increasing");
  }
  const firstGap = groundSignedGapM(separated, request.surface, request.ball_radius_m);
  const secondGap = groundSignedGapM(penetrating, request.surface, request.ball_radius_m);
  if (firstGap <= 0 || secondGap > 0) {
    throw new RangeError("contact states must straddle the physical sphere surface");
  }
  const speeds = [
    relativeNormalSpeedMps(separated, request.surface),
    relativeNormalSpeedMps(penetrating, request.surface),
  ];
  if (speeds.some((speed) => speed >= -1e-12)) {
    throw new RangeError("both bracket states require incoming relative normal velocity");
  }
};

/** Parse and freeze one exact flight-to-ground request record. */
export function parseFlightToGroundRequestRecord(payload: unknown): FlightToGroundRequest {
  const item = record(payload, "ground simulation request");
  exact(item, REQUEST_KEYS, "ground simulation request");
  const maxTimeRaw = finiteRaw(item.max_time_s, "max_time_s");
  const outputIntervalRaw = finiteRaw(item.output_interval_s, "output_interval_s");
  const inertiaRaw = finiteRaw(item.rotational_inertia_factor, "rotational_inertia_factor");
  if (inertiaRaw > 1) {
    throw new RangeError("rotational_inertia_factor must lie within (0, 1]");
  }
  if (outputIntervalRaw > maxTimeRaw) {
    throw new RangeError("output_interval_s must not exceed max_time_s");
  }
  const maxTimeS = positive(item.max_time_s, "max_time_s");
  const outputIntervalS = positive(item.output_interval_s, "output_interval_s");
  const rotationalInertiaFactor = positive(
    item.rotational_inertia_factor,
    "rotational_inertia_factor",
  );
  const request: FlightToGroundRequest = Object.freeze({
    schema_version: oneOf(
      item.schema_version,
      [FLIGHT_TO_GROUND_REQUEST_VERSION] as const,
      "schema_version",
    ),
    request_id: text(item.request_id, "request_id"),
    unit_system: oneOf(item.unit_system, ["SI"] as const, "unit_system"),
    surface: parseSurface(item.surface),
    last_separated_state: parseContactState(item.last_separated_state),
    first_penetrating_state: parseContactState(item.first_penetrating_state),
    ball_radius_m: positive(item.ball_radius_m, "ball_radius_m"),
    ball_mass_kg: positive(item.ball_mass_kg, "ball_mass_kg"),
    rotational_inertia_factor: rotationalInertiaFactor,
    max_time_s: maxTimeS,
    output_interval_s: outputIntervalS,
    max_events: integer(item.max_events, "max_events", 1),
    calibration: parseCalibration(item.calibration),
    provenance: parseProvenance(item.provenance),
  });
  validateBracket(request);
  return request;
}
