/** Import-only repeated-bounce execution request and result pairing contract. */

import {
  canonicalGroundJson,
  parseFlightToGroundRequest,
  stableFlightToGroundRequestJson,
} from "./flightGroundContract";
import type {
  FlightToGroundRequest,
  GroundFrame,
} from "./flightGroundTypes";
import {
  exact,
  oneOf,
  positive,
  record,
  text,
} from "./flightGroundValidation";
import {
  parseRepeatedBounceResult,
  type RepeatedBounceResult,
} from "./repeatedBounceWire";
import { sha256Text } from "./sha256";
import { parseUniqueJson } from "./strictJson";

export const REPEATED_BOUNCE_REQUEST_SCHEMA_VERSION =
  "ground-repeated-bounce-request/v1" as const;
export const MAX_REPEATED_BOUNCE_REQUEST_WIRE_BYTES = 1_048_576;
export const REPEATED_BOUNCE_MODEL_ID = "tools-ground-impact-bounce" as const;
export const REPEATED_BOUNCE_MODEL_VERSION = "1.0.0" as const;

export interface RepeatedBounceRequest {
  readonly schema_version: typeof REPEATED_BOUNCE_REQUEST_SCHEMA_VERSION;
  readonly unit_system: "SI";
  readonly frame: GroundFrame;
  readonly request_id: string;
  readonly surface_id: string;
  readonly model_id: typeof REPEATED_BOUNCE_MODEL_ID;
  readonly model_version: typeof REPEATED_BOUNCE_MODEL_VERSION;
  readonly capture_speed_m_s: number;
  readonly ground_request_sha256: string;
  readonly execution_input_sha256: string;
  readonly ground_request: FlightToGroundRequest;
}

export interface RepeatedBounceRequestResultPair {
  readonly request: RepeatedBounceRequest;
  readonly result: RepeatedBounceResult;
  readonly execution_input_sha256: string;
}

const REQUEST_KEYS = [
  "capture_speed_m_s", "execution_input_sha256", "frame", "ground_request",
  "ground_request_sha256", "model_id", "model_version", "request_id",
  "schema_version", "surface_id", "unit_system",
] as const;

const digest = (value: unknown, name: string): string => {
  const parsed = text(value, name);
  if (!/^[0-9a-f]{64}$/.test(parsed)) {
    throw new RangeError(name + " must be 64 lowercase hexadecimal characters");
  }
  return parsed;
};

const executionInputDigest = (
  groundRequestSha256: string,
  captureSpeedMps: number,
): string => sha256Text(canonicalGroundJson({
  capture_speed_m_s: captureSpeedMps,
  ground_request_sha256: groundRequestSha256,
  model_id: REPEATED_BOUNCE_MODEL_ID,
  model_version: REPEATED_BOUNCE_MODEL_VERSION,
  schema_version: REPEATED_BOUNCE_REQUEST_SCHEMA_VERSION,
}));

const validateEmbeddedIdentity = (request: RepeatedBounceRequest): void => {
  const ground = request.ground_request;
  const expectedGroundDigest = sha256Text(stableFlightToGroundRequestJson(ground));
  if (request.request_id !== ground.request_id) {
    throw new RangeError("request_id must match the embedded request authority");
  }
  if (request.surface_id !== ground.surface.surface_id) {
    throw new RangeError("surface_id must match the embedded request authority");
  }
  if (request.frame !== ground.surface.frame) {
    throw new RangeError("frame must match the embedded request authority");
  }
  if (request.ground_request_sha256 !== expectedGroundDigest) {
    throw new RangeError("ground_request_sha256 must match the embedded request authority");
  }
  if (request.execution_input_sha256 !== executionInputDigest(
    expectedGroundDigest,
    request.capture_speed_m_s,
  )) {
    throw new RangeError("execution_input_sha256 must match the request authority");
  }
};

/** Parse and deeply validate one request without executing physics. */
export const parseRepeatedBounceRequest = (value: unknown): RepeatedBounceRequest => {
  const item = record(value, "repeated bounce request");
  exact(item, REQUEST_KEYS, "repeated bounce request");
  const request: RepeatedBounceRequest = Object.freeze({
    schema_version: oneOf(
      item.schema_version,
      [REPEATED_BOUNCE_REQUEST_SCHEMA_VERSION] as const,
      "schema_version",
    ),
    unit_system: oneOf(item.unit_system, ["SI"] as const, "unit_system"),
    frame: oneOf(
      item.frame,
      ["target_frame:x_downrange,y_up,z_right"] as const,
      "frame",
    ),
    request_id: text(item.request_id, "request_id"),
    surface_id: text(item.surface_id, "surface_id"),
    model_id: oneOf(item.model_id, [REPEATED_BOUNCE_MODEL_ID] as const, "model_id"),
    model_version: oneOf(
      item.model_version,
      [REPEATED_BOUNCE_MODEL_VERSION] as const,
      "model_version",
    ),
    capture_speed_m_s: positive(item.capture_speed_m_s, "capture_speed_m_s"),
    ground_request_sha256: digest(item.ground_request_sha256, "ground_request_sha256"),
    execution_input_sha256: digest(item.execution_input_sha256, "execution_input_sha256"),
    ground_request: parseFlightToGroundRequest(item.ground_request),
  });
  validateEmbeddedIdentity(request);
  return request;
};

/** Parse bounded strict JSON with duplicate-key rejection. */
export const repeatedBounceRequestFromJson = (value: string): RepeatedBounceRequest => {
  if (typeof value !== "string") {
    throw new TypeError("repeated bounce request JSON must be text");
  }
  if (new TextEncoder().encode(value).byteLength >
    MAX_REPEATED_BOUNCE_REQUEST_WIRE_BYTES) {
    throw new RangeError("repeated bounce request exceeds maximum wire size");
  }
  return parseRepeatedBounceRequest(parseUniqueJson(value));
};

/** Serialize one validated request with the shared canonical numeric policy. */
export const stableRepeatedBounceRequestJson = (value: unknown): string => {
  const serialized = canonicalGroundJson(parseRepeatedBounceRequest(value));
  if (new TextEncoder().encode(serialized).byteLength >
    MAX_REPEATED_BOUNCE_REQUEST_WIRE_BYTES) {
    throw new RangeError("repeated bounce request exceeds maximum wire size");
  }
  return serialized;
};

/** Pair imported request/result evidence only when every v1 identity agrees. */
export const pairRepeatedBounceRequestResult = (
  requestValue: unknown,
  resultValue: unknown,
): RepeatedBounceRequestResultPair => {
  const request = parseRepeatedBounceRequest(requestValue);
  const result = parseRepeatedBounceResult(resultValue);
  if (result.request_id !== request.request_id) {
    throw new RangeError("result request identity must match the request");
  }
  if (result.surface_id !== request.surface_id) {
    throw new RangeError("result surface identity must match the request");
  }
  if (result.frame !== request.frame) {
    throw new RangeError("result frame identity must match the request");
  }
  if (result.model_id !== request.model_id || result.model_version !== request.model_version) {
    throw new RangeError("result model identity must match the request");
  }
  if (result.request_fingerprint_sha256 !== request.ground_request_sha256) {
    throw new RangeError("result request fingerprint must match the request");
  }
  return Object.freeze({
    request,
    result,
    execution_input_sha256: request.execution_input_sha256,
  });
};
