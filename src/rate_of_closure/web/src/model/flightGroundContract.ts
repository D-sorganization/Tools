/** Public strict JSON facade for the flight-to-ground v1 boundary. */

import { parseUniqueJson } from "./strictJson";
import { parseFlightToGroundRequestRecord } from "./flightGroundRequestContract";
import { parseFlightToGroundResultRecord } from "./flightGroundResultContract";
import type { FlightToGroundRequest, FlightToGroundResult } from "./flightGroundTypes";

const rejectSurrogates = (value: string, name: string): void => {
  if (/[\uD800-\uDFFF]/.test(value)) throw new RangeError(name + " must not contain surrogate code points");
};

const numberToken = (value: number): string => {
  if (!Number.isFinite(value)) throw new TypeError("canonical JSON numbers must be finite");
  if (Math.abs(value) > Number.MAX_SAFE_INTEGER) {
    throw new RangeError("canonical JSON numbers must lie within the cross-runtime safe range");
  }
  const fixed = value.toFixed(11);
  const normalized = fixed.replace(/\.?0+$/, "");
  return normalized === "" || normalized === "-0" ? "0" : normalized;
};

const canonicalToken = (value: unknown): string => {
  if (value === null) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") return numberToken(value);
  if (typeof value === "string") {
    rejectSurrogates(value, "canonical JSON string");
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) return "[" + value.map(canonicalToken).join(",") + "]";
  if (!value || typeof value !== "object") throw new TypeError("unsupported canonical JSON value");
  const entries = Object.entries(value as Record<string, unknown>).sort(([left], [right]) =>
    left < right ? -1 : left > right ? 1 : 0
  );
  return "{" + entries.map(([key, item]) => {
    rejectSurrogates(key, "canonical JSON key");
    return JSON.stringify(key) + ":" + canonicalToken(item);
  }).join(",") + "}";
};

/** Serialize JSON-compatible data using the shared 11-decimal numeric policy. */
export const canonicalNumericJson = (value: unknown): string => canonicalToken(value);

/** Preserve the public flight-ground facade name for existing consumers. */
export const canonicalGroundJson = canonicalNumericJson;

/** Parse and validate one request mapping. */
export const parseFlightToGroundRequest = (payload: unknown): FlightToGroundRequest =>
  parseFlightToGroundRequestRecord(payload);

/** Parse and validate one result mapping. */
export const parseFlightToGroundResult = (payload: unknown): FlightToGroundResult =>
  parseFlightToGroundResultRecord(payload);

/** Parse a strict request document, including duplicate-key rejection. */
export const flightToGroundRequestFromJson = (text: string): FlightToGroundRequest =>
  parseFlightToGroundRequest(parseUniqueJson(text));

/** Parse a strict result document, including duplicate-key rejection. */
export const flightToGroundResultFromJson = (text: string): FlightToGroundResult =>
  parseFlightToGroundResult(parseUniqueJson(text));

export const stableFlightToGroundRequestJson = (request: FlightToGroundRequest): string =>
  canonicalGroundJson(request);

export const stableFlightToGroundResultJson = (result: FlightToGroundResult): string =>
  canonicalGroundJson(result);

export * from "./flightGroundTypes";
