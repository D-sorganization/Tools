/** Strict deterministic wire contract for coplanar regional material plans. */

import { canonicalGroundJson } from "./flightGroundContract";
import type {
  GroundProvenance,
  GroundSurfaceProfile,
  GroundVec3,
} from "./flightGroundTypes";
import {
  canonicalNumber,
  exact,
  integer,
  parseProvenance,
  parseSurface,
  record,
  text,
  vector,
} from "./flightGroundValidation";
import { sha256Text } from "./sha256";
import { parseUniqueJson } from "./strictJson";

export const GROUND_REGIONAL_PLAN_REQUEST_VERSION =
  "ground-regional-material-plan-request/v1" as const;
export const GROUND_REGIONAL_PLAN_RESULT_VERSION =
  "ground-regional-material-plan-result/v1" as const;
export const GROUND_REGIONAL_PLAN_GEOMETRY_MODEL =
  "coplanar_static_material_overlays" as const;
export const GROUND_REGIONAL_PLAN_LIMITATIONS = Object.freeze([
  "coplanar_static_surfaces_only",
  "material_changes_only_no_geometry_or_velocity_discontinuities",
] as const);
export const MAX_GROUND_REGIONAL_PLAN_REGIONS = 4_096;
export const MAX_GROUND_REGIONAL_PLAN_WIRE_BYTES = 1_048_576;

const UNIT_TOLERANCE = 1e-10;
const GEOMETRY_TOLERANCE = 1e-10;
const ZERO: GroundVec3 = [0, 0, 0];

export interface GroundRegionalMaterialRegion {
  readonly region_id: string;
  readonly precedence: number;
  readonly lower_coordinate_m: number;
  readonly upper_coordinate_m: number;
  readonly surface: GroundSurfaceProfile;
}

export interface GroundRegionalMaterialPlanRequest {
  readonly request_id: string;
  readonly base_surface: GroundSurfaceProfile;
  readonly axis_origin_m: GroundVec3;
  readonly axis_unit: GroundVec3;
  readonly lower_coordinate_m: number;
  readonly upper_coordinate_m: number;
  readonly regions: readonly GroundRegionalMaterialRegion[];
  readonly provenance: GroundProvenance;
  readonly geometry_model: typeof GROUND_REGIONAL_PLAN_GEOMETRY_MODEL;
  readonly limitations: typeof GROUND_REGIONAL_PLAN_LIMITATIONS;
  readonly unit_system: "SI";
  readonly schema_version: typeof GROUND_REGIONAL_PLAN_REQUEST_VERSION;
}

export interface GroundRegionalMaterialPlanResult {
  readonly request: GroundRegionalMaterialPlanRequest;
  readonly request_sha256: string;
  readonly ordered_regions: readonly GroundRegionalMaterialRegion[];
  readonly provenance: GroundProvenance;
  readonly limitations: typeof GROUND_REGIONAL_PLAN_LIMITATIONS;
  readonly unit_system: "SI";
  readonly schema_version: typeof GROUND_REGIONAL_PLAN_RESULT_VERSION;
}

const requestKeys = [
  "axis_origin_m", "axis_unit", "base_surface", "geometry_model", "limitations",
  "lower_coordinate_m", "provenance", "regions", "request_id", "schema_version",
  "unit_system", "upper_coordinate_m",
] as const;
const regionKeys = [
  "lower_coordinate_m", "precedence", "region_id", "surface", "upper_coordinate_m",
] as const;
const resultKeys = [
  "limitations", "ordered_regions", "provenance", "request", "request_sha256",
  "schema_version", "unit_system",
] as const;

const array = (value: unknown, name: string): readonly unknown[] => {
  if (!Array.isArray(value)) throw new TypeError(name + " must be an array");
  return value;
};

const fixedText = <T extends string>(value: unknown, expected: T, name: string): T => {
  if (value !== expected) throw new RangeError("unsupported " + name + ": " + String(value));
  return expected;
};

const fixedLimitations = (value: unknown): typeof GROUND_REGIONAL_PLAN_LIMITATIONS => {
  const values = array(value, "limitations");
  if (values.length !== GROUND_REGIONAL_PLAN_LIMITATIONS.length ||
    values.some((item, index) => item !== GROUND_REGIONAL_PLAN_LIMITATIONS[index])) {
    throw new RangeError("limitations must declare the complete v1 qualification");
  }
  return GROUND_REGIONAL_PLAN_LIMITATIONS;
};

const parseDigest = (value: unknown, name: string): string => {
  const digest = text(value, name).toLowerCase();
  if (!/^[0-9a-f]{64}$/.test(digest)) {
    throw new RangeError(name + " must be 64 lowercase hexadecimal characters");
  }
  return digest;
};

const equalVector = (left: GroundVec3, right: GroundVec3): boolean =>
  left.every((value, index) => value === right[index]);

const dot = (left: GroundVec3, right: GroundVec3): number =>
  left[0] * right[0] + left[1] * right[1] + left[2] * right[2];

const parseRegion = (value: unknown): GroundRegionalMaterialRegion => {
  const item = record(value, "regional material");
  exact(item, regionKeys, "regional material");
  const lower = canonicalNumber(item.lower_coordinate_m, "lower_coordinate_m");
  const upper = canonicalNumber(item.upper_coordinate_m, "upper_coordinate_m");
  if (lower >= upper) {
    throw new RangeError("lower_coordinate_m must be below upper_coordinate_m");
  }
  return Object.freeze({
    region_id: text(item.region_id, "region_id"),
    precedence: integer(item.precedence, "precedence"),
    lower_coordinate_m: lower,
    upper_coordinate_m: upper,
    surface: parseSurface(item.surface),
  });
};

const validateAxis = (request: GroundRegionalMaterialPlanRequest): void => {
  if (Math.abs(Math.hypot(...request.axis_unit) - 1) > UNIT_TOLERANCE) {
    throw new RangeError("axis_unit must be a unit vector");
  }
  if (Math.abs(dot(request.axis_unit, request.base_surface.normal_unit)) > UNIT_TOLERANCE) {
    throw new RangeError("axis_unit must be tangent to the base plane");
  }
  const planeOffset: GroundVec3 = [
    request.axis_origin_m[0],
    request.axis_origin_m[1] - request.base_surface.height_m,
    request.axis_origin_m[2],
  ];
  if (Math.abs(dot(planeOffset, request.base_surface.normal_unit)) > 1e-9) {
    throw new RangeError("axis_origin_m must lie on the base plane");
  }
};

const validateRegions = (request: GroundRegionalMaterialPlanRequest): void => {
  if (!equalVector(request.base_surface.surface_velocity_m_s, ZERO)) {
    throw new RangeError("regional plan v1 supports static surfaces only");
  }
  const regionIds = request.regions.map((region) => region.region_id);
  const precedences = request.regions.map((region) => region.precedence);
  const surfaceIds = [request.base_surface.surface_id,
    ...request.regions.map((region) => region.surface.surface_id)];
  if (new Set(regionIds).size !== regionIds.length) {
    throw new RangeError("region_id values must be unique");
  }
  if (new Set(precedences).size !== precedences.length) {
    throw new RangeError("precedence values must be unique");
  }
  if (new Set(surfaceIds).size !== surfaceIds.length) {
    throw new RangeError("surface_id values must be unique");
  }
  request.regions.forEach((region) => validateRegionGeometry(region, request));
};

const validateRegionGeometry = (
  region: GroundRegionalMaterialRegion,
  request: GroundRegionalMaterialPlanRequest,
): void => {
  const surface = region.surface;
  if (!equalVector(surface.surface_velocity_m_s, ZERO)) {
    throw new RangeError("regional plan v1 supports static surfaces only");
  }
  if (surface.frame !== request.base_surface.frame ||
    surface.height_m !== request.base_surface.height_m ||
    !equalVector(surface.normal_unit, request.base_surface.normal_unit)) {
    throw new RangeError("regional profiles must share the coplanar static geometry");
  }
  if (region.lower_coordinate_m < request.lower_coordinate_m - GEOMETRY_TOLERANCE ||
    region.upper_coordinate_m > request.upper_coordinate_m + GEOMETRY_TOLERANCE) {
    throw new RangeError("regional bounds must lie inside the base domain");
  }
};

/** Parse one exact regional material plan request. */
export const parseGroundRegionalMaterialPlanRequest = (
  value: unknown,
): GroundRegionalMaterialPlanRequest => {
  const item = record(value, "regional material plan request");
  exact(item, requestKeys, "regional material plan request");
  const lower = canonicalNumber(item.lower_coordinate_m, "lower_coordinate_m");
  const upper = canonicalNumber(item.upper_coordinate_m, "upper_coordinate_m");
  if (lower >= upper) throw new RangeError("lower_coordinate_m must be below upper_coordinate_m");
  const regionValues = array(item.regions, "regions");
  if (regionValues.length === 0) throw new RangeError("regional material plan requires at least one region");
  if (regionValues.length > MAX_GROUND_REGIONAL_PLAN_REGIONS) {
    throw new RangeError("regional material plan supports at most " + MAX_GROUND_REGIONAL_PLAN_REGIONS + " regions");
  }
  const parsed = Object.freeze({
    request_id: text(item.request_id, "request_id"),
    base_surface: parseSurface(item.base_surface),
    axis_origin_m: vector(item.axis_origin_m, "axis_origin_m"),
    axis_unit: vector(item.axis_unit, "axis_unit"),
    lower_coordinate_m: lower,
    upper_coordinate_m: upper,
    regions: Object.freeze(regionValues.map(parseRegion)),
    provenance: parseProvenance(item.provenance),
    geometry_model: fixedText(item.geometry_model, GROUND_REGIONAL_PLAN_GEOMETRY_MODEL, "geometry_model"),
    limitations: fixedLimitations(item.limitations),
    unit_system: fixedText(item.unit_system, "SI", "unit_system"),
    schema_version: fixedText(item.schema_version, GROUND_REGIONAL_PLAN_REQUEST_VERSION, "schema_version"),
  });
  validateAxis(parsed);
  validateRegions(parsed);
  return parsed;
};

const canonicalRegions = (
  request: GroundRegionalMaterialPlanRequest,
): readonly GroundRegionalMaterialRegion[] => [...request.regions].sort((left, right) =>
  right.precedence - left.precedence ||
  (left.region_id < right.region_id ? -1 : left.region_id > right.region_id ? 1 : 0));

/** Parse one result and reject any material evidence not copied from its request. */
export const parseGroundRegionalMaterialPlanResult = (
  value: unknown,
): GroundRegionalMaterialPlanResult => {
  const item = record(value, "regional material plan result");
  exact(item, resultKeys, "regional material plan result");
  const request = parseGroundRegionalMaterialPlanRequest(item.request);
  const requestSha256 = parseDigest(item.request_sha256, "request_sha256");
  if (requestSha256 !== sha256Text(canonicalGroundJson(request))) {
    throw new RangeError("request_sha256 does not match the embedded request");
  }
  const orderedRegions = Object.freeze(array(item.ordered_regions, "ordered_regions").map(parseRegion));
  if (canonicalGroundJson(orderedRegions) !== canonicalGroundJson(canonicalRegions(request))) {
    throw new RangeError("ordered region surface identity or canonical precedence order does not match request evidence");
  }
  const provenance = parseProvenance(item.provenance);
  if (provenance.input_sha256 !== requestSha256) {
    throw new RangeError("result provenance input_sha256 must match request_sha256");
  }
  return Object.freeze({
    request,
    request_sha256: requestSha256,
    ordered_regions: orderedRegions,
    provenance,
    limitations: fixedLimitations(item.limitations),
    unit_system: fixedText(item.unit_system, "SI", "unit_system"),
    schema_version: fixedText(item.schema_version, GROUND_REGIONAL_PLAN_RESULT_VERSION, "schema_version"),
  });
};

const parseWireDocument = (value: string): unknown => {
  if (typeof value !== "string") throw new TypeError("regional material plan JSON must be text");
  if (new TextEncoder().encode(value).byteLength > MAX_GROUND_REGIONAL_PLAN_WIRE_BYTES) {
    throw new RangeError("regional material plan exceeds maximum wire size");
  }
  return parseUniqueJson(value);
};

export const groundRegionalMaterialPlanRequestFromJson = (
  value: string,
): GroundRegionalMaterialPlanRequest =>
  parseGroundRegionalMaterialPlanRequest(parseWireDocument(value));

export const groundRegionalMaterialPlanResultFromJson = (
  value: string,
): GroundRegionalMaterialPlanResult =>
  parseGroundRegionalMaterialPlanResult(parseWireDocument(value));

export const stableGroundRegionalMaterialPlanJson = (value: unknown): string =>
  canonicalGroundJson(value);
