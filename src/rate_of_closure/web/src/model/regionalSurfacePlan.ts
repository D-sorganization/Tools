/** Presentation draft adapter for the strict regional-material wire contract. */

import {
  GROUND_REGIONAL_PLAN_GEOMETRY_MODEL,
  GROUND_REGIONAL_PLAN_LIMITATIONS,
  GROUND_REGIONAL_PLAN_REQUEST_VERSION,
  parseGroundRegionalMaterialPlanRequest,
  type GroundRegionalMaterialPlanRequest,
  type GroundRegionalMaterialRegion,
} from "./groundRegionalPlan";
import { GROUND_TARGET_FRAME, type GroundSurfaceProfile } from "./flightGroundTypes";
import { canonicalGroundJson } from "./flightGroundContract";
import { sha256Text } from "./sha256";

export const MAX_REGIONAL_SURFACE_EDITOR_ROWS = 8;
export const REGIONAL_SURFACE_EDITOR_PROVIDER =
  "tools.rate_of_closure.regional_surface_editor";
export const REGIONAL_SURFACE_EDITOR_VERSION = "1.0.0";

export interface SurfaceMaterialDraft {
  readonly surface_id: string;
  readonly normal_restitution: number;
  readonly static_friction: number;
  readonly kinetic_friction: number;
  readonly rolling_resistance: number;
  readonly firmness_pa: number;
  readonly hardness_fraction: number;
  readonly grass_height_m: number;
  readonly compressibility_fraction: number;
  readonly compression_damping_fraction: number;
  readonly turf_density_kg_m3: number;
  readonly moisture_fraction: number;
}

export interface RegionalOverlayDraft {
  readonly region_id: string;
  readonly precedence: number;
  readonly lower_coordinate_m: number;
  readonly upper_coordinate_m: number;
  readonly surface: SurfaceMaterialDraft;
}

export interface RegionalSurfacePlanDraft {
  readonly request_id: string;
  readonly lower_coordinate_m: number;
  readonly upper_coordinate_m: number;
  readonly source_revision: string;
  readonly calibration_kind: "unvalidated";
  readonly base_surface: SurfaceMaterialDraft;
  readonly regions: readonly RegionalOverlayDraft[];
}

const illustrativeSurface = (
  surfaceId: string,
  rough: boolean,
): SurfaceMaterialDraft => rough ? {
  surface_id: surfaceId,
  normal_restitution: 0.31,
  static_friction: 0.52,
  kinetic_friction: 0.41,
  rolling_resistance: 0.08,
  firmness_pa: 700_000,
  hardness_fraction: 0.48,
  grass_height_m: 0.035,
  compressibility_fraction: 0.34,
  compression_damping_fraction: 0.38,
  turf_density_kg_m3: 240,
  moisture_fraction: 0.42,
} : {
  surface_id: surfaceId,
  normal_restitution: 0.42,
  static_friction: 0.35,
  kinetic_friction: 0.28,
  rolling_resistance: 0.04,
  firmness_pa: 1_200_000,
  hardness_fraction: 0.70,
  grass_height_m: 0.012,
  compressibility_fraction: 0.20,
  compression_damping_fraction: 0.25,
  turf_density_kg_m3: 180,
  moisture_fraction: 0.30,
};

export const illustrativeRegionalSurfacePlanDraft = (): RegionalSurfacePlanDraft => ({
  request_id: "illustrative-regional-plan",
  lower_coordinate_m: 0,
  upper_coordinate_m: 300,
  source_revision: "interactive-illustrative-draft-v1",
  calibration_kind: "unvalidated",
  base_surface: illustrativeSurface("illustrative-fairway", false),
  regions: [{
    region_id: "illustrative-rough-band",
    precedence: 10,
    lower_coordinate_m: 120,
    upper_coordinate_m: 150,
    surface: illustrativeSurface("illustrative-rough", true),
  }],
});

const surfacePayload = (surface: SurfaceMaterialDraft): GroundSurfaceProfile => ({
  ...surface,
  provider_id: REGIONAL_SURFACE_EDITOR_PROVIDER,
  provider_version: REGIONAL_SURFACE_EDITOR_VERSION,
  frame: GROUND_TARGET_FRAME,
  height_m: 0,
  normal_unit: [0, 1, 0],
  surface_velocity_m_s: [0, 0, 0],
});

const surfaceDraft = (surface: GroundSurfaceProfile): SurfaceMaterialDraft => ({
  surface_id: surface.surface_id,
  normal_restitution: surface.normal_restitution,
  static_friction: surface.static_friction,
  kinetic_friction: surface.kinetic_friction,
  rolling_resistance: surface.rolling_resistance,
  firmness_pa: surface.firmness_pa,
  hardness_fraction: surface.hardness_fraction,
  grass_height_m: surface.grass_height_m,
  compressibility_fraction: surface.compressibility_fraction,
  compression_damping_fraction: surface.compression_damping_fraction,
  turf_density_kg_m3: surface.turf_density_kg_m3,
  moisture_fraction: surface.moisture_fraction,
});

const assertEditorSurface = (surface: GroundSurfaceProfile): void => {
  if (surface.provider_id !== REGIONAL_SURFACE_EDITOR_PROVIDER ||
      surface.provider_version !== REGIONAL_SURFACE_EDITOR_VERSION) {
    throw new RangeError("surface is not qualified by the editor provider v1");
  }
};

const overlayDraft = (region: GroundRegionalMaterialRegion): RegionalOverlayDraft => ({
  region_id: region.region_id,
  precedence: region.precedence,
  lower_coordinate_m: region.lower_coordinate_m,
  upper_coordinate_m: region.upper_coordinate_m,
  surface: surfaceDraft(region.surface),
});

/** Project a fully validated editor-v1 request without relabelling evidence. */
export const editorDraftFromGroundRegionalSurfacePlanRequest = (
  request: GroundRegionalMaterialPlanRequest,
): RegionalSurfacePlanDraft => {
  if (request.provenance.producer !== REGIONAL_SURFACE_EDITOR_PROVIDER ||
      request.provenance.producer_version !== REGIONAL_SURFACE_EDITOR_VERSION) {
    throw new RangeError("request is not qualified by the editor producer v1");
  }
  if (canonicalGroundJson(request.axis_origin_m) !== "[0,0,0]" ||
      canonicalGroundJson(request.axis_unit) !== "[1,0,0]") {
    throw new RangeError("request uses an unsupported editor axis qualification");
  }
  if (request.regions.length > MAX_REGIONAL_SURFACE_EDITOR_ROWS) {
    throw new RangeError(
      `editor supports one to at most ${MAX_REGIONAL_SURFACE_EDITOR_ROWS} regions`,
    );
  }
  assertEditorSurface(request.base_surface);
  request.regions.forEach((region) => assertEditorSurface(region.surface));
  const draft: RegionalSurfacePlanDraft = {
    request_id: request.request_id,
    lower_coordinate_m: request.lower_coordinate_m,
    upper_coordinate_m: request.upper_coordinate_m,
    source_revision: request.provenance.source_revision,
    calibration_kind: "unvalidated",
    base_surface: surfaceDraft(request.base_surface),
    regions: request.regions.map(overlayDraft),
  };
  if (request.provenance.input_sha256 !== sha256Text(canonicalGroundJson(draft))) {
    throw new RangeError("editor provenance digest does not match the editable request");
  }
  return draft;
};

export const buildGroundRegionalSurfacePlanRequest = (
  draft: RegionalSurfacePlanDraft,
): GroundRegionalMaterialPlanRequest => {
  if (draft.calibration_kind !== "unvalidated") {
    throw new RangeError("this editor slice supports unvalidated drafts only");
  }
  if (draft.regions.length < 1 ||
      draft.regions.length > MAX_REGIONAL_SURFACE_EDITOR_ROWS) {
    throw new RangeError(
      `editor supports one to at most ${MAX_REGIONAL_SURFACE_EDITOR_ROWS} regions`,
    );
  }
  const inputDigest = sha256Text(canonicalGroundJson(draft));
  return parseGroundRegionalMaterialPlanRequest({
    request_id: draft.request_id,
    base_surface: surfacePayload(draft.base_surface),
    axis_origin_m: [0, 0, 0],
    axis_unit: [1, 0, 0],
    lower_coordinate_m: draft.lower_coordinate_m,
    upper_coordinate_m: draft.upper_coordinate_m,
    regions: draft.regions.map((region) => ({
      region_id: region.region_id,
      precedence: region.precedence,
      lower_coordinate_m: region.lower_coordinate_m,
      upper_coordinate_m: region.upper_coordinate_m,
      surface: surfacePayload(region.surface),
    })),
    provenance: {
      producer: REGIONAL_SURFACE_EDITOR_PROVIDER,
      producer_version: REGIONAL_SURFACE_EDITOR_VERSION,
      source_revision: draft.source_revision,
      input_sha256: inputDigest,
    },
    geometry_model: GROUND_REGIONAL_PLAN_GEOMETRY_MODEL,
    limitations: GROUND_REGIONAL_PLAN_LIMITATIONS,
    unit_system: "SI",
    schema_version: GROUND_REGIONAL_PLAN_REQUEST_VERSION,
  });
};

/** Keep untouched imported evidence exact; bind fresh provenance after edits. */
export const regionalSurfacePlanRequestForDraft = (
  draft: RegionalSurfacePlanDraft,
  importedRequest: GroundRegionalMaterialPlanRequest | null = null,
): GroundRegionalMaterialPlanRequest => {
  if (importedRequest !== null) {
    const importedDraft = editorDraftFromGroundRegionalSurfacePlanRequest(importedRequest);
    if (canonicalGroundJson(draft) === canonicalGroundJson(importedDraft)) {
      return importedRequest;
    }
  }
  return buildGroundRegionalSurfacePlanRequest(draft);
};
