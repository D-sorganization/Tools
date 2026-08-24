/** Strict leaf-field adapters for whole-workspace persistence. */

import type { UnitSelections } from "../components/ImpactExplorerPanel";
import type { ClubSpec, ClubType } from "./club";
import { validateScenario, type ImpactScenario } from "./impact";
import {
  PRIMARY_VIEW_IDS,
  REQUIRED_PRIMARY_VIEW_IDS,
  type PrimaryViewId,
  type PrimaryViewState,
} from "./viewPreferences";

const CLUB_TYPES: readonly ClubType[] = [
  "Driver", "Wood", "Hybrid", "Iron", "Wedge", "Putter",
];
const HEAD_STYLES = ["Auto", "Mallet", "Blade"] as const;
const CLUB_BOUNDS = {
  lengthM: [0.6, 1.3], headMassKg: [0.1, 0.5], loftDeg: [0, 70],
  lieDeg: [45, 80], moiAboutShaftKgM2: [5e-5, 2e-3],
  cgDepthM: [0, 0.08], cgHeightM: [0, 0.06],
} as const;
const UNIT_OPTIONS = {
  speed: ["mph", "m/s", "km/h"], rotation: ["deg/s", "rad/s", "rpm"],
  length: ["mm", "cm", "in"], distance: ["yd", "m", "ft"],
} as const;

const exactRecord = (
  value: unknown, keys: readonly string[], context: string,
): Record<string, unknown> => {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError(`${context} must be an object`);
  }
  const record = value as Record<string, unknown>;
  const actual = Object.keys(record);
  if (actual.length !== keys.length || keys.some((key) => !(key in record))) {
    throw new TypeError(`${context} has invalid fields`);
  }
  return record;
};

const finite = (value: unknown, context: string): number => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError(`${context} must be finite`);
  }
  return value;
};

export const workspaceScenarioDocument = (
  scenario: ImpactScenario,
): Record<string, number> => {
  validateScenario(scenario);
  return {
    clubhead_speed_mph: scenario.clubheadSpeedMph,
    omega_plane_dps: scenario.omegaPlaneDps,
    omega_shaft_dps: scenario.omegaShaftDps,
    lie_angle_deg: scenario.lieAngleDeg,
    com_to_face_mm: scenario.comToFaceMm,
    impact_offset_toe_mm: scenario.impactOffsetToeMm,
    impact_offset_high_mm: scenario.impactOffsetHighMm,
    contact_duration_us: scenario.contactDurationUs,
  };
};

export const workspaceScenarioFromDocument = (value: unknown): ImpactScenario => {
  const data = exactRecord(value, [
    "clubhead_speed_mph", "omega_plane_dps", "omega_shaft_dps",
    "lie_angle_deg", "com_to_face_mm", "impact_offset_toe_mm",
    "impact_offset_high_mm", "contact_duration_us",
  ], "model_session.scenario");
  const scenario: ImpactScenario = {
    clubheadSpeedMph: finite(data.clubhead_speed_mph, "clubhead_speed_mph"),
    omegaPlaneDps: finite(data.omega_plane_dps, "omega_plane_dps"),
    omegaShaftDps: finite(data.omega_shaft_dps, "omega_shaft_dps"),
    lieAngleDeg: finite(data.lie_angle_deg, "lie_angle_deg"),
    comToFaceMm: finite(data.com_to_face_mm, "com_to_face_mm"),
    impactOffsetToeMm: finite(data.impact_offset_toe_mm, "impact_offset_toe_mm"),
    impactOffsetHighMm: finite(data.impact_offset_high_mm, "impact_offset_high_mm"),
    contactDurationUs: finite(data.contact_duration_us, "contact_duration_us"),
  };
  validateScenario(scenario);
  return scenario;
};

export const workspaceClubDocument = (club: ClubSpec): Record<string, unknown> => ({
  name: club.name, club_type: club.clubType, length_m: club.lengthM,
  head_mass_kg: club.headMassKg, loft_deg: club.loftDeg, lie_deg: club.lieDeg,
  moi_about_shaft_kg_m2: club.moiAboutShaftKgM2, cg_depth_m: club.cgDepthM,
  cg_height_m: club.cgHeightM, face_bulge_radius_m: club.faceBulgeRadiusM,
  face_roll_radius_m: club.faceRollRadiusM, head_style: club.headStyle ?? "Auto",
});

const optionalFinite = (value: unknown, context: string): number | null =>
  value === null ? null : finite(value, context);

export const workspaceClubFromDocument = (value: unknown): ClubSpec => {
  const data = exactRecord(value, [
    "name", "club_type", "length_m", "head_mass_kg", "loft_deg", "lie_deg",
    "moi_about_shaft_kg_m2", "cg_depth_m", "cg_height_m",
    "face_bulge_radius_m", "face_roll_radius_m", "head_style",
  ], "club_configuration.data");
  if (typeof data.name !== "string" || data.name.trim().length === 0 ||
      !CLUB_TYPES.includes(data.club_type as ClubType) ||
      !HEAD_STYLES.includes(data.head_style as (typeof HEAD_STYLES)[number])) {
    throw new TypeError("club configuration identity is invalid");
  }
  const club: ClubSpec = {
    name: data.name, clubType: data.club_type as ClubType,
    lengthM: finite(data.length_m, "length_m"),
    headMassKg: finite(data.head_mass_kg, "head_mass_kg"),
    loftDeg: finite(data.loft_deg, "loft_deg"),
    lieDeg: finite(data.lie_deg, "lie_deg"),
    moiAboutShaftKgM2: finite(data.moi_about_shaft_kg_m2, "moi_about_shaft_kg_m2"),
    cgDepthM: finite(data.cg_depth_m, "cg_depth_m"),
    cgHeightM: finite(data.cg_height_m, "cg_height_m"),
    faceBulgeRadiusM: optionalFinite(data.face_bulge_radius_m, "face_bulge_radius_m"),
    faceRollRadiusM: optionalFinite(data.face_roll_radius_m, "face_roll_radius_m"),
    headStyle: data.head_style as (typeof HEAD_STYLES)[number],
  };
  for (const key of Object.keys(CLUB_BOUNDS) as (keyof typeof CLUB_BOUNDS)[]) {
    const [low, high] = CLUB_BOUNDS[key];
    if (club[key] < low || club[key] > high) throw new RangeError(`${key} is out of range`);
  }
  for (const [key, entry] of [
    ["faceBulgeRadiusM", club.faceBulgeRadiusM],
    ["faceRollRadiusM", club.faceRollRadiusM],
  ] as const) {
    if (entry !== null && (entry < 0.1 || entry > 2)) {
      throw new RangeError(`${key} is out of range`);
    }
  }
  return club;
};

export const validatedWorkspaceUnits = (value: unknown): UnitSelections => {
  const units = exactRecord(value, Object.keys(UNIT_OPTIONS), "model_session.units");
  for (const key of Object.keys(UNIT_OPTIONS) as (keyof UnitSelections)[]) {
    if (!(UNIT_OPTIONS[key] as readonly unknown[]).includes(units[key])) {
      throw new TypeError(`unsupported ${key} unit`);
    }
  }
  return units as unknown as UnitSelections;
};

export const validatedWorkspaceModules = (
  layout: Record<string, unknown>,
): PrimaryViewState => {
  const order = layout.module_order;
  const visible = layout.visible_module_ids;
  if (!Array.isArray(order) || order.length !== PRIMARY_VIEW_IDS.length ||
      new Set(order).size !== order.length ||
      PRIMARY_VIEW_IDS.some((id) => !order.includes(id))) {
    throw new TypeError("module_order must contain every module exactly once");
  }
  if (!Array.isArray(visible) || visible.length === 0 ||
      new Set(visible).size !== visible.length ||
      visible.some((id) => !PRIMARY_VIEW_IDS.includes(id as PrimaryViewId)) ||
      REQUIRED_PRIMARY_VIEW_IDS.some((id) => !visible.includes(id)) ||
      !visible.includes(layout.active_module_id)) {
    throw new TypeError("module visibility and active module are invalid");
  }
  return {
    version: 2, order: [...order] as PrimaryViewId[],
    visible: [...visible] as PrimaryViewId[], active: layout.active_module_id as PrimaryViewId,
  };
};
