/** Strict whole-workspace adapter for the browser's supported live state. */

import type { UnitSelections } from "../components/ImpactExplorerPanel";
import type { ClubSpec, ClubType } from "./club";
import { validateScenario, type ImpactScenario } from "./impact";
import {
  PRIMARY_VIEW_IDS,
  REQUIRED_PRIMARY_VIEW_IDS,
  type PrimaryViewId,
  type PrimaryViewState,
} from "./viewPreferences";
import {
  viewWorkspaceDocument,
  viewWorkspaceFromDocument,
  type ViewWorkspace,
} from "./viewWorkspace";
import {
  simulationWorkspaceDocument,
  simulationWorkspaceFromSession,
  type SimulationWorkspaceSnapshot,
} from "./workspaceSimulationSession";
import {
  torqueWorkspaceDocument,
  torqueWorkspaceFromSession,
  type TorqueWorkspaceSnapshot,
} from "./workspaceTorqueSession";

const WORKSPACE_SCHEMA = "rate_of_closure.workspace";
const WORKSPACE_VERSION = 2;
const SESSION_SCHEMA = "rate_of_closure.explorer_session";
const CLUB_SCHEMA = "rate_of_closure.club_configuration";
const SESSION_PAYLOAD_VERSION = 3;
const CLUB_PAYLOAD_VERSION = 1;
const STABLE_ID = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;
const UTC_TIMESTAMP =
  /^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?Z$/;
const CLUB_TYPES: readonly ClubType[] = ["Driver", "Wood", "Hybrid", "Iron", "Wedge", "Putter"];
const HEAD_STYLES = ["Auto", "Mallet", "Blade"] as const;
const CLUB_BOUNDS = {
  lengthM: [0.6, 1.3], headMassKg: [0.1, 0.5], loftDeg: [0, 70], lieDeg: [45, 80],
  moiAboutShaftKgM2: [5e-5, 2e-3], cgDepthM: [0, 0.08], cgHeightM: [0, 0.06],
} as const;
const UNIT_OPTIONS = {
  speed: ["mph", "m/s", "km/h"],
  rotation: ["deg/s", "rad/s", "rpm"],
  length: ["mm", "cm", "in"],
  distance: ["yd", "m", "ft"],
} as const;

export interface WorkspaceSessionSnapshot {
  readonly scenario: ImpactScenario;
  readonly club: ClubSpec;
  readonly units: UnitSelections;
  readonly simulation: SimulationWorkspaceSnapshot;
  readonly torque: TorqueWorkspaceSnapshot;
  readonly modules: PrimaryViewState;
  readonly viewWorkspace: ViewWorkspace;
}

export interface WorkspaceFileMetadata {
  readonly documentId: string;
  readonly title: string;
  readonly createdAtUtc: string;
  readonly modifiedAtUtc: string;
  readonly appVersion: string;
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

function exactRecord(value: unknown, keys: readonly string[], context: string): Record<string, unknown> {
  if (!isRecord(value)) throw new TypeError(`${context} must be an object`);
  const actual = Object.keys(value);
  if (actual.length !== keys.length || keys.some((key) => !(key in value))) {
    throw new TypeError(`${context} has invalid fields`);
  }
  return value;
}

function objectRecord(value: unknown, context: string): Record<string, unknown> {
  if (!isRecord(value)) throw new TypeError(`${context} must be an object`);
  return value;
}

function finite(value: unknown, context: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError(`${context} must be finite`);
  }
  return value;
}

function scenarioDocument(scenario: ImpactScenario): Record<string, number> {
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
}

function scenarioFromDocument(value: unknown): ImpactScenario {
  const data = exactRecord(value, [
    "clubhead_speed_mph", "omega_plane_dps", "omega_shaft_dps", "lie_angle_deg",
    "com_to_face_mm", "impact_offset_toe_mm", "impact_offset_high_mm", "contact_duration_us",
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
}

function clubDocument(club: ClubSpec): Record<string, unknown> {
  return {
    name: club.name,
    club_type: club.clubType,
    length_m: club.lengthM,
    head_mass_kg: club.headMassKg,
    loft_deg: club.loftDeg,
    lie_deg: club.lieDeg,
    moi_about_shaft_kg_m2: club.moiAboutShaftKgM2,
    cg_depth_m: club.cgDepthM,
    cg_height_m: club.cgHeightM,
    face_bulge_radius_m: club.faceBulgeRadiusM,
    face_roll_radius_m: club.faceRollRadiusM,
    head_style: club.headStyle ?? "Auto",
  };
}

function optionalFinite(value: unknown, context: string): number | null {
  return value === null ? null : finite(value, context);
}

function clubFromDocument(value: unknown): ClubSpec {
  const data = exactRecord(value, [
    "name", "club_type", "length_m", "head_mass_kg", "loft_deg", "lie_deg",
    "moi_about_shaft_kg_m2", "cg_depth_m", "cg_height_m", "face_bulge_radius_m",
    "face_roll_radius_m", "head_style",
  ], "club_configuration.data");
  if (typeof data.name !== "string" || data.name.trim().length === 0 ||
      !CLUB_TYPES.includes(data.club_type as ClubType) ||
      !HEAD_STYLES.includes(data.head_style as typeof HEAD_STYLES[number])) {
    throw new TypeError("club configuration identity is invalid");
  }
  const club: ClubSpec = {
    name: data.name,
    clubType: data.club_type as ClubType,
    lengthM: finite(data.length_m, "length_m"),
    headMassKg: finite(data.head_mass_kg, "head_mass_kg"),
    loftDeg: finite(data.loft_deg, "loft_deg"),
    lieDeg: finite(data.lie_deg, "lie_deg"),
    moiAboutShaftKgM2: finite(data.moi_about_shaft_kg_m2, "moi_about_shaft_kg_m2"),
    cgDepthM: finite(data.cg_depth_m, "cg_depth_m"),
    cgHeightM: finite(data.cg_height_m, "cg_height_m"),
    faceBulgeRadiusM: optionalFinite(data.face_bulge_radius_m, "face_bulge_radius_m"),
    faceRollRadiusM: optionalFinite(data.face_roll_radius_m, "face_roll_radius_m"),
    headStyle: data.head_style as typeof HEAD_STYLES[number],
  };
  for (const key of Object.keys(CLUB_BOUNDS) as (keyof typeof CLUB_BOUNDS)[]) {
    const [low, high] = CLUB_BOUNDS[key];
    if (club[key] < low || club[key] > high) throw new RangeError(`${key} is out of range`);
  }
  for (const [key, value] of [
    ["faceBulgeRadiusM", club.faceBulgeRadiusM],
    ["faceRollRadiusM", club.faceRollRadiusM],
  ] as const) {
    if (value !== null && (value < 0.1 || value > 2)) throw new RangeError(`${key} is out of range`);
  }
  return club;
}

function validatedUnits(value: unknown): UnitSelections {
  const units = exactRecord(value, Object.keys(UNIT_OPTIONS), "model_session.units");
  for (const key of Object.keys(UNIT_OPTIONS) as (keyof UnitSelections)[]) {
    if (!(UNIT_OPTIONS[key] as readonly unknown[]).includes(units[key])) {
      throw new TypeError(`unsupported ${key} unit`);
    }
  }
  return units as unknown as UnitSelections;
}

function validatedModules(layout: Record<string, unknown>): PrimaryViewState {
  const order = layout.module_order;
  const visible = layout.visible_module_ids;
  if (!Array.isArray(order) || order.length !== PRIMARY_VIEW_IDS.length ||
      new Set(order).size !== order.length || PRIMARY_VIEW_IDS.some((id) => !order.includes(id))) {
    throw new TypeError("module_order must contain every module exactly once");
  }
  if (!Array.isArray(visible) || visible.length === 0 || new Set(visible).size !== visible.length ||
      visible.some((id) => !PRIMARY_VIEW_IDS.includes(id as PrimaryViewId)) ||
      REQUIRED_PRIMARY_VIEW_IDS.some((id) => !visible.includes(id)) ||
      !visible.includes(layout.active_module_id)) {
    throw new TypeError("module visibility and active module are invalid");
  }
  return {
    version: 2,
    order: [...order] as PrimaryViewId[],
    visible: [...visible] as PrimaryViewId[],
    active: layout.active_module_id as PrimaryViewId,
  };
}

function payload(
  value: unknown,
  schema: string,
  versions: readonly number[],
  context: string,
): { readonly version: number; readonly data: Record<string, unknown> } {
  const envelope = exactRecord(value, ["schema", "schema_version", "data"], context);
  if (
    envelope.schema !== schema ||
    typeof envelope.schema_version !== "number" ||
    !versions.includes(envelope.schema_version)
  ) {
    throw new TypeError(`${context} has an unsupported schema`);
  }
  return {
    version: envelope.schema_version,
    data: objectRecord(envelope.data, `${context}.data`),
  };
}

function validateMetadata(value: unknown): void {
  const data = exactRecord(value, [
    "document_id", "title", "created_at_utc", "modified_at_utc", "app_version", "provenance",
  ], "metadata");
  for (const key of ["document_id", "title", "created_at_utc", "modified_at_utc", "app_version"] as const) {
    if (typeof data[key] !== "string" || data[key].trim().length === 0) {
      throw new TypeError(`metadata.${key} must be non-empty text`);
    }
  }
  if (!STABLE_ID.test(data.document_id as string)) {
    throw new TypeError("metadata.document_id must be a stable identifier");
  }
  const createdText = data.created_at_utc as string;
  const modifiedText = data.modified_at_utc as string;
  if (!UTC_TIMESTAMP.test(createdText) || !UTC_TIMESTAMP.test(modifiedText)) {
    throw new TypeError("workspace metadata timestamps must be strict UTC text");
  }
  const created = Date.parse(createdText);
  const modified = Date.parse(modifiedText);
  if (!Number.isFinite(created) || !Number.isFinite(modified) || modified < created) {
    throw new TypeError("workspace metadata timestamps are invalid");
  }
  if (!isRecord(data.provenance) || Object.entries(data.provenance).some(
    ([key, entry]) => key.trim().length === 0 || typeof entry !== "string",
  )) throw new TypeError("metadata.provenance must map text keys to text values");
}

/** Serialize the browser-supported whole workspace without hidden file authority. */
export function createWorkspaceDocument(
  snapshot: WorkspaceSessionSnapshot,
  metadata: WorkspaceFileMetadata,
): string {
  const document = {
    schema: WORKSPACE_SCHEMA,
    schema_version: WORKSPACE_VERSION,
    metadata: {
      document_id: metadata.documentId,
      title: metadata.title,
      created_at_utc: metadata.createdAtUtc,
      modified_at_utc: metadata.modifiedAtUtc,
      app_version: metadata.appVersion,
      provenance: { surface: "rate-of-closure-file-adapter/v1" },
    },
    model_session: {
      schema: SESSION_SCHEMA,
      schema_version: SESSION_PAYLOAD_VERSION,
      data: {
        scenario: scenarioDocument(snapshot.scenario),
        units: snapshot.units,
        simulation_setup: simulationWorkspaceDocument(snapshot.simulation, snapshot.club),
        torque_selection: torqueWorkspaceDocument(snapshot.torque),
      },
    },
    prescribed_torque_profiles: snapshot.torque.profiles.map((profile) =>
      profile.toJsonObject()),
    club_configuration: {
      schema: CLUB_SCHEMA,
      schema_version: CLUB_PAYLOAD_VERSION,
      data: clubDocument(snapshot.club),
    },
    variation_plan: null,
    layout: {
      module_order: snapshot.modules.order,
      visible_module_ids: snapshot.modules.visible,
      active_module_id: snapshot.modules.active,
      view_workspace: {
        schema: "rate_of_closure.view_workspace",
        schema_version: 1,
        data: viewWorkspaceDocument(snapshot.viewWorkspace),
      },
    },
  };
  parseWorkspaceDocument(JSON.stringify(document));
  return `${JSON.stringify(document, null, 2)}\n`;
}

/** Parse and validate the entire document before exposing any applicable state. */
export interface WorkspaceParseOptions {
  readonly legacySimulationFallback?: SimulationWorkspaceSnapshot;
  readonly legacyTorqueFallback?: TorqueWorkspaceSnapshot;
}

/** Parse a current file or deliberately migrate v1 with an explicit fallback. */
export function parseWorkspaceDocument(
  text: string,
  options: WorkspaceParseOptions = {},
): WorkspaceSessionSnapshot {
  if (typeof text !== "string" || text.trim().length === 0) {
    throw new TypeError("workspace file must contain JSON text");
  }
  const root = exactRecord(JSON.parse(text), [
    "schema", "schema_version", "metadata", "model_session",
    "prescribed_torque_profiles", "club_configuration", "variation_plan", "layout",
  ], "workspace");
  if (root.schema !== WORKSPACE_SCHEMA || root.schema_version !== WORKSPACE_VERSION) {
    throw new TypeError("unsupported workspace schema");
  }
  validateMetadata(root.metadata);
  if (root.variation_plan !== null) {
    throw new TypeError("variation plans are not supported by this adapter");
  }
  const sessionEnvelope = payload(
    root.model_session,
    SESSION_SCHEMA,
    [1, 2, SESSION_PAYLOAD_VERSION],
    "model_session",
  );
  const session = exactRecord(
    sessionEnvelope.data,
    sessionEnvelope.version === 1
      ? ["scenario", "units"]
      : sessionEnvelope.version === 2
        ? ["scenario", "units", "simulation_setup"]
        : ["scenario", "units", "simulation_setup", "torque_selection"],
    "model_session.data",
  );
  const clubEnvelope = payload(
    root.club_configuration,
    CLUB_SCHEMA,
    [CLUB_PAYLOAD_VERSION],
    "club_configuration",
  );
  const club = exactRecord(
    clubEnvelope.data,
    Object.keys(clubDocument({
      name: "x", clubType: "Driver", lengthM: 1, headMassKg: 0.2, loftDeg: 10,
      lieDeg: 56, moiAboutShaftKgM2: 0.0005, cgDepthM: 0.02, cgHeightM: 0.02,
      faceBulgeRadiusM: null, faceRollRadiusM: null,
    })),
    "club_configuration.data",
  );
  const layout = exactRecord(root.layout, [
    "module_order", "visible_module_ids", "active_module_id", "view_workspace",
  ], "layout");
  const viewEnvelope = exactRecord(
    layout.view_workspace, ["schema", "schema_version", "data"], "layout.view_workspace",
  );
  if (viewEnvelope.schema !== "rate_of_closure.view_workspace" || viewEnvelope.schema_version !== 1) {
    throw new TypeError("unsupported view workspace payload");
  }
  const parsedClub = clubFromDocument(club);
  const simulation: SimulationWorkspaceSnapshot = simulationWorkspaceFromSession({
    isLegacy: sessionEnvelope.version === 1,
    setupDocument: session.simulation_setup,
    club: parsedClub,
    legacyFallback: options.legacySimulationFallback,
  });
  const torque: TorqueWorkspaceSnapshot = torqueWorkspaceFromSession({
    isLegacy: sessionEnvelope.version < SESSION_PAYLOAD_VERSION,
    selectionDocument: session.torque_selection,
    profileDocuments: root.prescribed_torque_profiles,
    legacyFallback: options.legacyTorqueFallback,
  });
  return {
    scenario: scenarioFromDocument(session.scenario),
    club: parsedClub,
    units: validatedUnits(session.units),
    simulation,
    torque,
    modules: validatedModules(layout),
    viewWorkspace: viewWorkspaceFromDocument(viewEnvelope.data),
  };
}
