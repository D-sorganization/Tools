import {
  GROUND_BALL_SETUP,
  BALL_HEIGHT_REFERENCE,
  ballCenterPosition,
  ballSetupFromJson,
  ballSetupToJson,
  resolveBallSetup,
  type BallSetup,
} from "./ballSetup";
import { analyzeTwist, type Twist6 } from "./screwAnalysis";
import type { SimulationInput, SimulationRunTs } from "./simulation";
import { createSpatialTarget, type SpatialTargetTs } from "./spatialTarget";
import {
  spatialTargetFromJson,
  spatialTargetToJson,
} from "./spatialTargetSerialization";
import { DEFAULT_TARGET, spatialTargetFromRegion } from "./targets";
import { SIMULATION_MODEL_LIMITATIONS } from "./modelLimitations";
import { simulationDocumentFormat } from "./simulationDocumentFormat";

export const BALL_SETUP_STORAGE_KEY = "rate_of_closure.ball_setup.web/v1";
export const SIMULATION_EXPORT_FORMAT = "rate_of_closure.simulation_run.web/5";

export interface BallSetupPreference {
  setup: BallSetup;
  userOverridden: boolean;
}

export interface LoadedBallSetupPreference extends BallSetupPreference {
  warning: string | null;
}

const record = (value: unknown): Record<string, unknown> | null =>
  typeof value === "object" && value !== null
    ? value as Record<string, unknown>
    : null;

function setupFromUnknown(value: unknown): BallSetup {
  return ballSetupFromJson(value);
}

const browserStorage = (): Storage | null => {
  if (typeof window === "undefined") return null;
  try {
    const candidate = window.localStorage;
    return typeof candidate?.getItem === "function" &&
      typeof candidate?.setItem === "function" ? candidate : null;
  } catch {
    return null;
  }
};

export function loadBallSetupPreference(
  storage: Storage | null = browserStorage(),
  fallback: BallSetup = { ...GROUND_BALL_SETUP },
): LoadedBallSetupPreference {
  const safeFallback = resolveBallSetup(fallback);
  let text: string | null;
  try {
    text = typeof storage?.getItem === "function"
      ? storage.getItem(BALL_SETUP_STORAGE_KEY)
      : null;
  } catch (error) {
    return {
      setup: safeFallback,
      userOverridden: false,
      warning: `Saved ball setup could not be read: ${(error as Error).message}`,
    };
  }
  if (!text) return { setup: safeFallback, userOverridden: false, warning: null };
  try {
    const data = record(JSON.parse(text));
    if (!data || data.schema_version !== 1) throw new Error("unsupported schema version");
    return {
      setup: setupFromUnknown(data.ball_setup),
      userOverridden: data.user_overridden === true,
      warning: null,
    };
  } catch {
    return {
      setup: safeFallback,
      userOverridden: false,
      warning: "Saved ball setup could not be loaded; the club default was restored safely.",
    };
  }
}

export function saveBallSetupPreference(
  preference: BallSetupPreference,
  storage: Storage | null = browserStorage(),
): string | null {
  if (!storage || typeof storage.setItem !== "function") return null;
  try {
    storage.setItem(BALL_SETUP_STORAGE_KEY, JSON.stringify({
      schema_version: 1,
      ball_setup: ballSetupToJson(preference.setup),
      user_overridden: preference.userOverridden,
    }));
    return null;
  } catch (error) {
    return `Ball setup could not be saved: ${(error as Error).message}`;
  }
}

export function exportBallSetupMetadata(setup: BallSetup) {
  const resolved = resolveBallSetup(setup);
  return {
    support_mode: resolved.supportMode,
    tee_height_m: resolved.teeHeightM,
    tee_height_unit: "m",
    height_reference: BALL_HEIGHT_REFERENCE,
    ball_center_m: ballCenterPosition(resolved),
  } as const;
}

export function createSimulationRunDocument(
  input: SimulationInput,
  run: SimulationRunTs,
  prescribedTorqueProfile: unknown,
  spatialTargetInput: SpatialTargetTs,
) {
  const setup = resolveBallSetup(input.ballSetup);
  const spatialTarget = createSpatialTarget(spatialTargetInput);
  const targetRecord = JSON.parse(spatialTargetToJson(spatialTarget)) as unknown;
  const clubScrewMotion = run.swing.map((sample) => {
    const twist: Twist6 = [...sample.angularVelocity, ...sample.velocity];
    const motion = analyzeTwist(twist, sample.position);
    return {
      t_s: sample.t,
      motion_kind: motion.kind,
      angular_rate_rad_s: motion.angularRateRadS,
      pitch_m_rad: motion.pitchMPerRad,
      axial_speed_m_s: motion.axialSpeedMps,
      r_isa_m: motion.radiusM,
      axis_direction: motion.axisDirection,
      axis_point_m: motion.axisPointM,
      orbital_velocity_m_s: motion.orbitalVelocityMps,
      axial_velocity_m_s: motion.axialVelocityMps,
      reconstruction_residual_m_s: motion.reconstructionResidualMps,
    };
  });
  return {
    format: SIMULATION_EXPORT_FORMAT,
    model_limitations: SIMULATION_MODEL_LIMITATIONS,
    spatial_target: targetRecord,
    solver_manifest: {
      schema: "swing_sim.solver_manifest",
      schema_version: 1,
      target: targetRecord,
    },
    variation_manifest: {
      schema: "swing_sim.variation_manifest",
      schema_version: 1,
      target: targetRecord,
    },
    parameters: {
      ...input,
      ballSetup: undefined,
      manualAttackAngleDeg: undefined,
      manualClubPathDeg: undefined,
      manualForwardShaftLeanDeg: undefined,
      shaftAxisDatum: undefined,
      ball_setup: ballSetupToJson(setup),
      manual_delivery: {
        attack_angle_deg: run.manualDelivery.manualAttackAngleDeg,
        club_path_deg: run.manualDelivery.manualClubPathDeg,
        forward_shaft_lean_deg: run.manualDelivery.manualForwardShaftLeanDeg,
        shaft_axis_datum: run.manualDelivery.shaftAxisDatum,
      },
    },
    ballSetupMetadata: exportBallSetupMetadata(setup),
    impactOutcome: run.impactOutcome,
    launch: run.launch,
    impactTimeS: run.impactTimeS,
    torqueRun: run.torqueRun,
    prescribedTorqueProfile,
    series: {
      swing: run.swing,
      flight: run.flight,
      clubScrewMotion: { frame: "app/world", units: "SI", rows: clubScrewMotion },
    },
  };
}

const csvCell = (value: unknown): string => {
  const raw = value === null || value === undefined ? "" : String(value);
  // Spreadsheet applications can execute formula-leading text on open. Keep
  // numeric cells numeric, but neutralize user-controlled text cells.
  const text = typeof value === "string" && /^[=+\-@]/.test(raw) ? `'${raw}` : raw;
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
};

/** Export trajectory rows with the exact canonical target manifest on every row. */
export function createSimulationRunCsv(
  run: SimulationRunTs,
  spatialTargetInput: SpatialTargetTs,
): string {
  const target = createSpatialTarget(spatialTargetInput);
  const record = JSON.parse(spatialTargetToJson(target)) as Record<string, unknown>;
  const position = record.position_m as Record<string, number>;
  const targetFields = [
    record.schema,
    record.schema_version,
    record.label,
    record.kind,
    position.x,
    position.elevation,
    position.right,
    record.frame,
    record.source_frame,
    record.units,
    record.elevation_source,
    record.ground_source,
    JSON.stringify(record.tolerance),
  ];
  const header = [
    "target_schema", "target_schema_version", "target_label", "target_kind",
    "target_x_downrange_m", "target_y_up_m", "target_z_right_m", "target_frame",
    "target_source_frame", "target_units", "target_elevation_source", "target_ground_source",
    "target_tolerance_json", "t_s", "x_downrange_m", "y_up_m", "z_right_m",
  ];
  const rows = run.flight.length === 0
    ? [[...targetFields, "", "", "", ""]]
    : run.flight.map((sample) => [
        ...targetFields,
        sample.time,
        ...sample.position,
      ]);
  return [header, ...rows].map((row) => row.map(csvCell).join(",")).join("\n");
}

/** Older run documents had a fixed ground-level ball and therefore migrate to Ground. */
export function ballSetupFromSimulationDocument(value: unknown): BallSetup {
  const data = record(value);
  if (!data) throw new Error("Simulation Settings JSON must be an object.");
  const format = simulationDocumentFormat(data);
  const parameters = record(data?.parameters);
  const rawSetup = parameters?.ballSetup ?? parameters?.ball_setup ?? data?.ball_setup;
  if (rawSetup === undefined) {
    if (format && ((format.web && format.version >= 4) || format.version >= 5)) {
      throw new Error(`Simulation schema version ${format.version} requires ball_setup.`);
    }
    return { ...GROUND_BALL_SETUP };
  }
  return setupFromUnknown(rawSetup);
}

/** Load a canonical v4+ target or migrate a legacy 2D run target/default. */
export function spatialTargetFromSimulationDocument(value: unknown): SpatialTargetTs {
  const data = record(value);
  if (!data) throw new Error("Simulation Settings JSON must be an object.");
  const format = simulationDocumentFormat(data);
  const parameters = record(data.parameters);
  const rawTarget = data.spatial_target ?? parameters?.spatial_target ??
    parameters?.target ?? data.target;
  if (rawTarget === undefined) {
    if (format && ((format.web && format.version >= 4) || format.version >= 5)) {
      throw new Error(`Simulation schema version ${format.version} requires spatial_target.`);
    }
    return spatialTargetFromRegion(DEFAULT_TARGET);
  }
  return spatialTargetFromJson(JSON.stringify(rawTarget));
}
