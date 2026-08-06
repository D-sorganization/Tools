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

export const BALL_SETUP_STORAGE_KEY = "rate_of_closure.ball_setup.web/v1";
export const SIMULATION_EXPORT_FORMAT = "rate_of_closure.simulation_run.web/3";

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
  prescribedTorqueProfile: unknown = null,
) {
  const setup = resolveBallSetup(input.ballSetup);
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
    parameters: {
      ...input,
      ballSetup: undefined,
      ball_setup: ballSetupToJson(setup),
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

/** Older run documents had a fixed ground-level ball and therefore migrate to Ground. */
export function ballSetupFromSimulationDocument(value: unknown): BallSetup {
  const data = record(value);
  if (!data) throw new Error("Simulation JSON must be an object.");
  if (data.format !== undefined) {
    const match = String(data.format).match(/^rate_of_closure\.simulation_run(?:\.web)?\/(\d+)$/);
    if (!match) throw new Error(`Unsupported simulation format: ${String(data.format)}.`);
    const version = Number(match[1]);
    const web = String(data.format).includes(".web/");
    if (version < 1 || version > (web ? 3 : 2)) {
      throw new Error(`Unsupported simulation schema version ${version}.`);
    }
  }
  const parameters = record(data?.parameters);
  const rawSetup = parameters?.ballSetup ?? parameters?.ball_setup ?? data?.ball_setup;
  if (rawSetup === undefined) {
    return { ...GROUND_BALL_SETUP };
  }
  return setupFromUnknown(rawSetup);
}
