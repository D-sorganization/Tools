/** Physical options and launch-origin authority for regional-ground jobs. */
import { canonicalGroundJson } from "./flightGroundContract";
import type {
  GroundProvenance,
  GroundSurfaceProfile,
  GroundVec3,
} from "./flightGroundTypes";
import {
  exact,
  finiteRaw,
  integer,
  parseProvenance,
  positive,
  record,
  text,
  vector,
} from "./flightGroundValidation";
import {
  parseGroundRegionalMaterialPlanRequest,
  type GroundRegionalMaterialPlanRequest,
} from "./groundRegionalPlan";
import { sha256Text } from "./sha256";

const QUALIFICATION_SCHEMA =
  "rate-of-closure/regional-ground-launch-origin-qualification/v1";
const QUALIFIER_ID = "tools.rate_of_closure.launch-origin-qualification";
const QUALIFIER_VERSION = "1.0.0";
const MAX_STEPS = 1_000_000;
const MAX_TRANSITIONS = 4_096;
const STANDARD_GRAVITY = [0, -9.80665, 0] as const;

const OPTION_FIELDS = ["settings", "source_revision"] as const;
const SETTING_FIELDS = [
  "integration_step_s", "max_steps", "max_surface_transitions",
  "velocity_tolerance_m_s_decimal", "angular_tolerance_rad_s_decimal",
  "slip_tolerance_m_s_decimal", "time_tolerance_s_decimal", "gravity_m_s2",
  "model_id", "model_version",
] as const;

export interface SkidRollSettingsWire {
  readonly integration_step_s: number;
  readonly max_steps: number;
  readonly max_surface_transitions: number;
  readonly velocity_tolerance_m_s_decimal: string;
  readonly angular_tolerance_rad_s_decimal: string;
  readonly slip_tolerance_m_s_decimal: string;
  readonly time_tolerance_s_decimal: string;
  readonly gravity_m_s2: GroundVec3;
  readonly model_id: string;
  readonly model_version: string;
}

export interface RegionalExecutionOptionsWire {
  readonly settings: SkidRollSettingsWire;
  readonly source_revision: string;
}

export interface LaunchQualificationInput {
  readonly ball_radius_m: number;
  readonly ball_setup: Readonly<Record<string, unknown>>;
}

export interface QualifiedExecutionAuthority {
  readonly options: RegionalExecutionOptionsWire;
  readonly plan: GroundRegionalMaterialPlanRequest;
  readonly planSha256: string;
}

const boundedPositive = (value: unknown, name: string): number => {
  const parsed = positive(value, name);
  if (parsed > 1) throw new RangeError(name + " exceeds 1");
  return parsed;
};

const boundedInteger = (
  value: unknown,
  name: string,
  maximum: number,
): number => {
  const parsed = integer(value, name, 1);
  if (parsed > maximum) throw new RangeError(name + " exceeds " + maximum);
  return parsed;
};

const canonicalDecimal = (value: unknown, name: string): string => {
  const decimal = text(value, name);
  const parsed = Number(decimal);
  const canonical = parsed.toFixed(17).replace(/0+$/, "").replace(/\.$/, "");
  if (!(parsed > 0 && parsed <= 1) || canonical !== decimal) {
    throw new RangeError(name + " must be canonical positive decimal text");
  }
  return decimal;
};

const parseSettings = (value: unknown): SkidRollSettingsWire => {
  const item = record(value, "skid_roll_settings");
  exact(item, SETTING_FIELDS, "skid_roll_settings");
  const gravity = vector(item.gravity_m_s2, "gravity_m_s2");
  if (canonicalGroundJson(gravity) !== canonicalGroundJson(STANDARD_GRAVITY)) {
    throw new RangeError("gravity_m_s2 must equal versioned standard gravity");
  }
  return Object.freeze({
    integration_step_s: boundedPositive(item.integration_step_s, "integration_step_s"),
    max_steps: boundedInteger(item.max_steps, "max_steps", MAX_STEPS),
    max_surface_transitions: boundedInteger(
      item.max_surface_transitions,
      "max_surface_transitions",
      MAX_TRANSITIONS,
    ),
    velocity_tolerance_m_s_decimal: canonicalDecimal(
      item.velocity_tolerance_m_s_decimal,
      "velocity_tolerance_m_s_decimal",
    ),
    angular_tolerance_rad_s_decimal: canonicalDecimal(
      item.angular_tolerance_rad_s_decimal,
      "angular_tolerance_rad_s_decimal",
    ),
    slip_tolerance_m_s_decimal: canonicalDecimal(
      item.slip_tolerance_m_s_decimal,
      "slip_tolerance_m_s_decimal",
    ),
    time_tolerance_s_decimal: canonicalDecimal(
      item.time_tolerance_s_decimal,
      "time_tolerance_s_decimal",
    ),
    gravity_m_s2: gravity,
    model_id: text(item.model_id, "skid/roll model_id"),
    model_version: text(item.model_version, "skid/roll model_version"),
  });
};

const parseOptions = (value: unknown): RegionalExecutionOptionsWire => {
  const item = record(value, "regional_execution_options");
  exact(item, OPTION_FIELDS, "regional_execution_options");
  return Object.freeze({
    settings: parseSettings(item.settings),
    source_revision: text(item.source_revision, "source_revision"),
  });
};

const qualificationDigest = (
  source: GroundRegionalMaterialPlanRequest,
  launch: LaunchQualificationInput,
  transferSurface: GroundSurfaceProfile,
): string => sha256Text(canonicalGroundJson({
  schema_version: QUALIFICATION_SCHEMA,
  source_plan_sha256: sha256Text(canonicalGroundJson(source)),
  transfer_surface: transferSurface,
  ball_radius_m: launch.ball_radius_m,
  ball_setup: launch.ball_setup,
}));

const expectedQualifiedPlan = (
  source: GroundRegionalMaterialPlanRequest,
  launch: LaunchQualificationInput,
  transferSurface: GroundSurfaceProfile,
  sourceRevision: string,
): GroundRegionalMaterialPlanRequest => {
  if (canonicalGroundJson(source.base_surface) !== canonicalGroundJson(transferSurface)) {
    throw new RangeError("source regional base surface must match transfer surface");
  }
  const setupHeight = finiteRaw(launch.ball_setup.tee_height_m, "tee_height_m");
  if (setupHeight < 0) throw new RangeError("tee_height_m must be nonnegative");
  const translation = -launch.ball_radius_m - setupHeight;
  const provenance: GroundProvenance = parseProvenance({
    producer: QUALIFIER_ID,
    producer_version: QUALIFIER_VERSION,
    source_revision: sourceRevision,
    input_sha256: qualificationDigest(source, launch, transferSurface),
  });
  return parseGroundRegionalMaterialPlanRequest({
    ...source,
    base_surface: {
      ...transferSurface,
      height_m: transferSurface.height_m + translation,
    },
    axis_origin_m: [
      source.axis_origin_m[0],
      source.axis_origin_m[1] + translation,
      source.axis_origin_m[2],
    ],
    regions: source.regions.map((region) => ({
      ...region,
      surface: {
        ...region.surface,
        height_m: region.surface.height_m + translation,
      },
    })),
    provenance,
  });
};

/** Parse and recompute all physical and launch-coordinate job authority. */
export const parseQualifiedExecutionAuthority = (
  rawOptions: unknown,
  rawPlan: unknown,
  rawPlanSha256: unknown,
  sourcePlan: GroundRegionalMaterialPlanRequest,
  launch: LaunchQualificationInput,
  transferSurface: GroundSurfaceProfile,
): QualifiedExecutionAuthority => {
  const options = parseOptions(rawOptions);
  const plan = parseGroundRegionalMaterialPlanRequest(rawPlan);
  const expected = expectedQualifiedPlan(
    sourcePlan,
    launch,
    transferSurface,
    options.source_revision,
  );
  if (canonicalGroundJson(plan) !== canonicalGroundJson(expected)) {
    throw new RangeError("qualified regional plan must match launch-origin translation");
  }
  const planSha256 = text(rawPlanSha256, "qualified_plan_sha256");
  if (!/^[0-9a-f]{64}$/.test(planSha256) ||
      planSha256 !== sha256Text(canonicalGroundJson(plan))) {
    throw new RangeError("qualified_plan_sha256 must match the qualified plan");
  }
  return Object.freeze({ options, plan, planSha256 });
};
