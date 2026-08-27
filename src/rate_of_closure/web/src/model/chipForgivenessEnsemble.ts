/** Browser adapter from retained swing runs to wedge forgiveness evidence. */

import { getClub, type ClubSpec } from "./club";
import {
  summarizeChipTrials,
  type ChipStudySummaryOptionsTs,
  type ChipStudySummaryTs,
  type ChipTrialCohortTs,
  type ChipTrialRecordTs,
} from "./chipForgiveness";
import type { ImpactScenario } from "./impact";
import { impactKinematics } from "./impactKinematics";
import type { SimulationInput } from "./simulation";
import type { VariationDatasetTs, VariationPlanTs } from "./variation";
import {
  FIRM_FAIRWAY_TURF,
  simulateReducedTurfContact,
  type ReducedTurfStatus,
  type TurfContactProfileTs,
} from "./turfContact";
import {
  defaultSwingVariationInput,
  type SwingVariationResultTs,
  type SwingVariationTrialTs,
} from "./variationSwingEnsemble";
import { wedgeGroundClearance, type ContactSequence } from "./wedgeGroundClearance";
import {
  representativeWedgeForClub,
  type RepresentativeWedge,
} from "./wedgeGroundGeometry";

const TARGET_CARRY_M = 27.432;
const CARRY_TOLERANCE_M = 2;
const LATERAL_TOLERANCE_M = 1;
const FAILURE_PENALTY = 16;
const GROUND_FIRST_PENALTY = 4;
const SIMULTANEOUS_PENALTY = 2;
const MISS_PENALTY = 12;
const MISSING_REQUIRED_METRIC_PENALTY = 12;
const UNSUPPORTED_TURF_PENALTY = 12;

const objectiveIdForTargetCarry = (targetCarryM: number): string => {
  const targetText = targetCarryM.toFixed(9).replace(/\.?0+$/, "");
  return `chip-target-${targetText}m-balanced-v1`;
};

const CHIP_METRIC_NAMES = [
  "carry_m", "lateral_m", "max_height_m", "landing_angle_deg",
  "leading_edge_clearance_at_ball_m", "minimum_pre_ball_clearance_m",
  "ground_after_ball_margin_s", "low_point_clearance_m",
  "delivered_bounce_deg", "path_projected_effective_bounce_deg",
  "reference_aoa_deg", "bounce_utilization_margin_deg",
  "peak_turf_penetration_m", "normal_turf_impulse_n_s",
  "shaft_rotation_rate_rad_s", "shaft_counterfactual_aoa_delta_deg",
  "shaft_shapley_aoa_deg", "shaft_vertical_velocity_share",
  "leading_edge_3d_rate_rad_s", "face_normal_3d_rate_rad_s",
  "leading_edge_relative_arc_heading_rate_rad_s",
] as const;

interface ChipStudyMetadataTs {
  candidateId: string;
  planSchema: "swing-sim.variation-plan/v2";
  coordinateFrame: "app_frame:x_target,y_up,z_right";
  seed: number;
  noiseModelId: string;
  objectiveId: string;
  turfProfileId: string;
  turfCalibrationStatus: TurfContactProfileTs["calibrationStatus"];
  solverId: "rate-of-closure/web-canonical";
  samplingDesign: "iid-monte-carlo-joint";
  inferenceMethodId: "wilson+mulberry32-iid-bootstrap-v1";
  limitations: string;
}

export interface ChipForgivenessStudyTs {
  records: ChipTrialRecordTs[];
  summary: ChipStudySummaryTs;
  metadata: ChipStudyMetadataTs;
  inputs: {
    baseSimulation: SimulationInput;
    wedge: RepresentativeWedge;
    ground: { frameId: "ground_frame:x_target,y_up,z_right"; pointM: [0, 0, 0]; normalUnit: [0, 1, 0] };
    turfProfile: TurfContactProfileTs;
    lossModel: {
      targetCarryM: number;
      carryToleranceM: number;
      lateralToleranceM: number;
      includeTurfPenetration: false;
      penalties: Record<string, number>;
    };
  };
  plan: VariationPlanTs;
  inputNames: string[];
  sampledInputs: number[][];
}

export interface ChipForgivenessEnsembleOptionsTs extends ChipStudySummaryOptionsTs {
  targetCarryM?: number;
  baseSimulation?: SimulationInput;
}

/** Representative 56-degree ground-mode wedge input for the explicit chip study. */
export function defaultChipVariationInput(): SimulationInput {
  const base = defaultSwingVariationInput({ supportMode: "ground", teeHeightM: 0 });
  const wedge = getClub("Sand Wedge");
  return {
    ...base,
    clubheadSpeedMph: 30,
    loftDeg: wedge.loftDeg,
    ballSetup: { supportMode: "ground", teeHeightM: 0 },
    club: {
      ...base.club!,
      headMassKg: wedge.headMassKg,
      moiAboutShaftKgM2: wedge.moiAboutShaftKgM2,
    },
  };
}

function clubForTrial(input: SimulationInput): ClubSpec {
  const base = getClub("Sand Wedge");
  return {
    ...base,
    name: "Variation Wedge",
    loftDeg: input.loftDeg,
    headMassKg: input.club?.headMassKg ?? base.headMassKg,
    moiAboutShaftKgM2: input.club?.moiAboutShaftKgM2 ?? base.moiAboutShaftKgM2,
  };
}

function scenarioForTrial(input: SimulationInput, club: ClubSpec): ImpactScenario {
  return {
    clubheadSpeedMph: input.clubheadSpeedMph,
    omegaPlaneDps: input.omegaDps[1],
    omegaShaftDps: input.omegaDps[0],
    lieAngleDeg: club.lieDeg,
    comToFaceMm: club.cgDepthM * 1_000,
    impactOffsetToeMm: input.impactOffsetToeMm,
    impactOffsetHighMm: input.impactOffsetHighMm,
    contactDurationUs: 450,
  };
}

function cohortForSequence(sequence: ContactSequence): ChipTrialCohortTs {
  return sequence === "simultaneous" ? "simultaneous_or_grazing" : sequence;
}

function cohortPenalty(cohort: ChipTrialCohortTs): number {
  if (cohort === "ground_first") return GROUND_FIRST_PENALTY;
  if (cohort === "simultaneous_or_grazing") return SIMULTANEOUS_PENALTY;
  if (cohort === "ground_only_miss" || cohort === "no_contact_miss") return MISS_PENALTY;
  if (cohort === "numerical_failure") return FAILURE_PENALTY;
  return 0;
}

function lossFor(
  cohort: ChipTrialCohortTs,
  metrics: Record<string, number | null>,
  targetCarryM: number,
  turfStatus: ReducedTurfStatus | null,
): { loss: number; constraintViolated: boolean } {
  const carry = metrics.carry_m;
  const lateral = metrics.lateral_m;
  let loss = cohortPenalty(cohort);
  if (carry !== null) loss += ((carry - targetCarryM) / CARRY_TOLERANCE_M) ** 2;
  if (lateral !== null) loss += (lateral / LATERAL_TOLERANCE_M) ** 2;
  const requiredMissing = ["ball_first", "ball_only", "simultaneous_or_grazing"]
    .includes(cohort) && (carry === null || lateral === null);
  if (requiredMissing) loss += MISSING_REQUIRED_METRIC_PENALTY;
  const turfUnsupported = turfStatus === "outside_calibrated_domain"
    || turfStatus === "step_limit";
  if (turfUnsupported) loss += UNSUPPORTED_TURF_PENALTY;
  const constraintViolated = [
    "ground_first", "simultaneous_or_grazing", "ground_only_miss",
    "no_contact_miss", "numerical_failure",
  ].includes(cohort)
    || (metrics.ground_after_ball_margin_s ?? 1) <= 0
    || requiredMissing
    || turfUnsupported;
  return { loss, constraintViolated };
}

function failedRecord(trial: SwingVariationTrialTs): ChipTrialRecordTs {
  return {
    trialIndex: trial.trialIndex,
    cohort: "numerical_failure",
    loss: FAILURE_PENALTY,
    constraintViolated: true,
    metrics: {},
    diagnostic: trial.error ?? "Numerical failure without a diagnostic.",
    turfContactStatus: null,
  };
}

function evaluatedRecord(
  trial: SwingVariationTrialTs, targetCarryM: number,
): ChipTrialRecordTs {
  if (trial.run === null) return failedRecord(trial);
  const club = clubForTrial(trial.input);
  const scenario = scenarioForTrial(trial.input, club);
  const ground = wedgeGroundClearance(trial.run, scenario, club);
  if (ground === null) throw new Error("chip forgiveness requires wedge geometry");
  const impact = impactKinematics(trial.run, scenario, club);
  const groundEvent = ground.firstGroundContact;
  const turf = groundEvent === null ? null : simulateReducedTurfContact(
    FIRM_FAIRWAY_TURF,
    [
      groundEvent.tangentialVelocityMps[0],
      groundEvent.normalVelocityMps,
      groundEvent.tangentialVelocityMps[2],
    ],
    trial.input.club?.headMassKg ?? club.headMassKg,
  );
  const landing = trial.run.flight[trial.run.flight.length - 1]?.position;
  const metrics: Record<string, number | null> = Object.fromEntries(
    CHIP_METRIC_NAMES.map((name) => [name, null]),
  );
  Object.assign(metrics, {
    carry_m: trial.run.launch?.carryM ?? null,
    lateral_m: landing?.[2] ?? null,
    max_height_m: trial.run.launch?.maxHeightM ?? null,
    landing_angle_deg: trial.run.launch?.landingAngleDeg ?? null,
    leading_edge_clearance_at_ball_m: ground.metrics.leadingEdgeClearanceAtBallM,
    minimum_pre_ball_clearance_m: ground.metrics.minimumPreBallClearanceM,
    ground_after_ball_margin_s: ground.metrics.groundAfterBallTimeMarginS,
    low_point_clearance_m: ground.lowPoint.worldPointM[1],
    delivered_bounce_deg: ground.metrics.deliveredBounceDegAtBall,
    path_projected_effective_bounce_deg:
      ground.metrics.pathProjectedEffectiveBounceDegAtBall,
    reference_aoa_deg: ground.metrics.referenceAoaDegAtBall,
    bounce_utilization_margin_deg: ground.metrics.bounceUtilizationMarginDeg,
    peak_turf_penetration_m: turf?.peakPenetrationM ?? null,
    normal_turf_impulse_n_s: turf?.normalImpulseNs ?? null,
    shaft_rotation_rate_rad_s: impact.shaftRotationRateDps * Math.PI / 180,
    shaft_counterfactual_aoa_delta_deg: impact.shaftAoaContributionDeg,
    shaft_shapley_aoa_deg: impact.shaftAoaShapleyDeg,
    shaft_vertical_velocity_share: impact.shaftVerticalVelocityShare,
    leading_edge_3d_rate_rad_s: impact.leadingEdgeRateDps * Math.PI / 180,
    face_normal_3d_rate_rad_s: impact.faceNormalRateDps * Math.PI / 180,
  });
  const cohort = cohortForSequence(ground.sequence);
  const turfStatus = turf?.status ?? null;
  const decision = lossFor(cohort, metrics, targetCarryM, turfStatus);
  return {
    trialIndex: trial.trialIndex,
    cohort,
    loss: decision.loss,
    constraintViolated: decision.constraintViolated,
    metrics,
    turfContactStatus: turfStatus,
  };
}

/** Analyze retained browser trials; failed post-processing stays an explicit failure. */
export function analyzeChipForgivenessEnsemble(
  ensemble: SwingVariationResultTs,
  options: ChipForgivenessEnsembleOptionsTs = {},
): ChipForgivenessStudyTs {
  const targetCarryM = options.targetCarryM ?? TARGET_CARRY_M;
  if (!Number.isFinite(targetCarryM) || targetCarryM <= 0) {
    throw new RangeError("targetCarryM must be finite and > 0");
  }
  const baseSimulation = options.baseSimulation ?? defaultChipVariationInput();
  const wedge = representativeWedgeForClub(clubForTrial(baseSimulation));
  if (wedge === null) throw new Error("chip forgiveness requires a wedge candidate");
  const records = ensemble.runs.map((trial) => {
    if (trial.status === "numerical_failure") return failedRecord(trial);
    try {
      return evaluatedRecord(trial, targetCarryM);
    } catch (error) {
      return failedRecord({
        ...trial,
        status: "numerical_failure",
        run: null,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  });
  return {
    records,
    summary: summarizeChipTrials(records, {
      ...options,
      turfCalibrationStatus: FIRM_FAIRWAY_TURF.calibrationStatus,
    }),
    metadata: {
      candidateId: wedge.headId,
      planSchema: "swing-sim.variation-plan/v2",
      coordinateFrame: "app_frame:x_target,y_up,z_right",
      seed: options.seed ?? ensemble.dataset.plan.seed,
      noiseModelId: ensemble.dataset.plan.noise
        .map((spec) => spec.specId ?? spec.variableKey).join("+"),
      objectiveId: objectiveIdForTargetCarry(targetCarryM),
      turfProfileId: FIRM_FAIRWAY_TURF.profileId,
      turfCalibrationStatus: FIRM_FAIRWAY_TURF.calibrationStatus,
      solverId: "rate-of-closure/web-canonical",
      samplingDesign: "iid-monte-carlo-joint",
      inferenceMethodId: "wilson+mulberry32-iid-bootstrap-v1",
      limitations: "Conditional on the retained joint Monte-Carlo plan and objective. Illustrative reduced single-point firm-fairway turf is diagnostic only; no force-coupled swing replay, divot shape, grass fracture, granular flow, or universal bounce claim.",
    },
    inputs: {
      baseSimulation,
      wedge,
      ground: {
        frameId: "ground_frame:x_target,y_up,z_right",
        pointM: [0, 0, 0],
        normalUnit: [0, 1, 0],
      },
      turfProfile: FIRM_FAIRWAY_TURF,
      lossModel: {
        targetCarryM,
        carryToleranceM: CARRY_TOLERANCE_M,
        lateralToleranceM: LATERAL_TOLERANCE_M,
        includeTurfPenetration: false,
        penalties: {
          groundFirst: GROUND_FIRST_PENALTY,
          simultaneous: SIMULTANEOUS_PENALTY,
          miss: MISS_PENALTY,
          numericalFailure: FAILURE_PENALTY,
          missingRequiredMetric: MISSING_REQUIRED_METRIC_PENALTY,
          unsupportedTurf: UNSUPPORTED_TURF_PENALTY,
        },
      },
    },
    plan: ensemble.dataset.plan,
    inputNames: ensemble.dataset.inputNames,
    sampledInputs: ensemble.dataset.inputs,
  };
}

const csvCell = (value: string | number | boolean | null | undefined): string => {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
};

const strictNumberReplacer = (_key: string, value: unknown): unknown => {
  if (typeof value === "number" && !Number.isFinite(value)) {
    throw new RangeError("chip forgiveness export contains a nonfinite number");
  }
  return value;
};

const rejectNonfiniteStudyNumbers = (study: ChipForgivenessStudyTs): void => {
  JSON.stringify(study, strictNumberReplacer);
};

/** Serialize complete plan, sampled population, records, and qualification evidence. */
export function chipForgivenessStudyToJson(study: ChipForgivenessStudyTs): string {
  return JSON.stringify({ schemaVersion: 1, ...study }, strictNumberReplacer, 2);
}

/** Serialize one all-trial row per record, retaining unavailable values as blanks. */
export function chipForgivenessStudyToCsv(study: ChipForgivenessStudyTs): string {
  rejectNonfiniteStudyNumbers(study);
  const metricNames = [...new Set(study.records.flatMap((record) =>
    Object.keys(record.metrics)))].sort();
  const header = [
    "candidate_id", "objective_id", "turf_profile_id", "sampling_design",
    "trial", "cohort", "loss", "constraint_violated", "diagnostic", "turf_status",
    ...study.inputNames, ...metricNames,
  ];
  const rows = study.records.map((record) => [
    study.metadata.candidateId,
    study.metadata.objectiveId,
    study.metadata.turfProfileId,
    study.metadata.samplingDesign,
    record.trialIndex,
    record.cohort,
    record.loss,
    record.constraintViolated,
    record.diagnostic,
    record.turfContactStatus,
    ...study.sampledInputs[record.trialIndex],
    ...metricNames.map((name) => record.metrics[name]),
  ]);
  // ⚡ Bolt Optimization: Replace chained array .map().join() with a single-pass loop
  // to eliminate intermediate array allocations and reduce GC pressure for large dataset exports.
  const allRows = [header, ...rows];
  let csvText = "";
  for (let i = 0; i < allRows.length; i++) {
    if (i > 0) csvText += "\n";
    const row = allRows[i];
    for (let j = 0; j < row.length; j++) {
      if (j > 0) csvText += ",";
      csvText += csvCell(row[j]);
    }
  }
  return csvText + "\n";
}

/** Project decision and physical metrics onto the shared scatter/marginal schema. */
export function chipForgivenessVariationDataset(
  study: ChipForgivenessStudyTs,
): VariationDatasetTs {
  const metricNames = [...new Set(study.records.flatMap((record) =>
    Object.keys(record.metrics)))].sort();
  const outputNames = ["loss", "constraint_violated", ...metricNames];
  return {
    plan: study.plan,
    inputNames: study.inputNames,
    inputs: study.sampledInputs,
    outputNames,
    outputs: study.records.map((record) => [
      record.loss,
      record.constraintViolated ? 1 : 0,
      ...metricNames.map((name) => record.metrics[name] ?? null),
    ]),
    success: study.records.map((record) => record.cohort !== "numerical_failure"),
  };
}
