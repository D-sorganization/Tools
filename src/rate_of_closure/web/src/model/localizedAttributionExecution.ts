/** Genuine explicit baseline/one-source localized attribution execution. */

import { runSimulation } from "./simulation";
import type { SimulationInput, SimulationRunTs } from "./simulation";
import { attributionAuthorityFromValue, type AttributionAuthorityTs } from "./localizedAttribution";
import { record, stable, finite } from "./localizedAttributionContract";
import { resolvedBase } from "./variationSampling";
import { localizedTorqueJointId } from "./variationRegistry";
import { planFromJson, planToJson, stableSpecId, validatePlan, type VariationPlanTs } from "./variationSchema";
import { defaultSwingVariationInput, swingVariationInputForValues } from "./variationSwingInput";
import { SWING_VARIATION_OUTPUT_NAMES, swingOutputRow, type SwingTrialStatusTs } from "./variationSwingEnsemble";

export const PAIRED_REQUEST_SCHEMA_ID = "rate-of-closure/react-localized-paired-request";
export const PAIRED_RESULT_SCHEMA_ID = "rate-of-closure/react-localized-paired-result";
export const PAIRED_SCHEMA_VERSION = 1;
export const MAX_PAIRED_SOURCES = 2;

/** Explicit marker for a solver-declared numerical failure, never a contract defect. */
export class LocalizedPairedNumericalExecutionError extends Error {
  constructor(message: string) { super(message); this.name = "LocalizedPairedNumericalExecutionError"; }
}
export type LocalizedPairedTrialExecutor = (input: SimulationInput) => SimulationRunTs;

export interface LocalizedPairedRequestTs {
  schemaId: typeof PAIRED_REQUEST_SCHEMA_ID;
  schemaVersion: 1;
  designId: string;
  sourcePlanJson: string;
  interventionDeltasNm: Readonly<Record<string, number>>;
  statePointId: "swing.clubhead.reference";
  stateTimeS: number;
}
export interface LocalizedPairedProgressTs { completedRuns: number; totalRuns: number }
export interface LocalizedPairedResultTs {
  schemaId: typeof PAIRED_RESULT_SCHEMA_ID;
  schemaVersion: 1;
  requestIdentity: string;
  designIdentity: string;
  authority: AttributionAuthorityTs;
  explicitRows: readonly (readonly number[])[];
  trials: readonly LocalizedPairedTrialEvidenceTs[];
}
export interface LocalizedPairedTrialEvidenceTs {
  status: SwingTrialStatusTs;
  state: readonly number[] | null;
  outputs: readonly (number | null)[];
}

export function localizedPairedPlan(plan: VariationPlanTs): VariationPlanTs {
  validatePlan(plan);
  const noise = plan.noise.filter((spec) => localizedTorqueJointId(spec.variableKey) !== null);
  if (plan.mode !== "swing" || noise.length < 1 || noise.length > MAX_PAIRED_SOURCES) {
    throw new Error("paired study requires one or two localized shoulder/wrist sources");
  }
  return {
    ...plan, baseVariables: resolvedBase(plan), noise, nRuns: noise.length * 2, groups: [],
  };
}

export function buildLocalizedPairedRequest(
  plan: VariationPlanTs,
  interventionDeltasNm: Readonly<Record<string, number>>,
  stateTimeS: number,
): LocalizedPairedRequestTs {
  const paired = localizedPairedPlan(plan);
  return normalizePairedRequest({
    schemaId: PAIRED_REQUEST_SCHEMA_ID, schemaVersion: 1,
    designId: `react.localized-paired.${paired.noise.map(stableSpecId).join("+")}`,
    sourcePlanJson: planToJson(paired), interventionDeltasNm,
    statePointId: "swing.clubhead.reference", stateTimeS,
  });
}

const canonical = (value: unknown): string => {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  const object = value as Record<string, unknown>;
  return `{${Object.keys(object).sort().map((key) =>
    `${JSON.stringify(key)}:${canonical(object[key])}`).join(",")}}`;
};

const sha256 = async (value: unknown): Promise<string> => {
  const bytes = new TextEncoder().encode(canonical(value));
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
};

export const pairedSourcePlan = (request: LocalizedPairedRequestTs): VariationPlanTs =>
  planFromJson(request.sourcePlanJson);

export function normalizePairedRequest(value: unknown): LocalizedPairedRequestTs {
  const raw = record(value, [
    "schemaId", "schemaVersion", "designId", "sourcePlanJson",
    "interventionDeltasNm", "statePointId", "stateTimeS",
  ], "localized paired request");
  if (raw.schemaId !== PAIRED_REQUEST_SCHEMA_ID || raw.schemaVersion !== 1) {
    throw new Error("invalid localized paired request schema");
  }
  if (typeof raw.sourcePlanJson !== "string") throw new Error("sourcePlanJson must be text");
  const plan = planFromJson(raw.sourcePlanJson);
  if (raw.sourcePlanJson !== planToJson(plan)) throw new Error("sourcePlanJson must be canonical");
  validatePlan(plan);
  if (plan.mode !== "swing" || (plan.groups?.length ?? 0) !== 0 ||
      plan.noise.length < 1 || plan.noise.length > MAX_PAIRED_SOURCES) {
    throw new Error("paired study requires one or two ungrouped localized swing sources");
  }
  if (plan.nRuns !== plan.noise.length * 2) {
    throw new Error("paired study run count must equal two times the source count");
  }
  const deltaRaw = record(raw.interventionDeltasNm,
    plan.noise.map(stableSpecId), "intervention delta roster");
  const deltas: Record<string, number> = {};
  plan.noise.forEach((spec) => {
    const specId = stableSpecId(spec);
    const joint = localizedTorqueJointId(spec.variableKey);
    if (joint === null || spec.timeWindowS == null || spec.pointIds?.[0] !== joint) {
      throw new Error("paired study supports only exact localized torque loci");
    }
    const delta = finite(deltaRaw[specId], `delta ${specId}`);
    if (delta === 0) throw new Error("intervention delta must be nonzero");
    deltas[specId] = delta;
  });
  if (raw.statePointId !== "swing.clubhead.reference") {
    throw new Error("state point must be swing.clubhead.reference");
  }
  const stateTimeS = finite(raw.stateTimeS, "state time");
  if (!(stateTimeS >= 0 && stateTimeS <= 1.5 && Number.isInteger(stateTimeS * 1000))) {
    throw new Error("state time must lie on the current 0..1.5 s, 1 ms swing grid");
  }
  return Object.freeze({
    schemaId: PAIRED_REQUEST_SCHEMA_ID, schemaVersion: 1,
    designId: stable(raw.designId, "designId"), sourcePlanJson: raw.sourcePlanJson,
    interventionDeltasNm: Object.freeze(deltas),
    statePointId: "swing.clubhead.reference", stateTimeS,
  });
}

const identityPayload = (request: LocalizedPairedRequestTs): unknown => ({
  schema: "rate-of-closure/react-localized-paired-request-identity@1",
  ...request,
});

export async function pairedRequestIdentity(request: LocalizedPairedRequestTs): Promise<string> {
  return sha256(identityPayload(normalizePairedRequest(request)));
}

export async function pairedDesignIdentity(
  request: LocalizedPairedRequestTs,
  requestIdentity: string,
): Promise<string> {
  return sha256({
    schema: "rate-of-closure/localized-attribution-design@1",
    design_id: request.designId,
    request_identity: requestIdentity,
  });
}

const targetRows = (request: LocalizedPairedRequestTs): Record<string, unknown>[] => {
  const state = ["position_x_m", "position_y_m", "position_z_m"].map((name) => ({
    target_id: `state.clubhead.${name}.${request.stateTimeS}`,
    kind: "state", name, unit: "m", convention: "app-frame-cartesian-v1",
    time_s: request.stateTimeS, point_id: request.statePointId,
    coordinate_frame: "app_frame:x_target,y_up,z_right",
  }));
  const scalar = [
    ["impact", "impact_time_s", "s"], ["impact", "clubhead_speed_mps", "m/s"],
    ["impact", "spin_loft_deg", "deg"], ["impact", "face_to_path_deg", "deg"],
    ["impact", "spin_axis_tilt_deg", "deg"], ["shot", "ball_speed_mph", "mph"],
    ["shot", "launch_angle_deg", "deg"], ["shot", "launch_azimuth_deg", "deg"],
    ["shot", "spin_rpm", "rpm"], ["shot", "carry_m", "m"],
    ["shot", "lateral_m", "m"], ["shot", "max_height_m", "m"],
    ["shot", "flight_time_s", "s"], ["shot", "landing_angle_deg", "deg"],
  ].map(([kind, name, unit]) => ({
    target_id: `${kind}.${name}`, kind, name, unit,
    convention: kind === "impact" ? "rate-of-closure-impact-v1" : "rate-of-closure-flight-v1",
    time_s: null, point_id: null, coordinate_frame: null,
  }));
  return [...state, ...scalar];
};

const executeRows = (
  request: LocalizedPairedRequestTs,
  onProgress: (progress: LocalizedPairedProgressTs) => void,
  executor: LocalizedPairedTrialExecutor,
): { trials: LocalizedPairedTrialEvidenceTs[]; rows: number[][]; plan: VariationPlanTs } => {
  const source = pairedSourcePlan(request);
  const plan = { ...source, nRuns: source.noise.length * 2, groups: [] };
  const base = resolvedBase(plan);
  const rows: number[][] = [];
  source.noise.forEach((spec, sourceIndex) => {
    const baseline = source.noise.map((item) => base[item.variableKey]);
    const perturbed = [...baseline];
    perturbed[sourceIndex] += request.interventionDeltasNm[stableSpecId(spec)];
    rows.push(baseline, perturbed);
  });
  const trials = rows.map((row, trialIndex): LocalizedPairedTrialEvidenceTs => {
    const values = { ...base };
    source.noise.forEach((spec, column) => { values[spec.variableKey] = row[column]; });
    const { input } = swingVariationInputForValues(
      plan, values, defaultSwingVariationInput(plan.ballSetup),
    );
    let run: SimulationRunTs;
    try {
      run = executor(input);
    } catch (error) {
      if (!(error instanceof LocalizedPairedNumericalExecutionError)) throw error;
      onProgress({ completedRuns: trialIndex + 1, totalRuns: rows.length });
      return { status: "numerical_failure", state: null,
        outputs: SWING_VARIATION_OUTPUT_NAMES.map(() => null) };
    }
    const sample = run.swing.find((item) => item.t === request.stateTimeS);
    const status: SwingTrialStatusTs = run.impactOutcome.status === "hit"
      ? "evaluated_hit" : "evaluated_no_impact";
    const evidence: LocalizedPairedTrialEvidenceTs = {
      status, state: sample?.joints[sample.joints.length - 1] ?? null,
      outputs: swingOutputRow(run, input) };
    onProgress({ completedRuns: trialIndex + 1, totalRuns: rows.length });
    return evidence;
  });
  return { trials, rows, plan };
};

const targetValue = (
  trial: LocalizedPairedTrialEvidenceTs,
  target: Record<string, unknown>,
): number | null => {
  if (trial.status === "numerical_failure") return null;
  if (target.kind === "state") {
    const axes: Readonly<Record<string, number>> = {
      position_x_m: 0, position_y_m: 1, position_z_m: 2,
    };
    const axis = axes[String(target.name)];
    if (axis === undefined) return null;
    return trial.state?.[axis] ?? null;
  }
  const index = SWING_VARIATION_OUTPUT_NAMES.indexOf(target.name as never);
  return index < 0 ? null : trial.outputs[index];
};

export async function executeLocalizedPairedWork(
  rawRequest: LocalizedPairedRequestTs,
  onProgress: (progress: LocalizedPairedProgressTs) => void,
  executor: LocalizedPairedTrialExecutor = runSimulation,
): Promise<LocalizedPairedResultTs> {
  const request = normalizePairedRequest(rawRequest);
  const { trials, rows, plan } = executeRows(request, onProgress, executor);
  const requestIdentity = await pairedRequestIdentity(request);
  const designIdentity = await pairedDesignIdentity(request, requestIdentity);
  const authority = localizedAuthorityFromEvidence(
    request, plan, rows, trials, designIdentity,
  );
  return Object.freeze({
    schemaId: PAIRED_RESULT_SCHEMA_ID, schemaVersion: 1,
    requestIdentity, designIdentity, authority,
    explicitRows: rows.map((row) => Object.freeze([...row])),
    trials: trials.map((trial) => Object.freeze({
      status: trial.status,
      state: trial.state === null ? null : Object.freeze([...trial.state]),
      outputs: Object.freeze([...trial.outputs]),
    })),
  });
}

export function localizedAuthorityFromEvidence(
  request: LocalizedPairedRequestTs,
  plan: VariationPlanTs,
  rows: readonly (readonly number[])[],
  trials: readonly LocalizedPairedTrialEvidenceTs[],
  designIdentity: string,
): AttributionAuthorityTs {
  const sources = plan.noise.map((spec) => ({
    spec_id: stableSpecId(spec), variable_key: spec.variableKey,
    joint_id: localizedTorqueJointId(spec.variableKey), time_window_s: spec.timeWindowS,
    unit: "N·m",
  }));
  const targets = targetRows(request);
  const pairs = sources.map((source, index) => ({
    source_spec_id: source.spec_id, baseline_trial_index: index * 2,
    perturbed_trial_index: index * 2 + 1, baseline_status: trials[index * 2].status,
    perturbed_status: trials[index * 2 + 1].status,
    baseline_source_value: rows[index * 2][index],
    perturbed_source_value: rows[index * 2 + 1][index],
  }));
  const observations = pairs.flatMap((pair) => targets.map((target) => {
    const baseline = targetValue(trials[pair.baseline_trial_index], target);
    const perturbed = targetValue(trials[pair.perturbed_trial_index], target);
    const failed = pair.baseline_status === "numerical_failure" || pair.perturbed_status === "numerical_failure";
    const noImpact = target.kind !== "state" &&
      (pair.baseline_status === "evaluated_no_impact" || pair.perturbed_status === "evaluated_no_impact");
    const available = !failed && !noImpact && baseline !== null && perturbed !== null;
    return { ...pair, target_id: target.target_id,
      baseline_target_value: available ? baseline : null,
      perturbed_target_value: available ? perturbed : null,
      response: available ? perturbed - baseline : null,
      availability: failed ? "numerical_failure" : noImpact ? "no_impact_unavailable" :
        available ? "available" : "nonfinite_unavailable" };
  }));
  const authority = attributionAuthorityFromValue({
    schema_id: "rate-of-closure/localized-attribution-authority", schema_version: 1,
    authority_id: `paired-attribution.${designIdentity}`,
    interpretation: "paired-planted-intervention-noncausal",
    sources, targets, pairs, observations,
  });
  return authority;
}
