import {
  MAX_OBSERVATIONS, MAX_PAIRS, MAX_SOURCES, MAX_TARGETS, TARGET_REGISTRY,
  boundedArray, deepFreeze, finite, index, nullableFinite, record,
  requireAuthorityShape, responseMatches, stable, type RecordValue,
} from "./localizedAttributionContract";

export const ATTRIBUTION_AUTHORITY_SCHEMA_ID =
  "rate-of-closure/localized-attribution-authority";
export const ATTRIBUTION_VIEW_SCHEMA_ID = "rate-of-closure/localized-attribution-view";
export const ATTRIBUTION_SCHEMA_VERSION = 1;
export const ATTRIBUTION_INTERPRETATION = "paired-planted-intervention-noncausal";
export const ATTRIBUTION_CAVEAT =
  "Paired planted-intervention response only; this view does not infer causality " +
  "from Monte Carlo scatter or correlation.";

type TrialStatus = "evaluated_hit" | "evaluated_no_impact" | "numerical_failure";
type Availability = "available" | "no_impact_unavailable" | "numerical_failure" |
  "nonfinite_unavailable";
type TargetKind = "state" | "impact" | "shot";

export interface AttributionSourceTs {
  specId: string;
  variableKey: string;
  jointId: "joint.shoulder" | "joint.wrist";
  timeWindowS: readonly [number, number];
  unit: "N·m";
}
export interface AttributionTargetTs {
  targetId: string;
  kind: TargetKind;
  name: string;
  unit: string;
  convention: string;
  timeS: number | null;
  pointId: string | null;
  coordinateFrame: string | null;
}
export interface AttributionPairTs {
  sourceSpecId: string;
  baselineTrialIndex: number;
  perturbedTrialIndex: number;
  baselineStatus: TrialStatus;
  perturbedStatus: TrialStatus;
  baselineSourceValue: number;
  perturbedSourceValue: number;
}
export interface AttributionObservationTs {
  sourceSpecId: string;
  targetId: string;
  baselineTrialIndex: number;
  perturbedTrialIndex: number;
  baselineStatus: TrialStatus;
  perturbedStatus: TrialStatus;
  baselineSourceValue: number;
  perturbedSourceValue: number;
  baselineTargetValue: number | null;
  perturbedTargetValue: number | null;
  response: number | null;
  availability: Availability;
}
export interface AttributionAuthorityTs {
  authorityId: string;
  interpretation: typeof ATTRIBUTION_INTERPRETATION;
  sources: readonly AttributionSourceTs[];
  targets: readonly AttributionTargetTs[];
  pairs: readonly AttributionPairTs[];
  observations: readonly AttributionObservationTs[];
}
export interface AttributionViewDefinitionTs {
  schemaId: typeof ATTRIBUTION_VIEW_SCHEMA_ID;
  schemaVersion: 1;
  authorityId: string;
  sourceSpecId: string;
  targetId: string;
  baselineTrialIndex: number;
  perturbedTrialIndex: number;
}
export interface AttributionDenominatorTs {
  totalPairs: number;
  availablePairs: number;
  typedNoImpactPairs: number;
  unavailableNoImpactPairs: number;
  failedPairs: number;
  nonfinitePairs: number;
}
export interface AttributionViewTs {
  source: AttributionSourceTs;
  target: AttributionTargetTs;
  selected: AttributionObservationTs;
  observations: readonly AttributionObservationTs[];
  denominator: AttributionDenominatorTs;
}
const JOINT_BY_VARIABLE: Readonly<Record<string, AttributionSourceTs["jointId"]>> = {
  "swing_sim.swing.shoulder_commanded_torque_offset_nm": "joint.shoulder",
  "swing_sim.swing.wrist_commanded_torque_offset_nm": "joint.wrist",
};
const STATUSES = new Set<TrialStatus>([
  "evaluated_hit", "evaluated_no_impact", "numerical_failure",
]);
const AVAILABILITIES = new Set<Availability>([
  "available", "no_impact_unavailable", "numerical_failure", "nonfinite_unavailable",
]);
const sourceFromValue = (value: unknown): AttributionSourceTs => {
  const raw = record(value, [
    "spec_id", "variable_key", "joint_id", "time_window_s", "unit",
  ], "source");
  const variableKey = stable(raw.variable_key, "variable_key");
  const expectedJoint = JOINT_BY_VARIABLE[variableKey];
  if (!expectedJoint) throw new Error("unsupported localized variable");
  const jointId = stable(raw.joint_id, "joint_id");
  if (jointId !== expectedJoint) throw new Error("joint mismatch");
  if (!Array.isArray(raw.time_window_s) || raw.time_window_s.length !== 2) {
    throw new Error("window must contain start and end");
  }
  const start = finite(raw.time_window_s[0], "window start");
  const end = finite(raw.time_window_s[1], "window end");
  if (!(start >= 0 && start < end)) throw new Error("window must be half-open start < end");
  if (raw.unit !== "N·m") throw new Error("localized source unit must be N·m");
  return {
    specId: stable(raw.spec_id, "spec_id"), variableKey,
    jointId: expectedJoint, timeWindowS: [start, end], unit: "N·m",
  };
};
const targetFromValue = (value: unknown): AttributionTargetTs => {
  const raw = record(value, [
    "target_id", "kind", "name", "unit", "convention", "time_s", "point_id",
    "coordinate_frame",
  ], "target");
  if (raw.kind !== "state" && raw.kind !== "impact" && raw.kind !== "shot") {
    throw new Error("invalid target kind");
  }
  const kind = raw.kind;
  const name = stable(raw.name, "target name");
  const target: AttributionTargetTs = {
    targetId: stable(raw.target_id, "target_id"), kind, name,
    unit: stable(raw.unit, "target unit"),
    convention: stable(raw.convention, "target convention"),
    timeS: nullableFinite(raw.time_s, "target time"),
    pointId: raw.point_id === null ? null : stable(raw.point_id, "point_id"),
    coordinateFrame: raw.coordinate_frame === null
      ? null : stable(raw.coordinate_frame, "coordinate_frame"),
  };
  const definition = TARGET_REGISTRY[name];
  if (!definition || definition.kind !== target.kind || definition.unit !== target.unit ||
      definition.convention !== target.convention ||
      definition.coordinateFrame !== target.coordinateFrame) {
    throw new Error("target registry mismatch");
  }
  if (kind === "state") {
    if (target.timeS === null || target.timeS < 0 ||
        !target.pointId?.startsWith("swing.") ||
        target.coordinateFrame === null) {
      throw new Error("invalid state target locus");
    }
  } else {
    if (target.timeS !== null || target.pointId !== null ||
        target.coordinateFrame !== null) throw new Error(`invalid ${kind} target`);
  }
  return target;
};
const observationFromValue = (value: unknown): AttributionObservationTs => {
  const raw = record(value, [
    "source_spec_id", "target_id", "baseline_trial_index", "perturbed_trial_index",
    "baseline_status", "perturbed_status", "baseline_source_value",
    "perturbed_source_value", "baseline_target_value", "perturbed_target_value",
    "response", "availability",
  ], "observation");
  if (!STATUSES.has(raw.baseline_status as TrialStatus) ||
      !STATUSES.has(raw.perturbed_status as TrialStatus)) throw new Error("invalid status");
  if (!AVAILABILITIES.has(raw.availability as Availability)) {
    throw new Error("invalid availability");
  }
  const baselineTrialIndex = index(raw.baseline_trial_index, "baseline_trial_index");
  const perturbedTrialIndex = index(raw.perturbed_trial_index, "perturbed_trial_index");
  if (baselineTrialIndex === perturbedTrialIndex) throw new Error("pair trials must differ");
  const observation: AttributionObservationTs = {
    sourceSpecId: stable(raw.source_spec_id, "source_spec_id"),
    targetId: stable(raw.target_id, "target_id"), baselineTrialIndex, perturbedTrialIndex,
    baselineStatus: raw.baseline_status as TrialStatus,
    perturbedStatus: raw.perturbed_status as TrialStatus,
    baselineSourceValue: finite(raw.baseline_source_value, "baseline source value"),
    perturbedSourceValue: finite(raw.perturbed_source_value, "perturbed source value"),
    baselineTargetValue: nullableFinite(raw.baseline_target_value, "baseline target"),
    perturbedTargetValue: nullableFinite(raw.perturbed_target_value, "perturbed target"),
    response: nullableFinite(raw.response, "response"),
    availability: raw.availability as Availability,
  };
  if (observation.availability === "available") {
    if (observation.baselineTargetValue === null || observation.perturbedTargetValue === null ||
        observation.response === null || !responseMatches(observation.response,
          observation.perturbedTargetValue - observation.baselineTargetValue)) {
      throw new Error("response must equal perturbed minus baseline");
    }
  } else if (observation.response !== null ||
      (observation.baselineTargetValue !== null && observation.perturbedTargetValue !== null)) {
    throw new Error("unavailable pair must retain null target/response");
  }
  return observation;
};

const pairFromValue = (value: unknown): AttributionPairTs => {
  const raw = record(value, [
    "source_spec_id", "baseline_trial_index", "perturbed_trial_index",
    "baseline_status", "perturbed_status", "baseline_source_value",
    "perturbed_source_value",
  ], "pair roster");
  if (!STATUSES.has(raw.baseline_status as TrialStatus) ||
      !STATUSES.has(raw.perturbed_status as TrialStatus)) throw new Error("invalid status");
  const baselineTrialIndex = index(raw.baseline_trial_index, "baseline_trial_index");
  const perturbedTrialIndex = index(raw.perturbed_trial_index, "perturbed_trial_index");
  if (baselineTrialIndex === perturbedTrialIndex) throw new Error("pair trials must differ");
  return {
    sourceSpecId: stable(raw.source_spec_id, "source_spec_id"), baselineTrialIndex,
    perturbedTrialIndex, baselineStatus: raw.baseline_status as TrialStatus,
    perturbedStatus: raw.perturbed_status as TrialStatus,
    baselineSourceValue: finite(raw.baseline_source_value, "baseline source value"),
    perturbedSourceValue: finite(raw.perturbed_source_value, "perturbed source value"),
  };
};

const expectedAvailability = (
  observation: AttributionObservationTs, target: AttributionTargetTs,
): Availability => {
  const statuses = [observation.baselineStatus, observation.perturbedStatus];
  if (statuses.includes("numerical_failure")) return "numerical_failure";
  if (target.kind !== "state" && statuses.includes("evaluated_no_impact")) {
    return "no_impact_unavailable";
  }
  return observation.baselineTargetValue === null || observation.perturbedTargetValue === null
    ? "nonfinite_unavailable" : "available";
};

export function attributionAuthorityFromValue(value: unknown): AttributionAuthorityTs {
  const raw = record(value, [
    "schema_id", "schema_version", "authority_id", "interpretation",
    "sources", "targets", "pairs", "observations",
  ], "authority");
  if (raw.schema_id !== ATTRIBUTION_AUTHORITY_SCHEMA_ID ||
      raw.schema_version !== ATTRIBUTION_SCHEMA_VERSION ||
      raw.interpretation !== ATTRIBUTION_INTERPRETATION) {
    throw new Error("invalid attribution authority schema");
  }
  const sourceRows = boundedArray(raw.sources, MAX_SOURCES, "sources");
  const targetRows = boundedArray(raw.targets, MAX_TARGETS, "targets");
  const pairRows = boundedArray(raw.pairs, MAX_PAIRS, "pairs");
  const observationRows = boundedArray(raw.observations, MAX_OBSERVATIONS, "observations");
  requireAuthorityShape(
    sourceRows.length, targetRows.length, pairRows.length, observationRows.length,
  );
  const sources = sourceRows.map(sourceFromValue);
  const targets = targetRows.map(targetFromValue);
  const pairs = pairRows.map(pairFromValue);
  const observations = observationRows.map(observationFromValue);
  const sourceIds = new Set(sources.map((source) => source.specId));
  const targetMap = new Map(targets.map((target) => [target.targetId, target]));
  if (sourceIds.size !== sources.length || sourceIds.size === 0 ||
      targetMap.size !== targets.length || targetMap.size === 0) throw new Error("duplicate IDs");
  if (observations.length === 0) throw new Error("observations must be nonempty");
  const pairMap = new Map(pairs.map((pair) => [[pair.sourceSpecId,
    pair.baselineTrialIndex, pair.perturbedTrialIndex].join("|"), pair]));
  if (pairMap.size !== pairs.length ||
      new Set(pairs.map((pair) => pair.sourceSpecId)).size !== sourceIds.size) {
    throw new Error("pair matrix must cover every source with unique pairs");
  }
  const keys = new Set<string>();
  observations.forEach((observation) => {
    const target = targetMap.get(observation.targetId);
    if (!sourceIds.has(observation.sourceSpecId) || !target) throw new Error("unknown reference");
    if (observation.availability !== expectedAvailability(observation, target)) {
      throw new Error("availability does not match typed outcomes");
    }
    const pair = pairMap.get([observation.sourceSpecId,
      observation.baselineTrialIndex, observation.perturbedTrialIndex].join("|"));
    if (!pair || pair.baselineStatus !== observation.baselineStatus ||
        pair.perturbedStatus !== observation.perturbedStatus ||
        pair.baselineSourceValue !== observation.baselineSourceValue ||
        pair.perturbedSourceValue !== observation.perturbedSourceValue) {
      throw new Error("observation conflicts with pair roster");
    }
    const key = [observation.sourceSpecId, observation.targetId,
      observation.baselineTrialIndex, observation.perturbedTrialIndex].join("|");
    if (keys.has(key)) throw new Error("duplicate attribution observation");
    keys.add(key);
  });
  const expected = new Set(pairs.flatMap((pair) => targets.map((target) => [
    pair.sourceSpecId, target.targetId, pair.baselineTrialIndex,
    pair.perturbedTrialIndex,
  ].join("|"))));
  if (keys.size !== expected.size || [...expected].some((key) => !keys.has(key))) {
    throw new Error("observation matrix must be complete");
  }
  return deepFreeze({
    authorityId: stable(raw.authority_id, "authority_id"),
    interpretation: ATTRIBUTION_INTERPRETATION, sources, targets, pairs, observations,
  });
}

export function attributionAuthorityToValue(authority: AttributionAuthorityTs): unknown {
  const value = {
    schema_id: ATTRIBUTION_AUTHORITY_SCHEMA_ID, schema_version: ATTRIBUTION_SCHEMA_VERSION,
    authority_id: authority.authorityId, interpretation: authority.interpretation,
    sources: authority.sources.map((source) => ({
      spec_id: source.specId, variable_key: source.variableKey, joint_id: source.jointId,
      time_window_s: [...source.timeWindowS], unit: source.unit,
    })),
    targets: authority.targets.map((target) => ({
      target_id: target.targetId, kind: target.kind, name: target.name, unit: target.unit,
      convention: target.convention, time_s: target.timeS, point_id: target.pointId,
      coordinate_frame: target.coordinateFrame,
    })),
    pairs: authority.pairs.map((pair) => ({
      source_spec_id: pair.sourceSpecId,
      baseline_trial_index: pair.baselineTrialIndex,
      perturbed_trial_index: pair.perturbedTrialIndex,
      baseline_status: pair.baselineStatus, perturbed_status: pair.perturbedStatus,
      baseline_source_value: pair.baselineSourceValue,
      perturbed_source_value: pair.perturbedSourceValue,
    })),
    observations: authority.observations.map((row) => ({
      source_spec_id: row.sourceSpecId, target_id: row.targetId,
      baseline_trial_index: row.baselineTrialIndex,
      perturbed_trial_index: row.perturbedTrialIndex,
      baseline_status: row.baselineStatus, perturbed_status: row.perturbedStatus,
      baseline_source_value: row.baselineSourceValue,
      perturbed_source_value: row.perturbedSourceValue,
      baseline_target_value: row.baselineTargetValue,
      perturbed_target_value: row.perturbedTargetValue,
      response: row.response, availability: row.availability,
    })),
  };
  attributionAuthorityFromValue(value);
  return value;
}

export function buildAttributionView(
  authority: AttributionAuthorityTs,
  definition: AttributionViewDefinitionTs,
): AttributionViewTs {
  if (authority.authorityId !== definition.authorityId) throw new Error("authority mismatch");
  const source = authority.sources.find((item) => item.specId === definition.sourceSpecId);
  const target = authority.targets.find((item) => item.targetId === definition.targetId);
  if (!source || !target) throw new Error("unknown attribution selection");
  const observations = authority.observations.filter((item) =>
    item.sourceSpecId === source.specId && item.targetId === target.targetId);
  const selected = observations.find((item) =>
    item.baselineTrialIndex === definition.baselineTrialIndex &&
    item.perturbedTrialIndex === definition.perturbedTrialIndex);
  if (!selected) throw new Error("selected attribution pair is unavailable");
  const typedNoImpactPairs = observations.filter((row) =>
    row.baselineStatus === "evaluated_no_impact" ||
    row.perturbedStatus === "evaluated_no_impact").length;
  const denominator: AttributionDenominatorTs = {
    totalPairs: observations.length,
    availablePairs: observations.filter((row) => row.availability === "available").length,
    typedNoImpactPairs,
    unavailableNoImpactPairs: observations.filter((row) =>
      row.availability === "no_impact_unavailable").length,
    failedPairs: observations.filter((row) => row.availability === "numerical_failure").length,
    nonfinitePairs: observations.filter((row) =>
      row.availability === "nonfinite_unavailable").length,
  };
  return { source, target, selected, observations, denominator };
}

const viewValue = (definition: AttributionViewDefinitionTs): RecordValue => ({
  schema_id: definition.schemaId, schema_version: definition.schemaVersion,
  authority_id: definition.authorityId, source_spec_id: definition.sourceSpecId,
  target_id: definition.targetId, baseline_trial_index: definition.baselineTrialIndex,
  perturbed_trial_index: definition.perturbedTrialIndex,
});

export const attributionViewToJson = (definition: AttributionViewDefinitionTs): string => {
  const encoded = JSON.stringify(viewValue(definition));
  attributionViewFromJson(encoded);
  return encoded;
};

export function attributionViewFromJson(text: string): AttributionViewDefinitionTs {
  const raw = record(JSON.parse(text), [
    "schema_id", "schema_version", "authority_id", "source_spec_id", "target_id",
    "baseline_trial_index", "perturbed_trial_index",
  ], "view definition");
  if (raw.schema_id !== ATTRIBUTION_VIEW_SCHEMA_ID ||
      raw.schema_version !== ATTRIBUTION_SCHEMA_VERSION) throw new Error("invalid view schema");
  const baselineTrialIndex = index(raw.baseline_trial_index, "baseline_trial_index");
  const perturbedTrialIndex = index(raw.perturbed_trial_index, "perturbed_trial_index");
  if (baselineTrialIndex === perturbedTrialIndex) throw new Error("pair trials must differ");
  return deepFreeze({
    schemaId: ATTRIBUTION_VIEW_SCHEMA_ID, schemaVersion: 1,
    authorityId: stable(raw.authority_id, "authority_id"),
    sourceSpecId: stable(raw.source_spec_id, "source_spec_id"),
    targetId: stable(raw.target_id, "target_id"), baselineTrialIndex, perturbedTrialIndex,
  });
}

export {
  attributionObservationsToCsv,
  attributionObservationsToRows,
} from "./localizedAttributionCsv";
