/** Plot-ready scalar variation data with units and explicit availability. */

import { variableDef, type VariationDatasetTs } from "./variation";

export type ScalarVariableKindTs = "input" | "contact" | "impact" | "shot";
export type ScalarCohortTs = "evaluated" | "failure";

export interface ScalarPlotVariableTs {
  key: string;
  label: string;
  unit: string;
  kind: ScalarVariableKindTs;
}

export interface ScalarScatterPointTs {
  trialIndex: number;
  x: number;
  y: number;
  cohort: ScalarCohortTs;
}

export interface CohortAvailabilityTs {
  total: number;
  plotted: number;
  unavailable: number;
}

export interface ScalarScatterDataTs {
  xVariable: ScalarPlotVariableTs;
  yVariable: ScalarPlotVariableTs;
  points: ScalarScatterPointTs[];
  cohorts: Record<ScalarCohortTs, CohortAvailabilityTs>;
}

const OUTPUT_UNITS: Record<string, string> = {
  candidate_time_s: "s",
  closest_approach_m: "m",
  contact_margin_m: "m",
  impact_time_s: "s",
  clubhead_speed_mps: "m/s",
  spin_loft_deg: "deg",
  face_to_path_deg: "deg",
  spin_axis_tilt_deg: "deg",
  club_path_deg: "deg",
  face_angle_deg: "deg",
  attack_angle_deg: "deg",
  dynamic_loft_deg: "deg",
  ball_speed_mph: "mph",
  launch_angle_deg: "deg",
  launch_azimuth_deg: "deg",
  spin_rpm: "rpm",
  spin_axis_deg: "deg",
  carry_m: "m",
  lateral_m: "m",
  apex_m: "m",
  max_height_m: "m",
  landing_angle_deg: "deg",
  flight_time_s: "s",
};

const OUTPUT_LABELS: Record<string, string> = {
  candidate_time_s: "Candidate Contact Time",
  closest_approach_m: "Closest Approach",
  contact_margin_m: "Contact Margin",
  impact_time_s: "Impact Time",
  clubhead_speed_mps: "Clubhead Speed",
  spin_loft_deg: "Spin Loft",
  face_to_path_deg: "Face to Path",
  spin_axis_tilt_deg: "Spin-Axis Tilt",
  club_path_deg: "Club Path",
  face_angle_deg: "Face Angle",
  attack_angle_deg: "Attack Angle",
  dynamic_loft_deg: "Dynamic Loft",
  ball_speed_mph: "Ball Speed",
  launch_angle_deg: "Launch Angle",
  launch_azimuth_deg: "Launch Azimuth",
  spin_rpm: "Spin Rate",
  spin_axis_deg: "Spin-Axis Tilt",
  carry_m: "Carry",
  lateral_m: "Lateral Landing Position",
  apex_m: "Apex Height",
  max_height_m: "Maximum Height",
  landing_angle_deg: "Landing Angle",
  flight_time_s: "Flight Time",
};

const CONTACT_OUTPUTS = new Set([
  "candidate_time_s",
  "closest_approach_m",
  "contact_margin_m",
]);

const IMPACT_OUTPUTS = new Set([
  "impact_time_s",
  "clubhead_speed_mps",
  "spin_loft_deg",
  "face_to_path_deg",
  "spin_axis_tilt_deg",
  "club_path_deg",
  "face_angle_deg",
  "attack_angle_deg",
  "dynamic_loft_deg",
]);

export function buildScalarPlotVariables(
  dataset: VariationDatasetTs,
): ScalarPlotVariableTs[] {
  const inputs = dataset.inputNames.map((name) => {
    const definition = variableDef(name);
    if (!definition) throw new Error(`unknown variation input ${name}`);
    return {
      key: `input:${name}`,
      label: definition.label,
      unit: definition.unit,
      kind: "input" as const,
    };
  });
  const outputs = dataset.outputNames.map((name) => {
    const unit = OUTPUT_UNITS[name];
    if (unit === undefined) throw new Error(`unknown variation output ${name}`);
    return {
      key: `output:${name}`,
      label: OUTPUT_LABELS[name] ?? name,
      unit,
      kind: CONTACT_OUTPUTS.has(name)
        ? "contact" as const
        : IMPACT_OUTPUTS.has(name)
          ? "impact" as const
          : "shot" as const,
    };
  });
  return [...inputs, ...outputs];
}

const scalarValues = (
  dataset: VariationDatasetTs,
  variable: ScalarPlotVariableTs,
): Array<number | null> => {
  const [source, name] = variable.key.split(":", 2);
  if (source === "input") {
    const column = dataset.inputNames.indexOf(name);
    if (column < 0) throw new Error(`unknown input axis ${name}`);
    return dataset.inputs.map((row) => row[column]);
  }
  const column = dataset.outputNames.indexOf(name);
  if (source !== "output" || column < 0) throw new Error(`unknown output axis ${name}`);
  return dataset.outputs.map((row) => row[column]);
};

export function buildScalarScatter(
  dataset: VariationDatasetTs,
  xKey: string,
  yKey: string,
): ScalarScatterDataTs {
  const variables = buildScalarPlotVariables(dataset);
  const xVariable = variables.find((item) => item.key === xKey);
  const yVariable = variables.find((item) => item.key === yKey);
  if (!xVariable || !yVariable) throw new Error("scatter axes must be known variables");
  const xValues = scalarValues(dataset, xVariable);
  const yValues = scalarValues(dataset, yVariable);
  const points: ScalarScatterPointTs[] = [];
  const totals: Record<ScalarCohortTs, number> = { evaluated: 0, failure: 0 };
  const plotted: Record<ScalarCohortTs, number> = { evaluated: 0, failure: 0 };
  dataset.success.forEach((success, trialIndex) => {
    const cohort: ScalarCohortTs = success ? "evaluated" : "failure";
    totals[cohort] += 1;
    const x = xValues[trialIndex];
    const y = yValues[trialIndex];
    if (x !== null && y !== null && Number.isFinite(x) && Number.isFinite(y)) {
      points.push({ trialIndex, x, y, cohort });
      plotted[cohort] += 1;
    }
  });
  return {
    xVariable,
    yVariable,
    points,
    cohorts: {
      evaluated: availability(totals.evaluated, plotted.evaluated),
      failure: availability(totals.failure, plotted.failure),
    },
  };
}

const availability = (total: number, plotted: number): CohortAvailabilityTs => ({
  total,
  plotted,
  unavailable: total - plotted,
});
