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

export interface ScalarMarginalDataTs {
  variable: ScalarPlotVariableTs;
  binEdges: number[];
  counts: number[];
  nAvailable: number;
  nMissing: number;
}

export const OUTPUT_UNITS: Record<string, string> = {
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
  loss: "1",
  constraint_violated: "0/1",
  leading_edge_clearance_at_ball_m: "m",
  minimum_pre_ball_clearance_m: "m",
  ground_after_ball_margin_s: "s",
  low_point_clearance_m: "m",
  delivered_bounce_deg: "deg",
  path_projected_effective_bounce_deg: "deg",
  reference_aoa_deg: "deg",
  bounce_utilization_margin_deg: "deg",
  peak_turf_penetration_m: "m",
  normal_turf_impulse_n_s: "N s",
  shaft_rotation_rate_rad_s: "rad/s",
  shaft_counterfactual_aoa_delta_deg: "deg",
  shaft_shapley_aoa_deg: "deg",
  shaft_aoa_contribution_deg: "deg",
  shaft_vertical_velocity_share: "1",
  leading_edge_3d_rate_rad_s: "rad/s",
  face_normal_3d_rate_rad_s: "rad/s",
  leading_edge_relative_arc_heading_rate_rad_s: "rad/s",
};

export const OUTPUT_LABELS: Record<string, string> = {
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
  loss: "Decision Loss",
  constraint_violated: "Constraint Violation",
  leading_edge_clearance_at_ball_m: "Leading-Edge Clearance at Ball",
  minimum_pre_ball_clearance_m: "Minimum Pre-Ball Clearance",
  ground_after_ball_margin_s: "Ground-After-Ball Time Margin",
  low_point_clearance_m: "Low-Point Clearance",
  delivered_bounce_deg: "Delivered Bounce",
  path_projected_effective_bounce_deg: "Path-Projected Effective Bounce",
  reference_aoa_deg: "Reference-Point Angle of Attack",
  bounce_utilization_margin_deg: "Bounce-Utilization Margin",
  peak_turf_penetration_m: "Peak Turf Penetration",
  normal_turf_impulse_n_s: "Normal Turf Impulse",
  shaft_rotation_rate_rad_s: "Shaft Rotation Rate",
  shaft_counterfactual_aoa_delta_deg: "Shaft Counterfactual AoA Change",
  shaft_shapley_aoa_deg: "Shaft AoA Shapley Contribution",
  shaft_aoa_contribution_deg: "Shaft AoA Counterfactual Contribution",
  shaft_vertical_velocity_share: "Shaft Vertical-Velocity Share",
  leading_edge_3d_rate_rad_s: "Leading-Edge 3D Rate",
  face_normal_3d_rate_rad_s: "Face-Normal 3D Rate",
  leading_edge_relative_arc_heading_rate_rad_s: "Leading-Edge Arc-Relative Heading Rate",
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

export const scalarValues = (
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

export function buildScalarMarginal(
  dataset: VariationDatasetTs,
  key: string,
  binCount = 12,
): ScalarMarginalDataTs {
  if (!Number.isInteger(binCount) || binCount < 2 || binCount > 100) {
    throw new Error("binCount must be an integer in [2, 100]");
  }
  const variable = buildScalarPlotVariables(dataset).find((item) => item.key === key);
  if (!variable) throw new Error(`unknown marginal variable ${key}`);
  const values = scalarValues(dataset, variable).filter(
    (value): value is number => value !== null && Number.isFinite(value),
  );
  if (values.length === 0) {
    return { variable, binEdges: [], counts: [], nAvailable: 0, nMissing: dataset.success.length };
  }
  const low = Math.min(...values);
  const high = Math.max(...values);
  const span = high - low || Math.max(Math.abs(low), 1) * 1e-9;
  const binEdges = Array.from({ length: binCount + 1 }, (_, index) => low + span * index / binCount);
  const counts = Array(binCount).fill(0) as number[];
  values.forEach((value) => {
    const index = Math.min(Math.floor((value - low) / span * binCount), binCount - 1);
    counts[index] += 1;
  });
  return {
    variable, binEdges, counts, nAvailable: values.length,
    nMissing: dataset.success.length - values.length,
  };
}

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

export function distributionMatrixToCsv(
  dataset: VariationDatasetTs,
  variableKeys: string[],
  outcomes?: string[],
): string {
  const selected = selectedMatrixVariables(dataset, variableKeys);
  const columns = selected.map((variable) => scalarValues(dataset, variable));
  if (outcomes !== undefined && outcomes.length !== dataset.success.length) {
    throw new Error("matrix outcomes must align with dataset trials");
  }
  const rows = dataset.success.map((success, trialIndex) => [
    String(trialIndex),
    outcomes?.[trialIndex] ?? (success ? "evaluated" : "failure"),
    ...columns.map((column) => {
      const value = column[trialIndex];
      return value === null || !Number.isFinite(value) ? "" : String(value);
    }),
  ]);
  return [
    ["trial_index", "outcome", ...variableKeys],
    ...rows,
  ].map((row) => row.map(csvCell).join(",")).join("\n");
}

export function distributionMatrixToSvg(
  dataset: VariationDatasetTs,
  variableKeys: string[],
  outcomes?: string[],
): string {
  const selected = selectedMatrixVariables(dataset, variableKeys);
  const size = 150;
  const pad = 14;
  const cells = selected.flatMap((row, rowIndex) =>
    selected.map((column, columnIndex) => {
      const originX = columnIndex * size;
      const originY = rowIndex * size;
      const title = xmlEscape(rowIndex === columnIndex
        ? `${row.label} marginal histogram`
        : `${column.label} versus ${row.label}`);
      let marks: string;
      if (rowIndex === columnIndex) {
        const marginal = buildScalarMarginal(dataset, row.key);
        const maximum = Math.max(...marginal.counts, 1);
        const width = (size - 2 * pad) / Math.max(marginal.counts.length, 1);
        marks = marginal.counts.map((count, index) => {
          const height = count / maximum * (size - 2 * pad);
          return `<rect x="${pad + index * width}" y="${size - pad - height}" width="${Math.max(width - 1, 1)}" height="${height}" fill="#38bdf8" opacity="0.75"/>`;
        }).join("");
      } else {
        const points = buildScalarScatter(dataset, column.key, row.key).points;
        const xs = points.map((point) => point.x);
        const ys = points.map((point) => point.y);
        marks = points.map((point) =>
          `<circle cx="${matrixScale(point.x, xs, pad, size)}" cy="${size - matrixScale(point.y, ys, pad, size)}" r="2.3" fill="${matrixCohortColor(outcomes?.[point.trialIndex] ?? point.cohort)}" opacity="0.65"><title>Trial ${point.trialIndex + 1}</title></circle>`,
        ).join("");
      }
      const label = rowIndex === columnIndex
        ? `<text x="4" y="11" fill="#cbd5e1" font-size="8">${xmlEscape(row.label)} [${xmlEscape(row.unit || "unitless")}]</text>`
        : "";
      return `<g transform="translate(${originX} ${originY})"><title>${title}</title><rect width="${size}" height="${size}" fill="#020617" stroke="#1e293b"/>${marks}${label}</g>`;
    }),
  ).join("");
  const extent = selected.length * size;
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${extent} ${extent}" role="img"><title>Scatter matrix with marginal histograms</title>${cells}</svg>`;
}

const selectedMatrixVariables = (
  dataset: VariationDatasetTs,
  variableKeys: string[],
): ScalarPlotVariableTs[] => {
  if (new Set(variableKeys).size !== variableKeys.length) {
    throw new Error("matrix variable keys must be unique");
  }
  const variables = buildScalarPlotVariables(dataset);
  return variableKeys.map((key) => {
    const variable = variables.find((item) => item.key === key);
    if (!variable) throw new Error(`unknown matrix variable ${key}`);
    return variable;
  });
};

const matrixScale = (
  value: number,
  values: number[],
  pad: number,
  size: number,
): number => {
  const low = Math.min(...values);
  const high = Math.max(...values);
  return pad + (value - low) / Math.max(high - low, 1e-12) * (size - 2 * pad);
};

const xmlEscape = (value: string): string => value
  .split("&").join("&amp;")
  .split("<").join("&lt;")
  .split(">").join("&gt;");

export const matrixCohortColor = (cohort: string): string => ({
  evaluated_hit: "#38bdf8",
  evaluated_no_impact: "#f59e0b",
  numerical_failure: "#ef6464",
  evaluated: "#38bdf8",
  failure: "#ef6464",
}[cohort] ?? "#38bdf8");

const csvCell = (value: string): string =>
  /[",\r\n]/.test(value) ? `"${value.split('"').join('""')}"` : value;

const availability = (total: number, plotted: number): CohortAvailabilityTs => ({
  total,
  plotted,
  unavailable: total - plotted,
});
