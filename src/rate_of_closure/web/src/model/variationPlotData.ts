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
  launch_azimuth_deg: "Launch Direction",
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
  let low = Infinity, high = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v < low) low = v;
    if (v > high) high = v;
  }
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
  // ⚡ Bolt Optimization: Replacing intermediate array allocations and chained map/join calls
  // with a single-pass for loop using string concatenation to reduce garbage collection pressure.
  let csv = csvCell("trial_index") + "," + csvCell("outcome");
  for (let i = 0; i < variableKeys.length; i++) {
    csv += "," + csvCell(variableKeys[i]);
  }

  const successArray = dataset.success;
  for (let i = 0; i < successArray.length; i++) {
    csv += "\n" + csvCell(String(i)) + "," + csvCell(outcomes?.[i] ?? (successArray[i] ? "evaluated" : "failure"));
    for (let j = 0; j < columns.length; j++) {
      const value = columns[j][i];
      csv += "," + csvCell(value === null || !Number.isFinite(value) ? "" : String(value));
    }
  }

  return csv;
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
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        for (let i = 0; i < points.length; i++) {
          const p = points[i];
          if (p.x < minX) minX = p.x;
          if (p.x > maxX) maxX = p.x;
          if (p.y < minY) minY = p.y;
          if (p.y > maxY) maxY = p.y;
        }
        const spanX = Math.max(maxX - minX, 1e-12);
        const spanY = Math.max(maxY - minY, 1e-12);
        const scaleX = (value: number) => pad + (value - minX) / spanX * (size - 2 * pad);
        const scaleY = (value: number) => pad + (value - minY) / spanY * (size - 2 * pad);
        marks = points.map((point) =>
          `<circle cx="${scaleX(point.x)}" cy="${size - scaleY(point.y)}" r="2.3" fill="${matrixCohortColor(outcomes?.[point.trialIndex] ?? point.cohort)}" opacity="0.65"><title>Trial ${point.trialIndex + 1}</title></circle>`,
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
