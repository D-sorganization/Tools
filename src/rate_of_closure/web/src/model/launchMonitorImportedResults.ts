import { finiteLaunchMonitorScalar, type LaunchMonitorRow } from "./launchMonitorAnalysisTypes";

export interface RankedValue {
  label: string;
  value: number;
  rank: number;
  method: string;
}

export interface PcaScore {
  id: string;
  pc1: number;
  pc2: number;
}

export interface ImportedAdvancedResults {
  pcaLoadings: RankedValue[];
  pcaScores: PcaScore[];
  featureImportance: RankedValue[];
  performance: Array<{ metric: string; value: number; method: string }>;
  residualColumns: string[];
}

const firstColumn = (columns: string[], patterns: RegExp[]) =>
  columns.find((column) => patterns.some((pattern) => pattern.test(column))) ?? null;

const text = (row: LaunchMonitorRow, column: string | null, fallback: string) =>
  column && row[column] !== null && row[column] !== undefined ? String(row[column]) : fallback;

export function importedAdvancedResults(rows: LaunchMonitorRow[]): ImportedAdvancedResults {
  const columns = Object.keys(rows[0] ?? {});
  const featureColumn = firstColumn(columns, [/^feature$/i, /^variable$/i, /^predictor$/i, /^metric$/i]);
  const loadingColumn = firstColumn(columns, [/^loading$/i, /pca.*loading/i, /component.*loading/i]);
  const importanceColumn = firstColumn(columns, [/^importance$/i, /permutation.*importance/i, /feature.*importance/i]);
  const rankColumn = firstColumn(columns, [/^rank$/i, /importance.*rank/i]);
  const methodColumn = firstColumn(columns, [/^method$/i, /^model$/i, /estimator/i, /analysis.*method/i]);
  const componentColumn = firstColumn(columns, [/^component$/i, /^principal_component$/i, /^pc$/i]);
  const pc1Column = firstColumn(columns, [/^pc[_ -]?1$/i, /component[_ -]?1.*score/i]);
  const pc2Column = firstColumn(columns, [/^pc[_ -]?2$/i, /component[_ -]?2.*score/i]);
  const idColumn = firstColumn(columns, [/^shot_id$/i, /^row_id$/i, /^sample_id$/i]);

  const ranked = (valueColumn: string | null, kind: "loading" | "importance") => {
    if (!valueColumn || !featureColumn) return [];
    return rows.flatMap((row, index) => {
      const value = finiteLaunchMonitorScalar(row[valueColumn]);
      if (value === null) return [];
      const component = kind === "loading" ? text(row, componentColumn, "PCA") : "";
      const label = `${text(row, featureColumn, `feature ${index + 1}`)}${component ? ` · ${component}` : ""}`;
      const suppliedRank = rankColumn ? finiteLaunchMonitorScalar(row[rankColumn]) : null;
      return [{ label, value, rank: suppliedRank ?? 0, method: text(row, methodColumn, kind) }];
    }).sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).map(
      (item, index) => ({ ...item, rank: item.rank || index + 1 }),
    );
  };

  const pcaScores = pc1Column && pc2Column ? rows.flatMap((row, index) => {
    const pc1 = finiteLaunchMonitorScalar(row[pc1Column]);
    const pc2 = finiteLaunchMonitorScalar(row[pc2Column]);
    return pc1 === null || pc2 === null ? [] : [{ id: text(row, idColumn, String(index + 1)), pc1, pc2 }];
  }) : [];

  const performance = columns.filter((column) =>
    /(held.?out|test|validation).*(r2|rmse|mae)|^(r2|rmse|mae).*(held.?out|test|validation)/i.test(column))
    .flatMap((column) => rows.flatMap((row) => {
      const value = finiteLaunchMonitorScalar(row[column]);
      return value === null ? [] : [{ metric: column, value, method: text(row, methodColumn, "unspecified") }];
    })).filter((item, index, items) => items.findIndex(
      (candidate) => candidate.metric === item.metric && candidate.method === item.method && candidate.value === item.value,
    ) === index);

  return {
    pcaLoadings: ranked(loadingColumn, "loading"),
    pcaScores,
    featureImportance: ranked(importanceColumn, "importance"),
    performance,
    residualColumns: columns.filter((column) => /(residual|model.*spread|spread.*model)/i.test(column)),
  };
}

