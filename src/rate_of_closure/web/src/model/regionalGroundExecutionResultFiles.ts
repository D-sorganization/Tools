/** Canonical browser download boundary for complete job-bound study results. */

import {
  stableRegionalGroundExecutionResultJson,
  type RegionalGroundExecutionResult,
} from "./regionalGroundExecutionResult";
import { scalarEnsembleToCsv } from "./scalarEnsembleCsv";

const safeFileStem = (jobId: string): string =>
  jobId.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "") || "regional-ground-study";

export const downloadRegionalGroundExecutionResult = (
  result: RegionalGroundExecutionResult,
): void => {
  const source = stableRegionalGroundExecutionResultJson(result);
  const url = URL.createObjectURL(
    new Blob([source], { type: "application/json;charset=utf-8" }),
  );
  try {
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${safeFileStem(result.job_id)}.regional-ground-result.json`;
    anchor.click();
  } finally {
    URL.revokeObjectURL(url);
  }
};

/** Download every validated scalar row without inferring missing values. */
export const downloadRegionalGroundExecutionRowsCsv = (
  result: RegionalGroundExecutionResult,
): void => {
  const source = scalarEnsembleToCsv(result.dataset);
  const url = URL.createObjectURL(new Blob([source], { type: "text/csv;charset=utf-8" }));
  try {
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${safeFileStem(result.job_id)}.regional-ground-rows.csv`;
    anchor.click();
  } finally {
    URL.revokeObjectURL(url);
  }
};
