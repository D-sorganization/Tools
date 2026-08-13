import { describe, expect, it } from "vitest";

import { createScalarEnsemble, scalarEnsembleRowId } from "./scalarEnsembleContract";
import { nonCompleteReasonSummary, scalarEnsembleToCsv } from "./scalarEnsembleCsv";

const reasonEnsemble = (
  rows: ReadonlyArray<{ cohort: "complete" | "failed"; reason_code: string | null }>,
) => createScalarEnsemble({
  result_id: "reason-test",
  provenance: {
    adapter_id: "reason-test/v1", source_schema_version: "source/v1",
    source_provenance: "fixture",
  },
  stages: [{ key: "input", label: "Input" }],
  categories: [{ key: "launch", label: "Launch" }],
  variables: [
    { key: "speed", label: "Speed", unit: "m/s", stage_key: "input", category_key: "launch" },
  ],
  cohorts: [
    { key: "complete" as const, label: "Complete" },
    { key: "failed" as const, label: "Failed" },
  ],
  rows: rows.map((row, index) => ({
    row_id: scalarEnsembleRowId(index, "run"), trial_index: index,
    series_id: "run", cohort: row.cohort,
    values: { speed: 1 },
    attributes: { reason_code: row.reason_code },
  })),
});

describe("scalarEnsembleToCsv", () => {
  it("retains every row, declared scalar, cohort, series, trial, and attribute", () => {
    const result = createScalarEnsemble({
      result_id: "csv-test",
      provenance: {
        adapter_id: "csv-test/v1", source_schema_version: "source/v1",
        source_provenance: "fixture",
      },
      stages: [{ key: "input", label: "Input" }],
      categories: [{ key: "wind", label: "Wind" }],
      variables: [
        { key: "speed", label: "Speed", unit: "m/s", stage_key: "input", category_key: "wind" },
        { key: "miss", label: "Miss", unit: "m", stage_key: "input", category_key: "wind" },
      ],
      cohorts: [
        { key: "completed" as const, label: "Completed" },
        { key: "invalid" as const, label: "Invalid" },
      ],
      rows: [
        {
          row_id: scalarEnsembleRowId(0, "stock"), trial_index: 0,
          series_id: "stock", cohort: "completed" as const,
          values: { speed: 4, miss: 1.25 },
          attributes: { reason: null, label: "Stock", formula: "=SUM(1,2)" },
        },
        {
          row_id: scalarEnsembleRowId(1, "stock"), trial_index: 1,
          series_id: "stock", cohort: "invalid" as const,
          values: { speed: 5, miss: null },
          attributes: { reason: "bad, value", label: "Stock", formula: null },
        },
      ],
    });

    expect(scalarEnsembleToCsv(result)).toBe([
      "row_id,trial_index,series_id,cohort,speed,miss,attribute:formula,attribute:label,attribute:reason",
      'series:stock/trial:0,0,stock,completed,4,1.25,"\'=SUM(1,2)",Stock,',
      'series:stock/trial:1,1,stock,invalid,5,,,Stock,"bad, value"',
    ].join("\n"));
  });
});

describe("nonCompleteReasonSummary", () => {
  it("names the retained reason so a horizon timeout is not read as breakage", () => {
    const summary = nonCompleteReasonSummary(reasonEnsemble([
      { cohort: "complete", reason_code: null },
      { cohort: "failed", reason_code: "no_ground_crossing_before_max_time" },
      { cohort: "failed", reason_code: "no_ground_crossing_before_max_time" },
    ]));

    expect(summary).toBe(
      " Non-complete reasons: no_ground_crossing_before_max_time x2.");
  });

  it("orders by descending count then reason and labels missing reasons", () => {
    const summary = nonCompleteReasonSummary(reasonEnsemble([
      { cohort: "failed", reason_code: null },
      { cohort: "failed", reason_code: "overflow" },
      { cohort: "failed", reason_code: "overflow" },
    ]));

    expect(summary).toBe(" Non-complete reasons: overflow x2; unspecified x1.");
  });

  it("is empty when every retained row completed", () => {
    expect(nonCompleteReasonSummary(reasonEnsemble([
      { cohort: "complete", reason_code: null },
    ]))).toBe("");
  });
});
