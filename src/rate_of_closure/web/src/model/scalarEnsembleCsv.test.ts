import { describe, expect, it } from "vitest";

import { createScalarEnsemble, scalarEnsembleRowId } from "./scalarEnsembleContract";
import { scalarEnsembleToCsv } from "./scalarEnsembleCsv";

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
