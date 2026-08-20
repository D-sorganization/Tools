import { describe, expect, it } from "vitest";

import {
  buildScalarEnsembleScatter,
  createScalarEnsemble,
  scalarEnsembleRowId,
  type ScalarEnsembleInput,
} from "./scalarEnsembleContract";

type Cohort = "complete" | "failed";

const input = (): ScalarEnsembleInput<Cohort> => ({
  result_id: "example-ensemble",
  provenance: {
    adapter_id: "test-adapter/v1",
    source_schema_version: "source/v2",
    source_provenance: "seed=42",
  },
  stages: [
    { key: "input", label: "Inputs" },
    { key: "result", label: "Results" },
  ],
  categories: [
    { key: "delivery", label: "Delivery" },
    { key: "shot", label: "Shot" },
  ],
  variables: [
    { key: "speed", label: "Speed", unit: "m/s", stage_key: "input", category_key: "delivery" },
    { key: "carry", label: "Carry", unit: "m", stage_key: "result", category_key: "shot" },
  ],
  cohorts: [
    { key: "complete", label: "Completed" },
    { key: "failed", label: "Failed" },
  ],
  rows: [
    {
      row_id: scalarEnsembleRowId(0, "baseline"), trial_index: 0,
      series_id: "baseline", cohort: "complete", values: { speed: 30, carry: 27.4 },
      attributes: { status: "complete" },
    },
    {
      row_id: scalarEnsembleRowId(1, "baseline"), trial_index: 1,
      series_id: "baseline", cohort: "failed", values: { speed: 31, carry: null },
      attributes: { status: "failed" },
    },
    {
      row_id: scalarEnsembleRowId(2, "baseline"), trial_index: 2,
      series_id: "baseline", cohort: "complete", values: { speed: null, carry: 28.1 },
    },
    {
      row_id: scalarEnsembleRowId(0, "alternative"), trial_index: 0,
      series_id: "alternative", cohort: "complete", values: { speed: 29, carry: 26.8 },
    },
  ],
});

describe("scalar ensemble contract", () => {
  it("creates a stable unit-bearing result with caller-defined cohorts", () => {
    const result = createScalarEnsemble(input());

    expect(result.schema_version).toBe("scalar-ensemble/v1");
    expect(result.provenance).toEqual({
      adapter_id: "test-adapter/v1",
      source_schema_version: "source/v2",
      source_provenance: "seed=42",
    });
    expect(result.variables[0]).toMatchObject({ unit: "m/s", category_key: "delivery" });
    expect(result.rows[1].values.carry).toBeNull();
    expect(result.cohorts.map(({ key }) => key)).toEqual(["complete", "failed"]);
    expect(Object.isFrozen(result)).toBe(true);
    expect(Object.isFrozen(result.rows[0].values)).toBe(true);
    expect(JSON.parse(JSON.stringify(result))).toEqual({
      schema_version: "scalar-ensemble/v1",
      ...input(),
    });
  });

  it("builds only paired-finite points with exact availability by cohort", () => {
    const scatter = buildScalarEnsembleScatter(createScalarEnsemble(input()), "speed", "carry");

    expect(scatter.points).toEqual([
      {
        row_id: "series:baseline/trial:0", trial_index: 0, series_id: "baseline",
        cohort: "complete", x: 30, y: 27.4,
      },
      {
        row_id: "series:alternative/trial:0", trial_index: 0, series_id: "alternative",
        cohort: "complete", x: 29, y: 26.8,
      },
    ]);
    expect(scatter.availability).toEqual({
      overall: { total_rows: 4, x_finite: 3, y_finite: 3, paired_finite: 2, unavailable: 2 },
      by_cohort: {
        complete: { total_rows: 3, x_finite: 2, y_finite: 3, paired_finite: 2, unavailable: 1 },
        failed: { total_rows: 1, x_finite: 1, y_finite: 0, paired_finite: 0, unavailable: 1 },
      },
    });
  });

  it("enforces composite identity and exact variable coverage", () => {
    const base = input();
    const mismatchedId = {
      ...base, rows: [{ ...base.rows[0], row_id: "trial:0" }, ...base.rows.slice(1)],
    };
    expect(() => createScalarEnsemble(mismatchedId)).toThrow(/row_id.*composite identity/);

    const duplicate = { ...base, rows: [...base.rows.slice(0, 3), { ...base.rows[0] }] };
    expect(() => createScalarEnsemble(duplicate)).toThrow(/row_id.*unique/);

    const missingValue = {
      ...base, rows: [{ ...base.rows[0], values: { speed: 30 } }, ...base.rows.slice(1)],
    };
    expect(() => createScalarEnsemble(missingValue)).toThrow(/values.*variable keys/);
  });

  it("uses one RFC 3986 composite identifier algorithm across runtimes", () => {
    expect(scalarEnsembleRowId(7, "wedge/α!*"))
      .toBe("series:wedge%2F%CE%B1%21%2A/trial:7");
    expect(scalarEnsembleRowId(7)).toBe("trial:7");
  });

  it("rejects nonfinite raw scalars and invalid definitions", () => {
    const base = input();
    const nonfinite = {
      ...base,
      rows: [{
        ...base.rows[0], values: { ...base.rows[0].values, carry: Number.NaN },
      }, ...base.rows.slice(1)],
    };
    expect(() => createScalarEnsemble(nonfinite)).toThrow(/finite or null/);

    const duplicateStage = {
      ...base, stages: [base.stages[0], { ...base.stages[0] }],
    };
    expect(() => createScalarEnsemble(duplicateStage)).toThrow(/stage keys must be unique/);

    const emptyAttribute = {
      ...base,
      rows: [{
        ...base.rows[0], attributes: { status: "" },
      }, ...base.rows.slice(1)],
    };
    expect(() => createScalarEnsemble(emptyAttribute)).toThrow(/attribute.*nonempty/);
  });
});
