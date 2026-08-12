/** Strict cross-runtime Morris sensitivity report contract tests (#4142). */

import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/morris_global_sensitivity_golden_v1.json";
import {
  MORRIS_REPORT_SCHEMA_ID,
  parseMorrisReport,
  parseMorrisReportJson,
} from "./morrisGlobalSensitivityContract";

const cloneFixture = (): unknown => structuredClone(fixture);
const record = (value: unknown): Record<string, unknown> => value as Record<string, unknown>;
const firstEstimate = (value: unknown): Record<string, unknown> => {
  const estimates = record(value).estimates as unknown[];
  return record(estimates[0]);
};
const effectsOf = (value: unknown): Record<string, unknown> => record(firstEstimate(value).effects);
const estimatesOf = (value: unknown): unknown[] => record(value).estimates as unknown[];

const addSecondTargetMatrix = (value: unknown): void => {
  const estimates = estimatesOf(value);
  for (const sourceEstimate of [...estimates]) {
    const copy = structuredClone(sourceEstimate);
    record(record(copy).target).name = "clubhead_y_m";
    estimates.push(copy);
  }
};

describe("Morris global-sensitivity report parity", () => {
  it("parses the Python golden fixture with complete typed provenance", () => {
    const report = parseMorrisReport(cloneFixture());

    expect(report.schemaId).toBe(MORRIS_REPORT_SCHEMA_ID);
    expect(report.schemaVersion).toBe(1);
    expect(report.method).toBe("morris-elementary-effects");
    expect(report.design).toEqual({
      trajectories: 12,
      levels: 4,
      seed: 73,
      totalSamples: 36,
      normalizedStep: 2 / 3,
    });
    expect(report.estimates[0]).toMatchObject({
      availability: "available",
      sampleAdequacy: "adequate",
      source: {
        unit: "deg",
        bounds: [0, 1],
        timeWindowS: [0.01, 0.02],
        pointIds: ["clubhead"],
      },
      target: {
        unit: "m",
        kind: "state-point",
        timeS: 0.03,
        pointId: "clubhead",
        coordinateFrame: "app_frame:x_target,y_up,z_right",
      },
    });
    expect(Object.isFrozen(report)).toBe(true);
    expect(Object.isFrozen(report.design)).toBe(true);
    expect(Object.isFrozen(report.assumptions)).toBe(true);
    expect(Object.isFrozen(report.estimates)).toBe(true);
    expect(Object.isFrozen(report.estimates[0])).toBe(true);
    expect(Object.isFrozen(report.estimates[0].source)).toBe(true);
    expect(Object.isFrozen(report.estimates[0].source.bounds)).toBe(true);
    expect(Object.isFrozen(report.estimates[0].source.timeWindowS)).toBe(true);
    expect(Object.isFrozen(report.estimates[0].source.pointIds)).toBe(true);
    expect(Object.isFrozen(report.estimates[0].target)).toBe(true);
    expect(Object.isFrozen(report.estimates[0].effects)).toBe(true);
    expect(Object.isFrozen(report.estimates[0].denominator)).toBe(true);
  });

  it("accepts explicit unavailable null estimates with retained denominators", () => {
    const payload = record(cloneFixture());
    const estimate = firstEstimate(payload);
    estimate.availability = "insufficient-data";
    estimate.sample_adequacy = "insufficient";
    estimate.effects = {
      mu: null, mu_star: null, mu_star_standard_error: null, sigma: null,
    };
    estimate.denominator = {
      total_pairs: 12,
      valid_pairs: 1,
      typed_no_impact_pairs: 5,
      no_impact_unavailable_pairs: 4,
      failed_pairs: 3,
      nonfinite_pairs: 4,
    };

    expect(parseMorrisReport(payload).estimates[0].effects.muStar).toBeNull();
  });

  it("parses JSON text without coercing contract values", () => {
    const report = parseMorrisReportJson(JSON.stringify(fixture));
    expect(report.estimates).toHaveLength(2);
    expect(() => parseMorrisReportJson("not-json")).toThrow("valid JSON");
  });

  it("accepts a null-prototype record but rejects class and custom prototypes", () => {
    const plainPayload = record(cloneFixture());
    const nullPrototype = Object.assign(Object.create(null) as Record<string, unknown>, plainPayload);
    expect(parseMorrisReport(nullPrototype).schemaVersion).toBe(1);

    class ReportPayload {}
    const classPayload = Object.assign(new ReportPayload(), plainPayload);
    expect(() => parseMorrisReport(classPayload)).toThrow("plain object");

    const customDesign = Object.assign(Object.create({ inherited: true }) as Record<string, unknown>, record(plainPayload.design));
    plainPayload.design = customDesign;
    expect(() => parseMorrisReport(plainPayload)).toThrow("plain object");
  });

  it.each([
    ["schema ID", (root: Record<string, unknown>) => { root.schema_id = "other"; }],
    ["schema version", (root: Record<string, unknown>) => { root.schema_version = 2; }],
    ["method", (root: Record<string, unknown>) => { root.method = "sobol"; }],
    ["unknown root field", (root: Record<string, unknown>) => { root.extra = true; }],
    ["non-array estimates", (root: Record<string, unknown>) => { root.estimates = {}; }],
  ])("rejects malformed %s", (_name, mutate) => {
    const payload = record(cloneFixture());
    mutate(payload);
    expect(() => parseMorrisReport(payload)).toThrow();
  });

  it.each([
    ["availability", (item: Record<string, unknown>) => { item.availability = "partial"; }],
    ["adequacy", (item: Record<string, unknown>) => { item.sample_adequacy = "maybe"; }],
    ["unknown estimate field", (item: Record<string, unknown>) => { item.status = "ok"; }],
    ["non-finite effect", (item: Record<string, unknown>) => {
      record(item.effects).mu = Number.POSITIVE_INFINITY;
    }],
    ["negative uncertainty", (item: Record<string, unknown>) => {
      record(item.effects).sigma = -1;
    }],
    ["mu-star below absolute mean", (item: Record<string, unknown>) => {
      record(item.effects).mu_star = 1;
    }],
    ["unavailable finite estimates", (item: Record<string, unknown>) => {
      item.availability = "insufficient-data";
      item.sample_adequacy = "insufficient";
    }],
  ])("rejects invalid estimate %s", (_name, mutate) => {
    const payload = cloneFixture();
    mutate(firstEstimate(payload));
    expect(() => parseMorrisReport(payload)).toThrow();
  });

  it("rejects inconsistent source provenance for a repeated factor ID", () => {
    const payload = record(cloneFixture());
    const estimates = payload.estimates as unknown[];
    record(record(estimates[1]).source).spec_id = "face-window";
    expect(() => parseMorrisReport(payload)).toThrow("source provenance");
  });

  it("rejects inconsistent target provenance and duplicate source/target pairs", () => {
    const inconsistent = record(cloneFixture());
    record(record(estimatesOf(inconsistent)[1]).target).unit = "ft";
    expect(() => parseMorrisReport(inconsistent)).toThrow("target provenance");

    const duplicate = record(cloneFixture());
    estimatesOf(duplicate).push(structuredClone(estimatesOf(duplicate)[0]));
    expect(() => parseMorrisReport(duplicate)).toThrow("pairs must be unique");
  });

  it("requires the complete source-by-target estimate matrix", () => {
    const payload = record(cloneFixture());
    addSecondTargetMatrix(payload);
    expect(parseMorrisReport(payload).estimates).toHaveLength(4);
    estimatesOf(payload).pop();
    expect(() => parseMorrisReport(payload)).toThrow("every source/target");
  });

  it.each(["\u0000", "\u001f", "\u007f", "\u009f"])(
    "rejects control character U+%s in stable identifiers",
    (control) => {
      const payload = cloneFixture();
      record(firstEstimate(payload).source).spec_id = `face${control}window`;
      expect(() => parseMorrisReport(payload)).toThrow("control characters");
    },
  );

  it("rejects the former NUL-delimited composite-ID collision", () => {
    const payload = record(cloneFixture());
    const estimates = estimatesOf(payload);
    record(record(estimates[0]).source).spec_id = "a\u0000b";
    record(record(estimates[0]).target).name = "c";
    record(record(estimates[1]).source).spec_id = "a";
    record(record(estimates[1]).target).name = "b\u0000c";
    expect(() => parseMorrisReport(payload)).toThrow("control characters");
  });

  it.each([
    ["missing source field", (item: Record<string, unknown>) => { delete record(item.source).unit; }],
    ["excess target field", (item: Record<string, unknown>) => { record(item.target).extra = true; }],
    ["missing effect field", (item: Record<string, unknown>) => { delete record(item.effects).sigma; }],
    ["excess denominator field", (item: Record<string, unknown>) => { record(item.denominator).extra = 0; }],
  ])("rejects nested shape with %s", (_name, mutate) => {
    const payload = cloneFixture();
    mutate(firstEstimate(payload));
    expect(() => parseMorrisReport(payload)).toThrow("fields do not match");
  });

  it("accepts a numerically possible nonconstant metric tuple", () => {
    const payload = cloneFixture();
    const effects = effectsOf(payload);
    effects.mu = 0.5;
    effects.mu_star = 1.5;
    effects.sigma = Math.sqrt(27 / 11);
    effects.mu_star_standard_error = Math.sqrt(3 / 11 / 12);
    expect(parseMorrisReport(payload).estimates[0].effects.muStar).toBe(1.5);
  });

  it.each([
    ["variance identity", (effects: Record<string, unknown>) => { effects.sigma = 0.25; }],
    ["zero sigma mean magnitude", (effects: Record<string, unknown>) => { effects.mu_star = 3; }],
    ["zero sigma standard error", (effects: Record<string, unknown>) => { effects.mu_star_standard_error = 0.01; }],
  ])("rejects impossible Morris metric relationship: %s", (_name, mutate) => {
    const payload = cloneFixture();
    mutate(effectsOf(payload));
    expect(() => parseMorrisReport(payload)).toThrow("metric");
  });

  it("rejects an available all-zero tuple and accepts the constant-output state", () => {
    const payload = cloneFixture();
    const effects = effectsOf(payload);
    effects.mu = 0;
    effects.mu_star = 0;
    expect(() => parseMorrisReport(payload)).toThrow("constant-output");
    firstEstimate(payload).availability = "constant-output";
    expect(parseMorrisReport(payload).estimates[0].availability).toBe("constant-output");
  });

  it("uses a bounded floating tolerance for the metric identity", () => {
    const payload = cloneFixture();
    effectsOf(payload).sigma = 1e-8;
    expect(parseMorrisReport(payload).estimates[0].effects.sigma).toBe(1e-8);
    effectsOf(payload).sigma = 1e-6;
    expect(() => parseMorrisReport(payload)).toThrow("metric identity");
  });

  it.each([
    ["denominator sum", (item: Record<string, unknown>) => {
      record(item.denominator).failed_pairs = 1;
    }],
    ["trajectory denominator", (item: Record<string, unknown>) => {
      record(item.denominator).total_pairs = 11;
    }],
    ["typed miss subset", (item: Record<string, unknown>) => {
      record(item.denominator).typed_no_impact_pairs = 0;
      record(item.denominator).no_impact_unavailable_pairs = 1;
    }],
  ])("rejects broken %s invariant", (_name, mutate) => {
    const payload = cloneFixture();
    mutate(firstEstimate(payload));
    expect(() => parseMorrisReport(payload)).toThrow("denominator");
  });

  it.each([
    ["bounds", (item: Record<string, unknown>) => {
      record(item.source).bounds = [1, 1];
    }],
    ["source time locus", (item: Record<string, unknown>) => {
      record(item.source).time_window_s = [0.02, 0.01];
    }],
    ["duplicate source points", (item: Record<string, unknown>) => {
      record(item.source).point_ids = ["clubhead", "clubhead"];
    }],
    ["state-point frame", (item: Record<string, unknown>) => {
      record(item.target).coordinate_frame = null;
    }],
    ["trimmed unit", (item: Record<string, unknown>) => {
      record(item.target).unit = " m";
    }],
  ])("rejects invalid provenance %s", (_name, mutate) => {
    const payload = cloneFixture();
    mutate(firstEstimate(payload));
    expect(() => parseMorrisReport(payload)).toThrow();
  });

  it.each([
    ["odd levels", (design: Record<string, unknown>) => { design.levels = 5; }],
    ["negative seed", (design: Record<string, unknown>) => { design.seed = -1; }],
    ["sample count", (design: Record<string, unknown>) => { design.total_samples = 35; }],
    ["grid step", (design: Record<string, unknown>) => { design.normalized_step = 0.5; }],
  ])("rejects invalid design provenance %s", (_name, mutate) => {
    const payload = record(cloneFixture());
    mutate(record(payload.design));
    expect(() => parseMorrisReport(payload)).toThrow();
  });
});
