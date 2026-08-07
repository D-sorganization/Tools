import { describe, expect, it } from "vitest";

import {
  inferPortableModel,
  parsePortableModelBundle,
  type PortableModelBundle,
} from "./neuralModelBundle";

const bundle: PortableModelBundle = {
  schema: "launch-monitor-neural-bundle/v1",
  modelId: "trackman-carry-v1",
  vendor: "TrackMan",
  createdAt: "2026-08-06T00:00:00Z",
  features: [
    { name: "ball_speed_mph", unit: "mph", mean: 100, scale: 10, min: 80, max: 130 },
    { name: "launch_angle_deg", unit: "deg", mean: 10, scale: 2, min: 0, max: 30 },
  ],
  outputs: [{ name: "carry_yd", unit: "yd", mean: 200, scale: 20 }],
  layers: [
    { activation: "relu", weights: [[1, -1], [-1, 1]], bias: [0, 0] },
    { activation: "linear", weights: [[2, 3]], bias: [0.5] },
  ],
  metrics: [{ model: "neural_surrogate", target: "carry_yd", split: "test", mae: 3, rmse: 4.2, r2: 0.9 }] as never,
  learningCurve: [{ training_fraction: 1, training_rows: 100, validation_standardized_rmse: 0.2 }] as never,
  provenance: {
    datasetSha256: "a".repeat(64), rowCount: 8860,
  } as never,
};

describe("portable neural model bundle", () => {
  it("validates and executes standardized dense-layer inference", () => {
    const parsed = parsePortableModelBundle(JSON.stringify(bundle));
    expect(inferPortableModel(parsed, {
      ball_speed_mph: 120,
      launch_angle_deg: 8,
    })).toEqual({ carry_yd: 330 });
  });

  it("rejects unsafe dimensions, duplicate JSON fields, and missing inputs", () => {
    const invalid = structuredClone(bundle);
    invalid.layers[0].weights[0] = [1];
    expect(() => parsePortableModelBundle(JSON.stringify(invalid))).toThrow(/dimension/i);
    expect(() => parsePortableModelBundle('{"schema":"x","schema":"y"}'))
      .toThrow(/duplicate JSON field/i);
    expect(() => inferPortableModel(bundle, { ball_speed_mph: 100 }))
      .toThrow(/launch_angle_deg/);
  });
});
