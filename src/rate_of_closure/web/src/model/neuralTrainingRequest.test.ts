import { describe, expect, it } from "vitest";

import { createTrainingRequest } from "./neuralTrainingRequest";

describe("neural training request", () => {
  it("creates a portable, configurable request without embedding private rows", () => {
    const request = createTrainingRequest({
      vendor: "Foresight",
      dataset: { fileName: "custom.csv", rowCount: 40, columns: ["speed", "carry"] },
      featureColumns: ["speed"],
      outputColumns: ["carry"],
      hiddenLayers: [32, 16],
      activation: "relu",
      alpha: 0.0001,
      epochs: 100,
      learningRate: 0.002,
      validationFraction: 0.2,
      randomSeed: 42,
    });
    expect(request.schema).toBe("launch-monitor-neural-training/v1");
    expect(request.dataset).toEqual({ fileName: "custom.csv", rowCount: 40, columns: ["speed", "carry"] });
    expect(JSON.stringify(request)).not.toContain("privateRows");
  });

  it("rejects leakage and invalid hyperparameters", () => {
    expect(() => createTrainingRequest({
      vendor: "TrackMan",
      dataset: { fileName: "shots.csv", rowCount: 10, columns: ["carry"] },
      featureColumns: ["carry"], outputColumns: ["carry"], hiddenLayers: [8], activation: "relu", alpha: 0.0001,
      epochs: 10, learningRate: 0.01, validationFraction: 0.2, randomSeed: 1,
    })).toThrow(/both feature and output/i);
  });
});
