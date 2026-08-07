import { describe, expect, it } from "vitest";

import { importedAdvancedResults } from "./launchMonitorImportedResults";

describe("imported advanced campaign results", () => {
  it("recognizes PCA scores and long-form loadings", () => {
    const result = importedAdvancedResults([
      { shot_id: "a", feature: "spin", component: "PC1", loading: -0.8, pc1: 1.2, pc2: -0.4 },
      { shot_id: "b", feature: "speed", component: "PC1", loading: 0.5, pc1: -0.7, pc2: 0.2 },
    ]);
    expect(result.pcaScores).toHaveLength(2);
    expect(result.pcaLoadings[0]).toMatchObject({ label: "spin · PC1", rank: 1, value: -0.8 });
  });

  it("recognizes ranked importance, residual fields, and held-out metrics", () => {
    const result = importedAdvancedResults([
      { feature: "spin", importance: 0.7, rank: 1, method: "permutation", held_out_r2: 0.61, model_spread_m: 4.2 },
      { feature: "speed", importance: 0.2, rank: 2, method: "permutation", held_out_r2: 0.61, model_spread_m: 1.8 },
    ]);
    expect(result.featureImportance.map((item) => item.label)).toEqual(["spin", "speed"]);
    expect(result.performance).toEqual([{ metric: "held_out_r2", value: 0.61, method: "permutation" }]);
    expect(result.residualColumns).toEqual(["model_spread_m"]);
  });
});

