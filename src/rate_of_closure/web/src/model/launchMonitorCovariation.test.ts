import { describe, expect, it } from "vitest";

import {
  analyzePlayerCovariation,
  rankCovariationPairs,
  type CovariationRequest,
} from "./launchMonitorCovariation";

const rows = [
  ...[0, 1, 2, 3].map((value) => ({ player_id: "alpha", x: value, y: value + 10, z: value * 2 })),
  ...[10, 11, 12, 13].map((value) => ({ player_id: "beta", x: value, y: value - 10, z: value * 2 })),
];

const request: CovariationRequest = {
  xColumn: "x", yColumn: "y", playerColumn: "player_id",
  minSamples: 4, confidenceLevel: 0.95,
};

describe("player covariation", () => {
  it("separates within-player, raw pooled, and between-player relationships", () => {
    const result = analyzePlayerCovariation(rows, request);
    expect(result.perPlayer).toHaveLength(2);
    expect(result.perPlayer.every((player) => player.pearsonR === 1)).toBe(true);
    expect(result.pooledRaw.pearsonR).toBeLessThan(0);
    expect(result.withinPlayerCentered.pearsonR).toBeCloseTo(1);
    expect(result.withinPlayerCentered.ciLower).toBeNull();
    expect(result.betweenPlayer.pearsonR).toBeCloseTo(-1);
    expect(result.meta.fixedEffectR).toBeCloseTo(1);
    expect(result.meta.randomEffectR).toBeCloseTo(1);
    expect(result.meta.iSquaredPct).toBe(0);
    expect(result.backingData[0]).toEqual(expect.objectContaining({
      playerId: "alpha", sourceIndex: 0, x: 0, y: 10, centeredX: -1.5, centeredY: -1.5,
    }));
    expect(result.warnings).toEqual(expect.arrayContaining([expect.stringMatching(/opposite signs/i)]));
  });

  it("reports incomplete players without silently adding them to meta-analysis", () => {
    const result = analyzePlayerCovariation([
      ...rows,
      { player_id: "short", x: 1, y: 2 },
      { player_id: "missing", x: "", y: 3 },
    ], request);
    expect(result.perPlayer.find((player) => player.playerId === "short")?.status)
      .toBe("insufficient_samples");
    expect(result.meta.contributorCount).toBe(2);
    expect(result.completePairCount).toBe(9);
  });

  it("ranks arbitrary numeric pairs and identifies consistent directions", () => {
    const ranking = rankCovariationPairs(rows, {
      columns: ["x", "y", "z"], playerColumn: "player_id",
      minSamples: 4, confidenceLevel: 0.95,
    });
    expect(ranking).toHaveLength(3);
    expect(ranking[0]).toEqual(expect.objectContaining({
      contributorCount: 2, directionConsistency: 1,
    }));
    expect(ranking[0].absoluteRandomEffectR).toBeCloseTo(1);
  });

  it("validates confidence, distinct columns, and player identifiers", () => {
    expect(() => analyzePlayerCovariation(rows, { ...request, xColumn: "y" })).toThrow(/different/);
    expect(() => analyzePlayerCovariation(rows, { ...request, confidenceLevel: 1 })).toThrow(/confidence/i);
    expect(() => analyzePlayerCovariation(rows, { ...request, playerColumn: "unknown" })).toThrow(/player/i);
  });

  it("does not synthesize a one-player meta-effect", () => {
    const result = analyzePlayerCovariation(rows.slice(0, 4), request);
    expect(result.meta.contributorCount).toBe(1);
    expect(result.meta.totalSampleCount).toBe(4);
    expect(result.meta.fixedEffectR).toBeNull();
    expect(result.meta.randomEffectR).toBeNull();
    expect(result.warnings.some((warning) => /opposite signs/i.test(warning))).toBe(false);
  });
});
