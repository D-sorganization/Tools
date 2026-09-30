import { describe, expect, it } from "vitest";

import { analyzeLongitudinalPerformance } from "./launchMonitorLongitudinal";

const rows = [
  ...[100, 103, 104, 108].flatMap((mean, index) => [-0.5, 0.5].map((offset) => ({
    player: "p1", session: `p1-${index + 1}`, order: index + 1, speed: mean + offset,
  }))),
  ...[90, 91, 94, 94].flatMap((mean, index) => [-0.5, 0.5].map((offset) => ({
    player: "p2", session: `p2-${index + 1}`, order: index + 1, speed: mean + offset,
  }))),
];

const directResidualStandardError = (values: number[]): number => {
  const xMean = 2.5;
  const yMean = values.reduce((sum, value) => sum + value, 0) / values.length;
  let sxx = 0;
  let sxy = 0;
  for (let index = 0; index < values.length; index += 1) {
    const dx = index + 1 - xMean;
    sxx += dx * dx;
    sxy += dx * (values[index] - yMean);
  }
  const slope = sxy / sxx;
  const intercept = yMean - slope * xMean;
  const residualSum = values.reduce((sum, value, index) =>
    sum + (value - intercept - slope * (index + 1)) ** 2, 0,
  );
  return Math.sqrt(residualSum / (values.length - 2) / sxx);
};

describe("launch monitor longitudinal performance", () => {
  it("reports session uncertainty, player slopes, and population synthesis", () => {
    const result = analyzeLongitudinalPerformance(rows, {
      metricColumn: "speed", sessionColumn: "session", sessionOrderColumn: "order",
      playerColumn: "player", playerIdentityAttested: true, sessionIdentityAttested: true,
      higherIsBetter: true, confidenceLevel: 0.95, minSessions: 3,
    });
    expect(result.sessionPoints).toHaveLength(8);
    expect(result.sessionPoints[0].standardError).toBeCloseTo(0.5);
    expect(result.players.every((player) => player.status === "ok")).toBe(true);
    expect(result.population.contributorCount).toBe(2);
    expect(result.population.randomEffectSlope).toBeGreaterThan(0);
    expect(result.population.improvementProbability).toBeGreaterThan(0.5);
  });

  it("retains representable residual noise in near-perfect public regressions", () => {
    const valuesByPlayer = {
      p1: [1_000_001_000_000, 1_000_002_000_000.0005, 1_000_002_999_999.9995, 1_000_004_000_000],
      p2: [2_000_002_000_000, 2_000_004_000_000.001, 2_000_005_999_999.999, 2_000_008_000_000],
    };
    const noisyRows = Object.entries(valuesByPlayer).flatMap(([player, values]) =>
      values.map((speed, index) => ({ player, session: `${player}-${index + 1}`, order: index + 1, speed })),
    );
    const expectedStandardErrors = Object.values(valuesByPlayer).map(directResidualStandardError);

    expect(expectedStandardErrors[0]).toBeCloseTo(0.00020716016510533694, 14);

    const result = analyzeLongitudinalPerformance(noisyRows, {
      metricColumn: "speed", sessionColumn: "session", sessionOrderColumn: "order",
      playerColumn: "player", playerIdentityAttested: true, sessionIdentityAttested: true,
      higherIsBetter: true, confidenceLevel: 0.95, minSessions: 3,
    });

    result.players.forEach((player, index) => {
      expect(player.standardError).toBeCloseTo(expectedStandardErrors[index], 14);
      expect(player.ciLower).toBeLessThan(player.slopePerSession!);
      expect(player.ciUpper).toBeGreaterThan(player.slopePerSession!);
      expect(player.pValue).toBeLessThan(0.05);
    });
    expect(result.population.contributorCount).toBe(2);
  });

  it("requires attested identities and unique per-player session order", () => {
    expect(() => analyzeLongitudinalPerformance(rows, {
      metricColumn: "speed", sessionColumn: "session", sessionOrderColumn: "order",
      playerColumn: "player", playerIdentityAttested: false, sessionIdentityAttested: true,
      higherIsBetter: true, confidenceLevel: 0.95, minSessions: 3,
    })).toThrow(/attested/);
  });
});
