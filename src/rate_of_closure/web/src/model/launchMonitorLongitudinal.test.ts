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

  it("requires attested identities and unique per-player session order", () => {
    expect(() => analyzeLongitudinalPerformance(rows, {
      metricColumn: "speed", sessionColumn: "session", sessionOrderColumn: "order",
      playerColumn: "player", playerIdentityAttested: false, sessionIdentityAttested: true,
      higherIsBetter: true, confidenceLevel: 0.95, minSessions: 3,
    })).toThrow(/attested/);
  });
});
