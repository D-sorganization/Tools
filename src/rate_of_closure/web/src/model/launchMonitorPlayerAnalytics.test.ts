import { describe, expect, it } from "vitest";

import {
  dispersionSummary,
  expectedStrokes,
  metricLabel,
  sessionTrend,
  sessionTrendExportRows,
  strokesGainedProxy,
} from "./launchMonitorPlayerAnalytics";
import { parseLaunchMonitorProject, PROJECT_CONTRACT_VERSION } from "./launchMonitorProject";

const rows = [
  { shot_id: "1", session_id: "2026-01", carry_distance: 145, lateral: -10, ball_speed: 140 },
  { shot_id: "2", session_id: "2026-01", carry_distance: 150, lateral: 0, ball_speed: 142 },
  { shot_id: "3", session_id: "2026-02", carry_distance: 155, lateral: 10, ball_speed: 146 },
];

describe("launch monitor player analytics", () => {
  it("reports signed directional dispersion and explicit yards", () => {
    const summary = dispersionSummary(rows);
    expect(summary).not.toBeNull();
    expect(summary?.leftCount).toBe(1);
    expect(summary?.centerCount).toBe(1);
    expect(summary?.rightCount).toBe(1);
    expect(summary?.meanLateralYards).toBeCloseTo(0);
    expect(metricLabel("carry_distance")).toBe("carry distance (yd)");
  });

  it("calculates a transparent Broadie-style proxy from retained shot values", () => {
    const results = strokesGainedProxy(rows, 150);
    expect(results).toHaveLength(3);
    expect(results[1].remainingYards).toBe(0);
    expect(results[1].strokesGainedProxy).toBeCloseTo(expectedStrokes(150) - 1);
    expect(results[0].expectedBefore).toBe(expectedStrokes(150));
  });

  it("summarizes session means and their slope", () => {
    const trend = sessionTrend(rows, "ball_speed");
    expect(trend?.players[0].points.map((point) => point.mean)).toEqual([141, 146]);
    expect(trend?.players[0].slopePerSession).toBeCloseTo(5);
    expect(trend?.players[0].changeFirstToLast).toBe(5);
  });

  it("sequences and fits each player's sessions independently", () => {
    const trend = sessionTrend([
      { player_id: "alpha", session_id: "2", ball_speed: 110 },
      { player_id: "beta", session_id: "2", ball_speed: 190 },
      { player_id: "alpha", session_id: "1", ball_speed: 100 },
      { player_id: "beta", session_id: "1", ball_speed: 200 },
    ], "ball_speed")!;
    expect(trend.players.map((player) => player.playerId)).toEqual(["alpha", "beta"]);
    expect(trend.players[0].points.map((point) => point.order)).toEqual([0, 1]);
    expect(trend.players[0].slopePerSession).toBe(10);
    expect(trend.players[1].slopePerSession).toBe(-10);
    expect(sessionTrendExportRows(trend)).toEqual(expect.arrayContaining([
      expect.objectContaining({ playerId: "alpha", playerSlopePerSession: 10 }),
      expect.objectContaining({ playerId: "beta", playerSlopePerSession: -10 }),
    ]));
  });

  it("validates saved project contracts", () => {
    const project = parseLaunchMonitorProject(JSON.stringify({
      contractVersion: PROJECT_CONTRACT_VERSION,
      savedAt: "2026-08-06T00:00:00Z",
      sourceName: "test.csv",
      rows,
      settings: {
        outcome: "ball_speed", predictors: ["carry_distance"], mode: "comprehensive",
        method: "pearson", missing: "pairwise", groupBy: "session_id",
        confidence: 0.95, minSamples: 3, targetDistanceYards: 150,
      },
    }));
    expect(project.rows).toHaveLength(3);
    expect(() => parseLaunchMonitorProject('{"contractVersion":"0"}')).toThrow(/Unsupported/);
  });
});
