import { describe, expect, it } from "vitest";

import {
  buildComparisonRows,
  exportComparisonCsv,
  exportComparisonJson,
  filterComparisonRows,
} from "./launchMonitorComparisonWorkspace";

describe("launchMonitorComparisonWorkspace model", () => {
  it("builds comparison rows for all 28 parameters", () => {
    const rows = buildComparisonRows();
    expect(rows).toHaveLength(28);
    const ids = rows.map((r) => r.parameterId);
    expect(ids).toContain("club_speed");
    expect(ids).toContain("ball_speed");
    expect(ids).toContain("apex_height");
    expect(ids).toContain("spin_axis");
    expect(ids).toContain("dynamic_lie");
  });

  it("calculates signed deltas only for comparable quantities", () => {
    const tm = {
      ball_speed: 72.5,
      launch_angle: 11.2,
      total_spin: 2600,
      club_path: 3.0,
      launch_direction: 1.5,
      curve: 8.0,
    };
    const fs = {
      ball_speed: 70.0,
      launch_angle: 12.0,
      total_spin: 2500,
      club_path: 2.0,
      launch_direction: 1.5,
      curve: 8.0,
    };

    const rows = buildComparisonRows(tm, fs);
    const byId = Object.fromEntries(rows.map((r) => [r.parameterId, r]));

    // Directly comparable: ball_speed
    expect(byId.ball_speed.isComparable).toBe(true);
    expect(byId.ball_speed.difference).toBeCloseTo(2.5, 2);
    expect(byId.ball_speed.differenceText).toBe("+2.50");
    expect(byId.ball_speed.reasons).toEqual([]);

    // Directly comparable: launch_angle
    expect(byId.launch_angle.isComparable).toBe(true);
    expect(byId.launch_angle.difference).toBeCloseTo(-0.8, 2);
    expect(byId.launch_angle.differenceText).toBe("-0.80");

    // Incompatible: club_path (reference_point, event_time)
    expect(byId.club_path.isComparable).toBe(false);
    expect(byId.club_path.difference).toBeNull();
    expect(byId.club_path.differenceText).toContain("Not comparable");
    expect(byId.club_path.reasons).toContain("reference_point");
    expect(byId.club_path.reasons).toContain("event_time");

    // Incompatible: launch_direction (sign_rule)
    expect(byId.launch_direction.isComparable).toBe(false);
    expect(byId.launch_direction.difference).toBeNull();
    expect(byId.launch_direction.reasons).toContain("sign_rule");

    // Incompatible: curve (availability)
    expect(byId.curve.isComparable).toBe(false);
    expect(byId.curve.difference).toBeNull();
    expect(byId.curve.reasons).toContain("availability");
  });

  it("filters comparison rows by group and search query", () => {
    const rows = buildComparisonRows();

    // Filter by group "ball_launch"
    const launchGroup = filterComparisonRows(rows, "ball_launch", "");
    expect(launchGroup).toHaveLength(4);
    expect(launchGroup.every((r) => r.group === "ball_launch")).toBe(true);

    // Filter by search query "spin"
    const spinSearch = filterComparisonRows(rows, "all", "spin");
    expect(spinSearch.length).toBeGreaterThanOrEqual(4);
    expect(spinSearch.some((r) => r.parameterId === "total_spin")).toBe(true);
  });

  it("exports comparison data to JSON and CSV", () => {
    const rows = buildComparisonRows({ ball_speed: 70, club_speed: 45 }, { ball_speed: 68, club_speed: 46 });

    const jsonStr = exportComparisonJson(rows, "Test Workspace");
    const parsed = JSON.parse(jsonStr);
    expect(parsed.schema_version).toBe("launch-monitor-comparison/v1");
    expect(parsed.source_name).toBe("Test Workspace");
    expect(parsed.total_count).toBe(28);
    expect(parsed.rows).toHaveLength(28);

    const csvStr = exportComparisonCsv(rows);
    expect(csvStr).toContain("Signed Difference (TM - FS)");
    expect(csvStr).toContain("Ball Speed");
    expect(csvStr).toContain("Club Speed");
  });
});
